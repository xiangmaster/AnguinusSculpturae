"""Trainers for the MamaSynth breast-MRI datasets.

Two families live here. The segmentation trainers are ordinary nnU-Nets fed from
blosc2 volumes instead of a preprocessed folder. The translation trainers take a
pre-contrast slice and predict the residual to the post-contrast slice; they build
up from a plain masked MSE through perceptual, structural, segmentation-based and
adversarial terms, and most of the classes below are combinations of those terms at
different weights and batch sizes.

Both families read their data through :class:`MamaSynthDataMixin`, which is what
lets them run without nnU-Net preprocessing. Paths to the dataset and to the
intensity statistics come from the environment, set by the launcher scripts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
    NonDetMultiThreadedAugmenter,
)
from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)
from torch import distributed as dist

from nnunetv2.training.mamasynth.data_loader import (
    MamaSynthSliceDataLoader,
    MamaSynthTranslationFixedValDataLoader,
    MamaSynthTranslationSliceDataLoader,
)
from nnunetv2.training.mamasynth.dataset import load_cases, load_split
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerCuria import (
    AbstractnnUNetTrainerCuria,
)
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

MAMASYNTH_DATASET_DIR_ENV = "MAMASYNTH_DATASET_DIR"
MAMASYNTH_PRE_STATS_PATH_ENV = "MAMASYNTH_PRE_STATS_PATH"


class MamaSynthResizeConvUNetDecoder(torch.nn.Module):
    """nnU-Net decoder using bilinear resize plus convolution instead of transposed convolution.

    Wraps an existing decoder and reuses its stages, skips and output heads; only the
    upsampling path changes.
    """

    def __init__(self, original_decoder: torch.nn.Module):
        """Adopt the original decoder's modules and replace each transposed convolution.

        Every transposed convolution becomes a 1x1 convolution with the same channel
        counts, applied after an explicit interpolation. Raises TypeError on decoders that
        are not built from 2D transposed convolutions.
        """
        super().__init__()
        if not all(
            isinstance(module, torch.nn.ConvTranspose2d)
            for module in original_decoder.transpconvs
        ):
            raise TypeError(
                "MamaSynthResizeConvUNetDecoder currently supports 2D ConvTranspose2d decoders only."
            )

        self.deep_supervision = original_decoder.deep_supervision
        self.encoder = original_decoder.encoder
        self.num_classes = original_decoder.num_classes
        self.stages = original_decoder.stages
        self.seg_layers = original_decoder.seg_layers
        self.upsample_convs = torch.nn.ModuleList(
            torch.nn.Conv2d(
                module.in_channels,
                module.out_channels,
                kernel_size=1,
                padding=0,
                bias=module.bias is not None,
            )
            for module in original_decoder.transpconvs
        )

    def forward(self, skips):
        """Decode the encoder skips from the bottleneck upward.

        At each stage the running feature map is interpolated to the skip's spatial size,
        projected to the right channel count, concatenated with the skip and passed
        through the stage. Outputs are returned coarse-to-fine reversed, so the
        full-resolution head comes first.
        """
        import torch.nn.functional as F

        low_resolution_input = skips[-1]
        segmentation_outputs = []
        for stage_index, stage in enumerate(self.stages):
            skip = skips[-(stage_index + 2)]
            x = F.interpolate(
                low_resolution_input,
                size=skip.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            x = self.upsample_convs[stage_index](x)
            x = torch.cat((x, skip), dim=1)
            x = stage(x)
            if self.deep_supervision:
                segmentation_outputs.append(self.seg_layers[stage_index](x))
            elif stage_index == len(self.stages) - 1:
                segmentation_outputs.append(self.seg_layers[-1](x))
            low_resolution_input = x

        segmentation_outputs.reverse()
        return (
            segmentation_outputs if self.deep_supervision else segmentation_outputs[0]
        )

    def compute_conv_feature_map_size(self, input_size):
        """Estimate the decoder's feature-map memory for a given input size.

        Mirrors the interface nnU-Net uses to size the network against a memory budget.
        """
        skip_sizes = []
        for stage_stride in self.encoder.strides[:-1]:
            skip_sizes.append(
                [size // stride for size, stride in zip(input_size, stage_stride)]
            )
            input_size = skip_sizes[-1]

        output = np.int64(0)
        for stage_index, stage in enumerate(self.stages):
            skip_size = skip_sizes[-(stage_index + 1)]
            output += stage.compute_conv_feature_map_size(skip_size)
            output += np.prod(
                [self.upsample_convs[stage_index].out_channels, *skip_size],
                dtype=np.int64,
            )
            if self.deep_supervision or stage_index == len(self.stages) - 1:
                output += np.prod([self.num_classes, *skip_size], dtype=np.int64)
        return output


class MamaSynthPatchDiscriminator(torch.nn.Module):
    """Patch discriminator over the post-contrast image plus a support channel.

    Scores overlapping patches rather than the whole image, so its output is a map of
    real/fake logits instead of a single number.
    """

    def __init__(
        self, input_channels: int = 2, base_channels: int = 32, dropout: float = 0.3
    ):
        """Stack strided convolutions with spectral normalization, activations and dropout.

        Three stride-2 blocks shrink the input before two stride-1 blocks, the last of
        which reduces to a single logit channel.
        """
        super().__init__()

        def convolution(
            in_channels: int, out_channels: int, stride: int
        ) -> torch.nn.Module:
            """One spectrally normalised convolution, which bounds the discriminator's Lipschitz constant."""
            return torch.nn.utils.spectral_norm(
                torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=4, stride=stride, padding=1
                )
            )

        self.network = torch.nn.Sequential(
            convolution(input_channels, base_channels, 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Dropout2d(dropout),
            convolution(base_channels, base_channels * 2, 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Dropout2d(dropout),
            convolution(base_channels * 2, base_channels * 4, 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Dropout2d(dropout),
            convolution(base_channels * 4, base_channels * 8, 1),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Dropout2d(dropout),
            convolution(base_channels * 8, 1, 1),
        )

    def forward(self, image_and_mask: torch.Tensor) -> torch.Tensor:
        """Score every patch of the stacked image and support channel."""
        return self.network(image_and_mask)


class MamaSynthDataMixin:
    """Replaces nnU-Net's preprocessed dataloading with on-the-fly blosc2 slice sampling.

    Slices are resized to a fixed canvas and z-scored individually in the data loader.
    nnU-Net's spatial and intensity augmentation and its deep-supervision target
    downsampling are kept by reusing the standard transform pipeline; the training
    loop, loss, logging and checkpointing are inherited untouched.
    """

    def __init__(
        self, plans, configuration, fold, dataset_json, device=torch.device("cuda")
    ):
        """Locate the dataset directory from the environment and load the case list.

        Raises RuntimeError when the variable is unset, which happens if the trainer is
        started directly rather than through the launcher that sets it.
        """
        super().__init__(plans, configuration, fold, dataset_json, device)
        dataset_dir = os.environ.get(MAMASYNTH_DATASET_DIR_ENV)
        if dataset_dir is None:
            raise RuntimeError(
                f"{MAMASYNTH_DATASET_DIR_ENV} is not set. Launch training via run_training_mamasynth, "
                "which points the trainer at the blosc2 dataset directory."
            )
        self.dataset_dir = Path(dataset_dir)
        # Point the base trainer's preprocessed-folder attribute somewhere valid; nothing
        # is ever read from it.
        self.preprocessed_dataset_folder = str(self.dataset_dir)
        self.cases = self._load_mamasynth_cases(self.dataset_dir)

    batch_size_override = None

    def _do_i_compile(self):
        """Disable torch.compile for these trainers."""
        return False

    @staticmethod
    def _load_mamasynth_cases(dataset_dir: Path):
        """Read the case list for the dataset; subclasses point this at their own reader."""
        return load_cases(dataset_dir)

    def _set_batch_size_and_oversample(self):
        """Apply ``batch_size_override`` to the configuration before the base setup runs."""
        if self.batch_size_override is not None:
            self.configuration_manager.configuration["batch_size"] = int(
                self.batch_size_override
            )
        super()._set_batch_size_and_oversample()

    def initialize(self):
        """Run the base initialization with the preprocessed-folder probe stubbed out."""
        import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as base_module
        from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2

        # The base trainer infers a dataset class from the preprocessed folder, which does
        # not exist here. The result is never used, so the single call is neutralised for
        # the duration of initialize().
        original = base_module.infer_dataset_class
        base_module.infer_dataset_class = lambda folder: nnUNetDatasetBlosc2
        try:
            super().initialize()
        finally:
            base_module.infer_dataset_class = original

    def on_train_start(self):
        """Base start-up sequence minus the steps that read the preprocessed folder.

        Builds the loaders, writes plans and dataset JSON, and plots the architecture,
        skipping dataset unpacking and the fingerprint copy.
        """
        from batchgenerators.utilities.file_and_folder_operations import (
            join,
            maybe_mkdir_p,
            save_json,
        )

        from nnunetv2.utilities.helpers import empty_cache

        if not self.was_initialized:
            self.initialize()
        self.dataloader_train, self.dataloader_val = self.get_dataloaders()
        maybe_mkdir_p(self.output_folder)
        self.set_deep_supervision_enabled(self.enable_deep_supervision)
        self.print_plans()
        empty_cache(self.device)
        save_json(
            self.plans_manager.plans,
            join(self.output_folder_base, "plans.json"),
            sort_keys=False,
        )
        save_json(
            self.dataset_json,
            join(self.output_folder_base, "dataset.json"),
            sort_keys=False,
        )
        self.plot_network_architecture()
        self._save_debug_information()

    def on_train_epoch_end(self, train_outputs):
        """Finish the epoch and refresh the example-batch preview."""
        super().on_train_epoch_end(train_outputs)
        if self.local_rank == 0:
            try:
                self._save_example_batch_png()
            except Exception as exc:
                self.print_to_log_file(f"Could not save example batch png: {exc}")

    def _save_example_batch_png(self, num_examples: int = 4):
        """Write augmented image and mask pairs exactly as the network receives them.

        Overwritten every epoch.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from batchgenerators.utilities.file_and_folder_operations import join

        batch = next(self.dataloader_train)
        data = batch["data"]
        seg = (
            batch["target"][0] if isinstance(batch["target"], list) else batch["target"]
        )
        keys = batch.get("keys", [""] * data.shape[0])
        n = min(num_examples, data.shape[0])

        fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n), squeeze=False)
        for i in range(n):
            image = data[i, 0].detach().cpu().float().numpy()
            mask = seg[i, 0].detach().cpu().numpy()
            axes[i, 0].imshow(image, cmap="gray")
            axes[i, 0].set_title(str(keys[i]), fontsize=7)
            axes[i, 1].imshow(image, cmap="gray")
            axes[i, 1].imshow(
                np.ma.masked_where(mask == 0, mask),
                cmap="autumn",
                alpha=0.6,
                vmin=1,
                vmax=1,
            )
            axes[i, 1].set_title("image + lesion mask", fontsize=8)
            for ax in axes[i]:
                ax.axis("off")
        fig.suptitle(
            f"epoch {self.current_epoch} - augmented training inputs", fontsize=9
        )
        fig.tight_layout()
        fig.savefig(join(self.output_folder, "train_examples.png"), dpi=100)
        plt.close(fig)

    def do_split(self):
        """Return the train and validation case ids for the current fold."""
        split = load_split(self.dataset_dir, self.fold)
        return split["train"], split["val"]

    def get_dataloaders(self):
        """Build augmented train and validation loaders over sampled slices.

        Keeps the standard nnU-Net transform pipeline but swaps in the slice loader, which
        applies the transforms itself rather than leaving them to the augmenter. Training
        uses worker processes where the environment allows; validation stays
        single-threaded.
        """
        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        rotation_for_DA, do_dummy_2d, _, mirror_axes = (
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )

        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=(
                self.label_manager.foreground_regions
                if self.label_manager.has_regions
                else None
            ),
            ignore_label=self.label_manager.ignore_label,
        )
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=(
                self.label_manager.foreground_regions
                if self.label_manager.has_regions
                else None
            ),
            ignore_label=self.label_manager.ignore_label,
        )

        tr_keys, val_keys = self.do_split()
        tr_cases = [c for c in self.cases if c.case_id in set(tr_keys)]
        val_cases = [c for c in self.cases if c.case_id in set(val_keys)]

        dl_tr = MamaSynthSliceDataLoader(
            tr_cases,
            self.batch_size,
            self.oversample_foreground_percent,
            transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
        )
        dl_val = MamaSynthSliceDataLoader(
            val_cases,
            self.batch_size,
            self.oversample_foreground_percent,
            transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
        )

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr,
                transform=None,
                num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        _ = next(mt_gen_train)
        return mt_gen_train, mt_gen_val


class nnUNetTrainerMamaSynth(MamaSynthDataMixin, nnUNetTrainer):
    """Standard nnU-Net baseline segmenting lesion and breast from pre-contrast slices."""

    def __init__(
        self, plans, configuration, fold, dataset_json, device=torch.device("cuda")
    ):
        """Fix the number of training and validation iterations per epoch."""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_iterations_per_epoch = 1000
        self.num_val_iterations_per_epoch = 200


class nnUNetTrainerMamaSynth_BS_64(nnUNetTrainerMamaSynth):
    """Segmentation baseline at batch size 64."""

    batch_size_override = 64


class nnUNetTrainerMamaSynth_BS_256(nnUNetTrainerMamaSynth):
    """Segmentation baseline at batch size 256."""

    batch_size_override = 256


class nnUNetTrainerMamaSynth_BS_128(nnUNetTrainerMamaSynth):
    """Segmentation baseline at batch size 128."""

    batch_size_override = 128


class nnUNetTrainerMamaSynthTranslation(MamaSynthDataMixin, nnUNetTrainer):
    """Pre-to-post image translation: take the pre-contrast slice, predict the delta.

    The network output is added to the input to form the synthetic post-contrast
    image, and the loss is a masked MSE on the delta. This is the root of the
    translation family; the subclasses add further loss terms on top.
    """

    batch_size_override = 16

    def __init__(
        self, plans, configuration, fold, dataset_json, device=torch.device("cuda")
    ):
        """Set the schedule, optimizer hyperparameters and checkpoint-selection state.

        Deep supervision is switched off, since there is a single full-resolution output,
        and the intensity statistics used to normalise the images are loaded here.
        """
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 100
        self.num_epochs = 150
        # One full-resolution output only, so there are no auxiliary heads to supervise.
        self.enable_deep_supervision = False
        self.initial_lr = 1e-3
        self.weight_decay = 1e-4
        self.pre_stats = self._load_pre_stats()
        self.validation_objective_ema_decay = 0.8
        self.validation_objective_ema = None
        self.best_raw_validation_objective = None
        self._best_ema = None
        self.validation_mse_ema = None
        self.best_validation_mse_ema = None
        self.validation_dice_ema = None
        self.best_validation_dice_ema = None

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=True,
    ):
        """Build the standard U-Net with a single output channel and no deep supervision."""
        return nnUNetTrainer.build_network_architecture(
            plans_manager,
            configuration_manager,
            num_input_channels,
            1,
            enable_deep_supervision=False,
        )

    def _load_pre_stats(self) -> dict:
        """Load the dataset-wide intensity mean and standard deviation.

        These normalise every slice consistently, so predicted deltas are comparable
        across cases. Raises if the path is unset or the file lacks either statistic.
        """
        stats_path = os.environ.get(MAMASYNTH_PRE_STATS_PATH_ENV)
        if stats_path is None:
            raise RuntimeError(
                f"{MAMASYNTH_PRE_STATS_PATH_ENV} is not set. Launch with nnUNetv2_train_mamasynth_translate "
                "or set it to training_pre_stats.json."
            )
        with Path(stats_path).open("r", encoding="utf-8") as handle:
            stats = json.load(handle)
        if "mean" not in stats or "std" not in stats:
            raise ValueError(f"{stats_path} must contain mean and std.")
        return {
            "mean": float(stats["mean"]),
            "std": float(stats["std"]),
            "path": str(stats_path),
        }

    def _build_loss(self):
        """Unused placeholder; the objective is assembled in the train and validation steps."""
        return torch.nn.MSELoss(reduction="none")

    def configure_optimizers(self):
        """AdamW with a polynomial LR decay over the full run."""
        from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler

        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
        )
        scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, scheduler

    def _ensure_translation_logger_keys(self):
        """Create the logging series this trainer family writes to, if absent.

        The logger is keyed by name and does not tolerate gaps, so every series has to
        exist before the first write and after a resume. Descriptive names are kept
        alongside the original short ones, back-filled from them so that runs started
        under the old names keep their history and their plots.
        """
        local_logger = getattr(self.logger, "local_logger", None)
        if local_logger is None:
            return
        logging_dict = getattr(local_logger, "my_fantastic_logging", None)
        if logging_dict is None:
            return
        logging_dict.setdefault("train_baseline_mses", [])
        logging_dict.setdefault("val_baseline_mses", [])
        logging_dict.setdefault("train_objective_losses", [])
        logging_dict.setdefault("val_objective_losses", [])
        # Descriptive names logged alongside the original short keys. Each is seeded from
        # its short counterpart so runs started before the rename keep an unbroken history.
        aliases = {
            "train_post_mses": "train_losses",
            "val_post_mses": "val_losses",
            "train_identity_baseline_post_mses": "train_baseline_mses",
            "val_identity_baseline_post_mses": "val_baseline_mses",
            "train_total_objectives": "train_objective_losses",
            "val_total_objectives": "val_objective_losses",
            "train_lpips_distances": "train_lpips_losses",
            "val_lpips_distances": "val_lpips_losses",
            "train_lesion_roi_ssim_losses": "train_ssim_losses",
            "val_lesion_roi_ssim_losses": "val_ssim_losses",
            "train_frozen_segmenter_soft_dice_losses": "train_dice_losses",
            "val_frozen_segmenter_soft_dice_losses": "val_dice_losses",
            "train_curia2_token_cosine_distances": "train_curia_losses",
            "val_curia2_token_cosine_distances": "val_curia_losses",
            "train_curia2_token_mses": "train_curia_losses",
            "val_curia2_token_mses": "val_curia_losses",
            "train_generator_adversarial_bces": "train_gan_losses",
            "val_generator_adversarial_bces": "val_gan_losses",
            "train_discriminator_bces": "train_discriminator_losses",
            "val_discriminator_bces": "val_discriminator_losses",
            "train_discriminator_balanced_patch_accuracies": "train_discriminator_accuracies",
            "val_discriminator_balanced_patch_accuracies": "val_discriminator_accuracies",
            "train_discriminator_real_patch_accuracies": "train_discriminator_real_accuracies",
            "val_discriminator_real_patch_accuracies": "val_discriminator_real_accuracies",
            "train_discriminator_fake_patch_accuracies": "train_discriminator_fake_accuracies",
            "val_discriminator_fake_patch_accuracies": "val_discriminator_fake_accuracies",
        }
        for canonical_key, legacy_key in aliases.items():
            if canonical_key not in logging_dict:
                logging_dict[canonical_key] = list(logging_dict.get(legacy_key, []))
        # Older runs have no stored average, so replay it from the logged raw values.
        if "val_total_objective_emas" not in logging_dict:
            ema_values = []
            ema = None
            for value in logging_dict.get("val_total_objectives", []):
                ema = (
                    float(value)
                    if ema is None
                    else self.validation_objective_ema_decay * ema
                    + (1.0 - self.validation_objective_ema_decay) * float(value)
                )
                ema_values.append(ema)
            logging_dict["val_total_objective_emas"] = ema_values
        logging_dict.setdefault("val_post_mse_emas", [])
        logging_dict.setdefault("val_frozen_segmenter_soft_dice_loss_emas", [])

    def _log_translation_metric(
        self, legacy_key: str, canonical_key: str, value: float
    ):
        """Log one value under both its short and its descriptive series name."""
        self.logger.log(legacy_key, value, self.current_epoch)
        self.logger.log(canonical_key, value, self.current_epoch)

    def _restore_validation_objective_selection_state(self):
        """Rebuild the checkpoint-selection state from the logged history.

        Called after loading a checkpoint so that a resumed run continues from the same
        running averages and best-so-far values instead of starting them over.
        """
        raw_values = self.logger.get_value("val_total_objectives", step=None) or []
        ema_values = self.logger.get_value("val_total_objective_emas", step=None) or []
        self.best_raw_validation_objective = min(raw_values) if raw_values else None
        self.validation_objective_ema = (
            float(ema_values[-1])
            if ema_values
            else (float(raw_values[-1]) if raw_values else None)
        )
        self._best_ema = (
            min(ema_values) if ema_values else (min(raw_values) if raw_values else None)
        )
        mse_ema_values = self.logger.get_value("val_post_mse_emas", step=None) or []
        self.validation_mse_ema = float(mse_ema_values[-1]) if mse_ema_values else None
        self.best_validation_mse_ema = min(mse_ema_values) if mse_ema_values else None
        dice_ema_values = (
            self.logger.get_value("val_frozen_segmenter_soft_dice_loss_emas", step=None)
            or []
        )
        self.validation_dice_ema = (
            float(dice_ema_values[-1]) if dice_ema_values else None
        )
        self.best_validation_dice_ema = (
            min(dice_ema_values) if dice_ema_values else None
        )

    def _update_best_ema_metric_checkpoint(
        self,
        value: float,
        metric_name: str,
        ema_log_key: str,
        checkpoint_name: str,
        ema_attr: str,
        best_ema_attr: str,
    ):
        """Update one metric's running average and save a checkpoint when it improves.

        Selection runs on an exponential moving average rather than the raw value,
        because single-epoch validation figures are noisy enough to pick a checkpoint
        that is merely lucky. Non-finite values are ignored.
        """
        from batchgenerators.utilities.file_and_folder_operations import join

        if not np.isfinite(value):
            return
        ema = getattr(self, ema_attr, None)
        decay = self.validation_objective_ema_decay
        ema = value if ema is None else decay * ema + (1.0 - decay) * value
        setattr(self, ema_attr, ema)
        self.logger.log(ema_log_key, ema, self.current_epoch)
        self.print_to_log_file(f"val_{metric_name}_ema", np.round(ema, decimals=6))
        best_ema = getattr(self, best_ema_attr, None)
        if best_ema is None or ema < best_ema:
            setattr(self, best_ema_attr, ema)
            self.print_to_log_file(
                f"New best EMA validation {metric_name}: {ema:.6f} "
                f"(decay={self.validation_objective_ema_decay:.2f})"
            )
            self.save_checkpoint(join(self.output_folder, checkpoint_name))

    def _update_best_translation_checkpoints(
        self,
        validation_objective: float,
        validation_mse: float = None,
        validation_dice: float = None,
    ):
        """Update the checkpoint selectors for the total objective and any sub-metrics.

        Writes the best-raw and best-averaged checkpoints for the total objective, and
        delegates the optional per-metric ones. Subclasses narrow which of these are kept.
        """
        from batchgenerators.utilities.file_and_folder_operations import join

        if np.isfinite(validation_objective):
            decay = self.validation_objective_ema_decay
            if self.validation_objective_ema is None:
                self.validation_objective_ema = validation_objective
            else:
                self.validation_objective_ema = (
                    decay * self.validation_objective_ema
                    + (1.0 - decay) * validation_objective
                )
            self.logger.log(
                "val_total_objective_emas",
                self.validation_objective_ema,
                self.current_epoch,
            )
            self.print_to_log_file(
                "val_total_objective_raw",
                np.round(validation_objective, decimals=6),
                "val_total_objective_ema",
                np.round(self.validation_objective_ema, decimals=6),
            )
            if (
                self.best_raw_validation_objective is None
                or validation_objective < self.best_raw_validation_objective
            ):
                self.best_raw_validation_objective = validation_objective
                self.print_to_log_file(
                    f"New best raw validation total objective: {validation_objective:.6f}"
                )
                self.save_checkpoint(
                    join(self.output_folder, "checkpoint_best_raw.pth")
                )
            if self._best_ema is None or self.validation_objective_ema < self._best_ema:
                self._best_ema = self.validation_objective_ema
                self.print_to_log_file(
                    f"New best EMA validation total objective: {self._best_ema:.6f} "
                    f"(decay={self.validation_objective_ema_decay:.2f})"
                )
                self.save_checkpoint(join(self.output_folder, "checkpoint_best.pth"))
        if validation_mse is not None:
            self._update_best_ema_metric_checkpoint(
                validation_mse,
                "post_mse",
                "val_post_mse_emas",
                "checkpoint_best_mse.pth",
                "validation_mse_ema",
                "best_validation_mse_ema",
            )
        if validation_dice is not None:
            self._update_best_ema_metric_checkpoint(
                validation_dice,
                "soft_dice_loss",
                "val_frozen_segmenter_soft_dice_loss_emas",
                "checkpoint_best_dice.pth",
                "validation_dice_ema",
                "best_validation_dice_ema",
            )

    @staticmethod
    def _masked_mse(
        prediction: torch.Tensor,
        target: torch.Tensor,
        acquired_image_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean squared error over acquired pixels; canvas padding carries zero weight.

        Dividing by the mask sum rather than the pixel count makes this a true mean over
        the masked region instead of one diluted by the excluded area.
        """
        acquired_image_mask = acquired_image_mask.to(dtype=prediction.dtype)
        squared_error = (prediction - target).square() * acquired_image_mask
        return squared_error.sum() / acquired_image_mask.sum().clamp_min(1.0)

    @staticmethod
    def get_translation_training_transforms(patch_size, rotation_for_DA, mirror_axes):
        """Spatial augmentation for translation: rotation and mirroring only.

        Scaling and elastic deformation are disabled, since the input and target have to
        stay in exact intensity correspondence.
        """
        from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
        from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
        from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms

        transforms = [
            SpatialTransform(
                patch_size,
                patch_center_dist_from_border=0,
                random_crop=False,
                p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA,
                p_scaling=0.0,
                scaling=(1.0, 1.0),
                p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False,
                mode_seg="nearest",
                border_mode_seg="zeros",
            )
        ]
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(MirrorTransform(allowed_axes=mirror_axes))
        return ComposeTransforms(transforms)

    def get_dataloaders(self):
        """Build the random training loader and the fixed validation loader.

        Validation uses a fixed slice window per case in a deterministic order, so the
        validation figures are comparable across epochs; its length sets the number of
        validation iterations per epoch.
        """
        patch_size = self.configuration_manager.patch_size
        rotation_for_DA, _, _, mirror_axes = (
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )
        tr_transforms = self.get_translation_training_transforms(
            patch_size, rotation_for_DA, mirror_axes
        )

        tr_keys, val_keys = self.do_split()
        tr_cases = [c for c in self.cases if c.case_id in set(tr_keys)]
        val_cases = [c for c in self.cases if c.case_id in set(val_keys)]
        dl_tr = MamaSynthTranslationSliceDataLoader(
            tr_cases,
            self.batch_size,
            self.pre_stats["mean"],
            self.pre_stats["std"],
            transforms=tr_transforms,
        )
        dl_val = MamaSynthTranslationFixedValDataLoader(
            val_cases, self.batch_size, self.pre_stats["mean"], self.pre_stats["std"]
        )
        self.fixed_validation_loader = dl_val
        self.num_val_iterations_per_epoch = dl_val.num_batches
        self.print_to_log_file(
            f"Fixed validation: {len(dl_val.samples)} slices "
            f"(maximum-lesion slice +/- 2 per case) in {dl_val.num_batches} batches"
        )

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr,
                transform=None,
                num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
        mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        _ = next(mt_gen_train)
        return mt_gen_train, mt_gen_val

    def on_validation_epoch_start(self):
        """Rewind the fixed validation loader so every epoch sees the same slices in order."""
        super().on_validation_epoch_start()
        self.fixed_validation_loader.reset()

    def train_step(self, batch: dict) -> dict:
        """Predict the delta, score it against the target and take one optimizer step.

        Also reports the error of the reconstructed post image and the error of leaving
        the input unchanged, which is the reference point the training curves are read
        against.
        """
        from nnunetv2.utilities.helpers import dummy_context

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        acquired_image_mask = batch["acquired_image_mask"].to(
            self.device, non_blocking=True
        )
        self.optimizer.zero_grad(set_to_none=True)
        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(data)
            loss = self._masked_mse(output, target, acquired_image_mask)
            recon_mse = self._masked_mse(
                data + output, data + target, acquired_image_mask
            )
            baseline_mse = self._masked_mse(data, data + target, acquired_image_mask)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {
            "loss": loss.detach().cpu().numpy(),
            "mse": recon_mse.detach().cpu().numpy(),
            "baseline_mse": baseline_mse.detach().cpu().numpy(),
        }

    def validation_step(self, batch: dict) -> dict:
        """Score one validation batch without updating the weights."""
        from nnunetv2.utilities.helpers import dummy_context

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        acquired_image_mask = batch["acquired_image_mask"].to(
            self.device, non_blocking=True
        )
        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(data)
            loss = self._masked_mse(output, target, acquired_image_mask)
            recon_mse = self._masked_mse(
                data + output, data + target, acquired_image_mask
            )
            baseline_mse = self._masked_mse(data, data + target, acquired_image_mask)
        return {
            "loss": loss.detach().cpu().numpy(),
            "mse": recon_mse.detach().cpu().numpy(),
            "baseline_mse": baseline_mse.detach().cpu().numpy(),
        }

    def on_train_epoch_end(self, train_outputs):
        """Average the training metrics across ranks, log them and refresh the preview."""
        self._ensure_translation_logger_keys()
        outputs = collate_outputs(train_outputs)
        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            baseline_losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs["mse"])
            dist.all_gather_object(baseline_losses_tr, outputs["baseline_mse"])
            train_mse = np.vstack(losses_tr).mean()
            train_baseline_mse = np.vstack(baseline_losses_tr).mean()
        else:
            train_mse = np.mean(outputs["mse"])
            train_baseline_mse = np.mean(outputs["baseline_mse"])
        self._log_translation_metric("train_losses", "train_post_mses", train_mse)
        self._log_translation_metric(
            "train_baseline_mses",
            "train_identity_baseline_post_mses",
            train_baseline_mse,
        )
        self._log_translation_metric(
            "train_objective_losses", "train_total_objectives", train_mse
        )
        if self.local_rank == 0:
            try:
                self._save_example_batch_png()
            except Exception as exc:
                self.print_to_log_file(f"Could not save translation example png: {exc}")

    def on_validation_epoch_end(self, val_outputs):
        """Average the validation metrics across ranks and log them."""
        self._ensure_translation_logger_keys()
        outputs = collate_outputs(val_outputs)
        if self.is_ddp:
            losses_val = [None for _ in range(dist.get_world_size())]
            baseline_losses_val = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_val, outputs["mse"])
            dist.all_gather_object(baseline_losses_val, outputs["baseline_mse"])
            val_mse = np.vstack(losses_val).mean()
            val_baseline_mse = np.vstack(baseline_losses_val).mean()
        else:
            val_mse = np.mean(outputs["mse"])
            val_baseline_mse = np.mean(outputs["baseline_mse"])
        self._log_translation_metric("val_losses", "val_post_mses", val_mse)
        self._log_translation_metric(
            "val_baseline_mses", "val_identity_baseline_post_mses", val_baseline_mse
        )
        self._log_translation_metric(
            "val_objective_losses", "val_total_objectives", val_mse
        )

    def on_epoch_end(self):
        """Print the epoch summary, roll the checkpoints and redraw the progress plot.

        The ratio of the reconstruction error to the identity baseline is logged as well,
        since it says whether the network is improving on simply copying its input.
        """
        from time import time

        from batchgenerators.utilities.file_and_folder_operations import join

        self.logger.log("epoch_end_timestamps", time(), self.current_epoch)
        self._ensure_translation_logger_keys()
        train_mse = float(self.logger.get_value("train_losses", step=-1))
        val_mse = float(self.logger.get_value("val_losses", step=-1))
        train_baseline_mse = float(
            self.logger.get_value("train_baseline_mses", step=-1)
        )
        val_baseline_mse = float(self.logger.get_value("val_baseline_mses", step=-1))
        train_ratio = train_mse / max(train_baseline_mse, 1e-8)
        val_ratio = val_mse / max(val_baseline_mse, 1e-8)
        self.print_to_log_file(
            "train_post_mse_valid_pixels",
            np.round(train_mse, decimals=6),
            "train_identity_baseline_post_mse_valid_pixels",
            np.round(train_baseline_mse, decimals=6),
            "train_post_mse_over_identity_baseline",
            np.round(train_ratio, decimals=4),
        )
        self.print_to_log_file(
            "val_post_mse_valid_pixels",
            np.round(val_mse, decimals=6),
            "val_identity_baseline_post_mse_valid_pixels",
            np.round(val_baseline_mse, decimals=6),
            "val_post_mse_over_identity_baseline",
            np.round(val_ratio, decimals=4),
        )
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.get_value('epoch_end_timestamps', step=-1) - self.logger.get_value('epoch_start_timestamps', step=-1), decimals=2)} s"
        )

        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (
            self.num_epochs - 1
        ):
            self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))

        self._update_best_translation_checkpoints(val_mse, validation_mse=val_mse)

        if self.local_rank == 0:
            self._plot_mse_progress_png()
        self.current_epoch += 1

    def _plot_mse_progress_png(self):
        """Draw reconstruction error against the identity baseline, epoch time and LR."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from batchgenerators.utilities.file_and_folder_operations import join

        self._ensure_translation_logger_keys()
        epochs = list(range(len(self.logger.get_value("train_losses", step=None))))
        fig, axes = plt.subplots(3, 1, figsize=(14, 16))
        axes[0].plot(
            epochs,
            self.logger.get_value("train_post_mses", step=None),
            label="train generated-post MSE (acquired pixels; padding excluded)",
            linewidth=2,
        )
        axes[0].plot(
            epochs,
            self.logger.get_value("train_identity_baseline_post_mses", step=None),
            label="train identity-baseline post MSE",
            linewidth=2,
            linestyle="--",
        )
        axes[0].plot(
            epochs,
            self.logger.get_value("val_post_mses", step=None),
            label="validation generated-post MSE (acquired pixels; padding excluded)",
            linewidth=2,
        )
        axes[0].plot(
            epochs,
            self.logger.get_value("val_identity_baseline_post_mses", step=None),
            label="validation identity-baseline post MSE",
            linewidth=2,
            linestyle="--",
        )
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("MSE")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        epoch_ends = self.logger.get_value("epoch_end_timestamps", step=None)
        epoch_starts = self.logger.get_value("epoch_start_timestamps", step=None)
        axes[1].plot(
            epochs,
            [e - s for e, s in zip(epoch_ends, epoch_starts)],
            label="epoch duration",
            linewidth=2,
        )
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("time [s]")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(
            epochs,
            self.logger.get_value("lrs", step=None),
            label="learning rate",
            linewidth=2,
        )
        axes[2].set_xlabel("epoch")
        axes[2].set_ylabel("learning rate")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            join(self.output_folder, "progress_translation_overview.png"), dpi=120
        )
        plt.close(fig)

    def _example_network_forward(self, data: torch.Tensor, batch: dict) -> torch.Tensor:
        """Network forward used by the example-image saver.

        Overridden by mixins that add input channels, so the preview is produced exactly
        the way the training step produces its output rather than from a bare image.
        """
        return self.network(data)

    def _save_example_batch_png(self, num_examples: int = 2):
        """Write a panel of input, reference, prediction, both deltas and the error.

        Padding is blanked out so the panels show only acquired anatomy, and the current
        error figures are put in the title.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from batchgenerators.utilities.file_and_folder_operations import join

        batch = next(self.dataloader_train)
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        acquired_image_mask = batch["acquired_image_mask"].to(
            self.device, non_blocking=True
        )
        keys = batch.get("keys", [""] * data.shape[0])
        was_training = self.network.training
        self.network.eval()
        with torch.no_grad():
            pred_delta = self._example_network_forward(data, batch)
            if isinstance(pred_delta, (list, tuple)):
                pred_delta = pred_delta[0]
        if was_training:
            self.network.train()

        pre = data.detach().cpu().float().numpy()
        target_delta = target.detach().cpu().float().numpy()
        pred_delta_np = pred_delta.detach().cpu().float().numpy()
        target_post = pre + target_delta
        pred_post = pre + pred_delta_np
        acquired_support = acquired_image_mask.detach().cpu().float().numpy() > 0.5
        err = np.abs(pred_post - target_post)
        n = min(num_examples, data.shape[0])

        fig, axes = plt.subplots(n, 6, figsize=(18, 3 * n), squeeze=False)
        titles = [
            "input pre-contrast (normalized)",
            "reference post-contrast (normalized)",
            "generated post-contrast (normalized)",
            "reference post-pre delta",
            "predicted post-pre delta",
            "absolute post-image error",
        ]
        for i in range(n):
            images = [
                pre[i, 0],
                target_post[i, 0],
                pred_post[i, 0],
                target_delta[i, 0],
                pred_delta_np[i, 0],
                err[i, 0],
            ]
            masks = [acquired_support[i, 0]] * len(images)
            for ax, image, mask, title in zip(axes[i], images, masks, titles):
                shown = np.where(mask, image, np.nan)
                ax.imshow(shown, cmap="gray")
                ax.set_title(title, fontsize=8)
                ax.axis("off")
            axes[i, 0].set_ylabel(str(keys[i]), fontsize=7)
        self._ensure_translation_logger_keys()
        train_mse = (
            self.logger.get_value("train_losses", step=-1)
            if self.logger.get_value("train_losses", step=None)
            else np.nan
        )
        val_mse = (
            self.logger.get_value("val_losses", step=-1)
            if self.logger.get_value("val_losses", step=None)
            else np.nan
        )
        train_baseline_mse = (
            self.logger.get_value("train_baseline_mses", step=-1)
            if self.logger.get_value("train_baseline_mses", step=None)
            else np.nan
        )
        val_baseline_mse = (
            self.logger.get_value("val_baseline_mses", step=-1)
            if self.logger.get_value("val_baseline_mses", step=None)
            else np.nan
        )
        fig.suptitle(
            f"epoch {self.current_epoch} | generated-post MSE on acquired pixels (padding excluded) | "
            f"train {train_mse:.6f} vs identity {train_baseline_mse:.6f} | "
            f"validation {val_mse:.6f} vs identity {val_baseline_mse:.6f}",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(join(self.output_folder, "translation_examples.png"), dpi=100)
        plt.close(fig)

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        """Load a checkpoint and restore the logging keys and selection state with it."""
        super().load_checkpoint(filename_or_checkpoint)
        self._ensure_translation_logger_keys()
        self._restore_validation_objective_selection_state()


class nnUNetTrainerMamaSynthTranslationLPIPS(nnUNetTrainerMamaSynthTranslation):
    """Translation with a perceptual term added to the masked pixel MSE.

    Both terms are taken over acquired pixels only; canvas padding is excluded.
    """

    lpips_weight = 1.0
    lpips_clip_sigma = 5.0
    lpips_net = "alex"
    _lpips_model = None

    def _ensure_translation_logger_keys(self):
        """Add the perceptual-term series to the logger."""
        super()._ensure_translation_logger_keys()
        local_logger = getattr(self.logger, "local_logger", None)
        if local_logger is None:
            return
        logging_dict = getattr(local_logger, "my_fantastic_logging", None)
        if logging_dict is None:
            return
        logging_dict.setdefault("train_lpips_losses", [])
        logging_dict.setdefault("val_lpips_losses", [])
        logging_dict.setdefault("train_objective_losses", [])
        logging_dict.setdefault("val_objective_losses", [])

    def _get_lpips_model(self):
        """Load the perceptual network once and keep it frozen in eval mode."""
        if self._lpips_model is not None:
            return self._lpips_model
        try:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        except Exception as exc:
            raise RuntimeError(
                "nnUNetTrainerMamaSynthTranslationLPIPS requires torchmetrics. Install the project dependency "
                "or run `pip install torchmetrics>=1.0`."
            ) from exc
        model = LearnedPerceptualImagePatchSimilarity(net_type=self.lpips_net).to(
            self.device
        )
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self._lpips_model = model
        return model

    def _lpips_preprocess(
        self, image: torch.Tensor, acquired_image_mask: torch.Tensor
    ) -> torch.Tensor:
        """Zero the padding, scale into [-1, 1] and replicate to three channels.

        The perceptual network expects natural-image-like RGB input in a bounded range,
        so the z-scored slice is clipped at a fixed number of standard deviations and
        divided by it. Zeroing first keeps synthetic padding from registering as texture.
        """
        image = image * acquired_image_mask.to(dtype=image.dtype)
        image = (
            image.clamp(-self.lpips_clip_sigma, self.lpips_clip_sigma)
            / self.lpips_clip_sigma
        )
        return image.expand(-1, 3, -1, -1)

    def _lpips_loss(
        self,
        prediction_post: torch.Tensor,
        target_post: torch.Tensor,
        acquired_image_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Perceptual distance between the generated and the real post-contrast image."""
        model = self._get_lpips_model()
        pred = self._lpips_preprocess(prediction_post, acquired_image_mask)
        target = self._lpips_preprocess(target_post, acquired_image_mask)
        # The metric object accumulates every call internally; without resetting on both
        # sides, memory grows for the whole run.
        model.reset()
        loss = model(pred, target)
        model.reset()
        return loss

    def train_step(self, batch: dict) -> dict:
        """Score the prediction with pixel and perceptual terms and take one optimizer step."""
        from nnunetv2.utilities.helpers import dummy_context

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        acquired_image_mask = batch["acquired_image_mask"].to(
            self.device, non_blocking=True
        )
        self.optimizer.zero_grad(set_to_none=True)
        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(data)
            prediction_post = data + output
            target_post = data + target
            mse_loss = self._masked_mse(
                prediction_post, target_post, acquired_image_mask
            )
            lpips_loss = self._lpips_loss(
                prediction_post, target_post, acquired_image_mask
            )
            loss = mse_loss + self.lpips_weight * lpips_loss
            baseline_mse = self._masked_mse(data, target_post, acquired_image_mask)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {
            "loss": loss.detach().cpu().numpy(),
            "mse": mse_loss.detach().cpu().numpy(),
            "baseline_mse": baseline_mse.detach().cpu().numpy(),
            "lpips": lpips_loss.detach().cpu().numpy(),
        }

    def validation_step(self, batch: dict) -> dict:
        """Score one validation batch without updating the weights."""
        from nnunetv2.utilities.helpers import dummy_context

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        acquired_image_mask = batch["acquired_image_mask"].to(
            self.device, non_blocking=True
        )
        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(data)
            prediction_post = data + output
            target_post = data + target
            mse_loss = self._masked_mse(
                prediction_post, target_post, acquired_image_mask
            )
            lpips_loss = self._lpips_loss(
                prediction_post, target_post, acquired_image_mask
            )
            loss = mse_loss + self.lpips_weight * lpips_loss
            baseline_mse = self._masked_mse(data, target_post, acquired_image_mask)
        return {
            "loss": loss.detach().cpu().numpy(),
            "mse": mse_loss.detach().cpu().numpy(),
            "baseline_mse": baseline_mse.detach().cpu().numpy(),
            "lpips": lpips_loss.detach().cpu().numpy(),
        }

    def on_train_epoch_end(self, train_outputs):
        """Average the training metrics across ranks, log them and refresh the preview."""
        self._ensure_translation_logger_keys()
        outputs = collate_outputs(train_outputs)
        if self.is_ddp:
            mses = [None for _ in range(dist.get_world_size())]
            baselines = [None for _ in range(dist.get_world_size())]
            lpipses = [None for _ in range(dist.get_world_size())]
            objectives = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(mses, outputs["mse"])
            dist.all_gather_object(baselines, outputs["baseline_mse"])
            dist.all_gather_object(lpipses, outputs["lpips"])
            dist.all_gather_object(objectives, outputs["loss"])
            train_mse = np.vstack(mses).mean()
            train_baseline_mse = np.vstack(baselines).mean()
            train_lpips = np.vstack(lpipses).mean()
            train_objective = np.vstack(objectives).mean()
        else:
            train_mse = np.mean(outputs["mse"])
            train_baseline_mse = np.mean(outputs["baseline_mse"])
            train_lpips = np.mean(outputs["lpips"])
            train_objective = np.mean(outputs["loss"])
        self._log_translation_metric("train_losses", "train_post_mses", train_mse)
        self._log_translation_metric(
            "train_baseline_mses",
            "train_identity_baseline_post_mses",
            train_baseline_mse,
        )
        self._log_translation_metric(
            "train_lpips_losses", "train_lpips_distances", train_lpips
        )
        self._log_translation_metric(
            "train_objective_losses", "train_total_objectives", train_objective
        )
        if self.local_rank == 0:
            try:
                self._save_example_batch_png()
            except Exception as exc:
                self.print_to_log_file(
                    f"Could not save LPIPS translation example png: {exc}"
                )

    def on_validation_epoch_end(self, val_outputs):
        """Average the validation metrics across ranks and log them."""
        self._ensure_translation_logger_keys()
        outputs = collate_outputs(val_outputs)
        if self.is_ddp:
            mses = [None for _ in range(dist.get_world_size())]
            baselines = [None for _ in range(dist.get_world_size())]
            lpipses = [None for _ in range(dist.get_world_size())]
            objectives = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(mses, outputs["mse"])
            dist.all_gather_object(baselines, outputs["baseline_mse"])
            dist.all_gather_object(lpipses, outputs["lpips"])
            dist.all_gather_object(objectives, outputs["loss"])
            val_mse = np.vstack(mses).mean()
            val_baseline_mse = np.vstack(baselines).mean()
            val_lpips = np.vstack(lpipses).mean()
            val_objective = np.vstack(objectives).mean()
        else:
            val_mse = np.mean(outputs["mse"])
            val_baseline_mse = np.mean(outputs["baseline_mse"])
            val_lpips = np.mean(outputs["lpips"])
            val_objective = np.mean(outputs["loss"])
        self._log_translation_metric("val_losses", "val_post_mses", val_mse)
        self._log_translation_metric(
            "val_baseline_mses", "val_identity_baseline_post_mses", val_baseline_mse
        )
        self._log_translation_metric(
            "val_lpips_losses", "val_lpips_distances", val_lpips
        )
        self._log_translation_metric(
            "val_objective_losses", "val_total_objectives", val_objective
        )

    def on_epoch_end(self):
        """Print the epoch summary, roll the checkpoints and redraw the progress plots.

        Checkpoint selection now runs on the combined objective rather than the pixel
        error alone, since that is what is being optimized.
        """
        from time import time

        from batchgenerators.utilities.file_and_folder_operations import join

        self.logger.log("epoch_end_timestamps", time(), self.current_epoch)
        self._ensure_translation_logger_keys()
        train_mse = float(self.logger.get_value("train_losses", step=-1))
        val_mse = float(self.logger.get_value("val_losses", step=-1))
        train_baseline_mse = float(
            self.logger.get_value("train_baseline_mses", step=-1)
        )
        val_baseline_mse = float(self.logger.get_value("val_baseline_mses", step=-1))
        train_lpips = float(self.logger.get_value("train_lpips_losses", step=-1))
        val_lpips = float(self.logger.get_value("val_lpips_losses", step=-1))
        train_objective = float(
            self.logger.get_value("train_objective_losses", step=-1)
        )
        val_objective = float(self.logger.get_value("val_objective_losses", step=-1))
        train_ratio = train_mse / max(train_baseline_mse, 1e-8)
        val_ratio = val_mse / max(val_baseline_mse, 1e-8)
        self.print_to_log_file(
            "train_total_objective",
            np.round(train_objective, decimals=6),
            "train_post_mse_valid_pixels",
            np.round(train_mse, decimals=6),
            "train_lpips_distance",
            np.round(train_lpips, decimals=6),
        )
        self.print_to_log_file(
            "train_identity_baseline_post_mse_valid_pixels",
            np.round(train_baseline_mse, decimals=6),
            "train_post_mse_over_identity_baseline",
            np.round(train_ratio, decimals=4),
        )
        self.print_to_log_file(
            "val_total_objective",
            np.round(val_objective, decimals=6),
            "val_post_mse_valid_pixels",
            np.round(val_mse, decimals=6),
            "val_lpips_distance",
            np.round(val_lpips, decimals=6),
        )
        self.print_to_log_file(
            "val_identity_baseline_post_mse_valid_pixels",
            np.round(val_baseline_mse, decimals=6),
            "val_post_mse_over_identity_baseline",
            np.round(val_ratio, decimals=4),
        )
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.get_value('epoch_end_timestamps', step=-1) - self.logger.get_value('epoch_start_timestamps', step=-1), decimals=2)} s"
        )

        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (
            self.num_epochs - 1
        ):
            self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))

        self._update_best_translation_checkpoints(val_objective, validation_mse=val_mse)

        if self.local_rank == 0:
            self._plot_mse_progress_png()
        self.current_epoch += 1

    def _plot_mse_progress_png(self):
        """Add a plot of the total objective and the perceptual distance."""
        super()._plot_mse_progress_png()
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from batchgenerators.utilities.file_and_folder_operations import join

        epochs = list(
            range(len(self.logger.get_value("train_lpips_losses", step=None)))
        )
        if not epochs:
            return
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        axes[0].plot(
            epochs,
            self.logger.get_value("train_total_objectives", step=None),
            label="train total weighted objective",
            linewidth=2,
        )
        axes[0].plot(
            epochs,
            self.logger.get_value("val_total_objectives", step=None),
            label="validation total weighted objective (raw)",
            linewidth=2,
        )
        axes[0].plot(
            epochs,
            self.logger.get_value("val_total_objective_emas", step=None),
            label=f"validation total objective EMA (decay={self.validation_objective_ema_decay:.2f})",
            linewidth=2,
            linestyle="--",
        )
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("weighted training objective")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(
            epochs,
            self.logger.get_value("train_lpips_losses", step=None),
            label="train LPIPS distance",
            linewidth=2,
        )
        axes[1].plot(
            epochs,
            self.logger.get_value("val_lpips_losses", step=None),
            label="validation LPIPS distance",
            linewidth=2,
        )
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("LPIPS distance")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            join(self.output_folder, "progress_total_objective_and_lpips.png"), dpi=120
        )
        plt.close(fig)

    def _save_example_batch_png(self, num_examples: int = 2):
        """Write the inherited example panel; the perceptual term appears in the plots."""
        super()._save_example_batch_png(num_examples=num_examples)


class nnUNetTrainerMamaSynthTranslationLPIPSSSIM(
    nnUNetTrainerMamaSynthTranslationLPIPS
):
    """Adds a structural-similarity term over the lesion region.

    The pixel and perceptual terms still cover the whole acquired image; the
    structural term looks only inside the lesion.
    """

    ssim_weight = 1.0
    ssim_data_range = 10.0
    ssim_window_size = 7

    def _ensure_translation_logger_keys(self):
        """Add the structural-similarity series to the logger."""
        super()._ensure_translation_logger_keys()
        local_logger = getattr(self.logger, "local_logger", None)
        if local_logger is None:
            return
        logging_dict = getattr(local_logger, "my_fantastic_logging", None)
        if logging_dict is None:
            return
        logging_dict.setdefault("train_ssim_losses", [])
        logging_dict.setdefault("val_ssim_losses", [])

    def _ssim_loss(
        self,
        prediction_post: torch.Tensor,
        target_post: torch.Tensor,
        lesion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """One minus the structural similarity, averaged over the lesion region.

        A full-image similarity map is computed with a uniform local window and then
        averaged inside the lesion only, which keeps the statistics local rather than
        restricting the window itself to the region. Returns zero for lesion-free slices.
        """
        from contextlib import nullcontext

        import torch.nn.functional as F

        # Do the similarity arithmetic in float32 outside mixed precision: the local
        # variance terms can underflow to NaN in half precision even when every other loss
        # is finite.
        autocast_context = (
            torch.autocast(prediction_post.device.type, enabled=False)
            if prediction_post.device.type == "cuda"
            else nullcontext()
        )
        with autocast_context:
            prediction = prediction_post.float()
            target = target_post.float()
            lesion_roi = lesion_mask.float()
            lesion_sum = lesion_roi.sum()
            if lesion_sum < 1.0:
                return prediction.sum() * 0.0

            channels = prediction.shape[1]
            window_size = int(self.ssim_window_size)
            padding = window_size // 2
            window_area = float(window_size * window_size)
            # Bessel correction, so the windowed variance is unbiased.
            covariance_norm = window_area / max(window_area - 1.0, 1.0)
            kernel = (
                prediction.new_ones((channels, 1, window_size, window_size))
                / window_area
            )

            def local_mean(image: torch.Tensor) -> torch.Tensor:
                """Windowed mean via a uniform convolution, with reflected borders."""
                padded = F.pad(
                    image, (padding, padding, padding, padding), mode="reflect"
                )
                return F.conv2d(padded, kernel, groups=channels)

            mu_x = local_mean(prediction)
            mu_y = local_mean(target)
            mu_x_sq = mu_x.square()
            mu_y_sq = mu_y.square()
            mu_xy = mu_x * mu_y

            sigma_x_sq = covariance_norm * (local_mean(prediction.square()) - mu_x_sq)
            sigma_y_sq = covariance_norm * (local_mean(target.square()) - mu_y_sq)
            sigma_xy = covariance_norm * (local_mean(prediction * target) - mu_xy)
            sigma_x_sq = sigma_x_sq.clamp_min(0.0)
            sigma_y_sq = sigma_y_sq.clamp_min(0.0)

            c1 = (0.01 * self.ssim_data_range) ** 2
            c2 = (0.03 * self.ssim_data_range) ** 2
            numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
            denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
            ssim_map = numerator / denominator.clamp_min(
                torch.finfo(prediction.dtype).eps
            )
            # Uniform regions can still divide near-zero by near-zero; map those to the worst
            # similarity rather than letting a NaN reach the optimizer.
            ssim_map = torch.nan_to_num(ssim_map, nan=-1.0, posinf=1.0, neginf=-1.0)
            ssim_value = (ssim_map * lesion_roi).sum() / lesion_sum.clamp_min(1.0)
            return 1.0 - ssim_value.clamp(-1.0, 1.0)

    def train_step(self, batch: dict) -> dict:
        """Score the prediction with all three terms and take one optimizer step."""
        from nnunetv2.utilities.helpers import dummy_context

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        acquired_image_mask = batch["acquired_image_mask"].to(
            self.device, non_blocking=True
        )
        lesion_mask = batch["lesion_mask"].to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(data)
            prediction_post = data + output
            target_post = data + target
            mse_loss = self._masked_mse(
                prediction_post, target_post, acquired_image_mask
            )
            lpips_loss = self._lpips_loss(
                prediction_post, target_post, acquired_image_mask
            )
            ssim_loss = self._ssim_loss(prediction_post, target_post, lesion_mask)
            loss = (
                mse_loss + self.lpips_weight * lpips_loss + self.ssim_weight * ssim_loss
            )
            baseline_mse = self._masked_mse(data, target_post, acquired_image_mask)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {
            "loss": loss.detach().cpu().numpy(),
            "mse": mse_loss.detach().cpu().numpy(),
            "baseline_mse": baseline_mse.detach().cpu().numpy(),
            "lpips": lpips_loss.detach().cpu().numpy(),
            "ssim": ssim_loss.detach().cpu().numpy(),
        }

    def validation_step(self, batch: dict) -> dict:
        """Score one validation batch without updating the weights."""
        from nnunetv2.utilities.helpers import dummy_context

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        acquired_image_mask = batch["acquired_image_mask"].to(
            self.device, non_blocking=True
        )
        lesion_mask = batch["lesion_mask"].to(self.device, non_blocking=True)
        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(data)
            prediction_post = data + output
            target_post = data + target
            mse_loss = self._masked_mse(
                prediction_post, target_post, acquired_image_mask
            )
            lpips_loss = self._lpips_loss(
                prediction_post, target_post, acquired_image_mask
            )
            ssim_loss = self._ssim_loss(prediction_post, target_post, lesion_mask)
            loss = (
                mse_loss + self.lpips_weight * lpips_loss + self.ssim_weight * ssim_loss
            )
            baseline_mse = self._masked_mse(data, target_post, acquired_image_mask)
        return {
            "loss": loss.detach().cpu().numpy(),
            "mse": mse_loss.detach().cpu().numpy(),
            "baseline_mse": baseline_mse.detach().cpu().numpy(),
            "lpips": lpips_loss.detach().cpu().numpy(),
            "ssim": ssim_loss.detach().cpu().numpy(),
        }

    def on_train_epoch_end(self, train_outputs):
        """Average the training metrics across ranks, log them and refresh the preview."""
        self._ensure_translation_logger_keys()
        outputs = collate_outputs(train_outputs)
        if self.is_ddp:
            mses = [None for _ in range(dist.get_world_size())]
            baselines = [None for _ in range(dist.get_world_size())]
            lpipses = [None for _ in range(dist.get_world_size())]
            ssims = [None for _ in range(dist.get_world_size())]
            objectives = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(mses, outputs["mse"])
            dist.all_gather_object(baselines, outputs["baseline_mse"])
            dist.all_gather_object(lpipses, outputs["lpips"])
            dist.all_gather_object(ssims, outputs["ssim"])
            dist.all_gather_object(objectives, outputs["loss"])
            train_mse = np.vstack(mses).mean()
            train_baseline_mse = np.vstack(baselines).mean()
            train_lpips = np.vstack(lpipses).mean()
            train_ssim = np.vstack(ssims).mean()
            train_objective = np.vstack(objectives).mean()
        else:
            train_mse = np.mean(outputs["mse"])
            train_baseline_mse = np.mean(outputs["baseline_mse"])
            train_lpips = np.mean(outputs["lpips"])
            train_ssim = np.mean(outputs["ssim"])
            train_objective = np.mean(outputs["loss"])
        self._log_translation_metric("train_losses", "train_post_mses", train_mse)
        self._log_translation_metric(
            "train_baseline_mses",
            "train_identity_baseline_post_mses",
            train_baseline_mse,
        )
        self._log_translation_metric(
            "train_lpips_losses", "train_lpips_distances", train_lpips
        )
        self._log_translation_metric(
            "train_ssim_losses", "train_lesion_roi_ssim_losses", train_ssim
        )
        self._log_translation_metric(
            "train_objective_losses", "train_total_objectives", train_objective
        )
        if self.local_rank == 0:
            try:
                self._save_example_batch_png()
            except Exception as exc:
                self.print_to_log_file(
                    f"Could not save LPIPS+SSIM translation example png: {exc}"
                )

    def on_validation_epoch_end(self, val_outputs):
        """Average the validation metrics across ranks and log them."""
        self._ensure_translation_logger_keys()
        outputs = collate_outputs(val_outputs)
        if self.is_ddp:
            mses = [None for _ in range(dist.get_world_size())]
            baselines = [None for _ in range(dist.get_world_size())]
            lpipses = [None for _ in range(dist.get_world_size())]
            ssims = [None for _ in range(dist.get_world_size())]
            objectives = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(mses, outputs["mse"])
            dist.all_gather_object(baselines, outputs["baseline_mse"])
            dist.all_gather_object(lpipses, outputs["lpips"])
            dist.all_gather_object(ssims, outputs["ssim"])
            dist.all_gather_object(objectives, outputs["loss"])
            val_mse = np.vstack(mses).mean()
            val_baseline_mse = np.vstack(baselines).mean()
            val_lpips = np.vstack(lpipses).mean()
            val_ssim = np.vstack(ssims).mean()
            val_objective = np.vstack(objectives).mean()
        else:
            val_mse = np.mean(outputs["mse"])
            val_baseline_mse = np.mean(outputs["baseline_mse"])
            val_lpips = np.mean(outputs["lpips"])
            val_ssim = np.mean(outputs["ssim"])
            val_objective = np.mean(outputs["loss"])
        self._log_translation_metric("val_losses", "val_post_mses", val_mse)
        self._log_translation_metric(
            "val_baseline_mses", "val_identity_baseline_post_mses", val_baseline_mse
        )
        self._log_translation_metric(
            "val_lpips_losses", "val_lpips_distances", val_lpips
        )
        self._log_translation_metric(
            "val_ssim_losses", "val_lesion_roi_ssim_losses", val_ssim
        )
        self._log_translation_metric(
            "val_objective_losses", "val_total_objectives", val_objective
        )

    def on_epoch_end(self):
        """Print the epoch summary, roll the checkpoints and redraw the progress plots."""
        from time import time

        from batchgenerators.utilities.file_and_folder_operations import join

        self.logger.log("epoch_end_timestamps", time(), self.current_epoch)
        self._ensure_translation_logger_keys()
        train_mse = float(self.logger.get_value("train_losses", step=-1))
        val_mse = float(self.logger.get_value("val_losses", step=-1))
        train_baseline_mse = float(
            self.logger.get_value("train_baseline_mses", step=-1)
        )
        val_baseline_mse = float(self.logger.get_value("val_baseline_mses", step=-1))
        train_lpips = float(self.logger.get_value("train_lpips_losses", step=-1))
        val_lpips = float(self.logger.get_value("val_lpips_losses", step=-1))
        train_ssim = float(self.logger.get_value("train_ssim_losses", step=-1))
        val_ssim = float(self.logger.get_value("val_ssim_losses", step=-1))
        train_objective = float(
            self.logger.get_value("train_objective_losses", step=-1)
        )
        val_objective = float(self.logger.get_value("val_objective_losses", step=-1))
        train_ratio = train_mse / max(train_baseline_mse, 1e-8)
        val_ratio = val_mse / max(val_baseline_mse, 1e-8)
        self.print_to_log_file(
            "train_total_objective",
            np.round(train_objective, decimals=6),
            "train_post_mse_valid_pixels",
            np.round(train_mse, decimals=6),
            "train_lpips_distance",
            np.round(train_lpips, decimals=6),
            "train_lesion_roi_ssim_loss",
            np.round(train_ssim, decimals=6),
        )
        self.print_to_log_file(
            "train_identity_baseline_post_mse_valid_pixels",
            np.round(train_baseline_mse, decimals=6),
            "train_post_mse_over_identity_baseline",
            np.round(train_ratio, decimals=4),
        )
        self.print_to_log_file(
            "val_total_objective",
            np.round(val_objective, decimals=6),
            "val_post_mse_valid_pixels",
            np.round(val_mse, decimals=6),
            "val_lpips_distance",
            np.round(val_lpips, decimals=6),
            "val_lesion_roi_ssim_loss",
            np.round(val_ssim, decimals=6),
        )
        self.print_to_log_file(
            "val_identity_baseline_post_mse_valid_pixels",
            np.round(val_baseline_mse, decimals=6),
            "val_post_mse_over_identity_baseline",
            np.round(val_ratio, decimals=4),
        )
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.get_value('epoch_end_timestamps', step=-1) - self.logger.get_value('epoch_start_timestamps', step=-1), decimals=2)} s"
        )

        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (
            self.num_epochs - 1
        ):
            self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))

        self._update_best_translation_checkpoints(val_objective, validation_mse=val_mse)

        if self.local_rank == 0:
            self._plot_mse_progress_png()
        self.current_epoch += 1

    def _plot_mse_progress_png(self):
        """Add a plot of the lesion structural-similarity term."""
        super()._plot_mse_progress_png()
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from batchgenerators.utilities.file_and_folder_operations import join

        epochs = list(range(len(self.logger.get_value("train_ssim_losses", step=None))))
        if not epochs:
            return
        fig, ax = plt.subplots(1, 1, figsize=(14, 5))
        ax.plot(
            epochs,
            self.logger.get_value("train_ssim_losses", step=None),
            label="train lesion-ROI SSIM loss",
            linewidth=2,
        )
        ax.plot(
            epochs,
            self.logger.get_value("val_ssim_losses", step=None),
            label="validation lesion-ROI SSIM loss",
            linewidth=2,
        )
        ax.set_xlabel("epoch")
        ax.set_ylabel("1 - SSIM")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            join(self.output_folder, "progress_lesion_roi_ssim_loss.png"), dpi=120
        )
        plt.close(fig)


class nnUNetTrainerMamaSynthTranslationLPIPSSSIMDice(
    nnUNetTrainerMamaSynthTranslationLPIPSSSIM
):
    """Adds a frozen segmenter's soft Dice on the generated post-contrast image.

    A separately trained lesion segmenter is run on the synthetic image and its
    output compared against the real lesion mask, so the generator is pushed to make
    the lesion recognisable and not merely close in intensity. The segmenter's
    weights never change; gradients flow through it back into the generator.

    Several optional lesion-local terms live here too, all switched off by default
    and enabled by the subclasses named after them: an extra reconstruction weight on
    the lesion boundary band, a soft recall over that band, first-moment matching
    inside the lesion, and a one-sided penalty for lesion pixels darker than the real
    post image.
    """

    dice_weight = 1.0
    tversky_weight = 0.0
    tversky_alpha = 0.2
    tversky_beta = 0.8
    lesion_mask_dilation_radius = 0
    contour_weight = 0.0
    contour_dilation_radius = 3
    contour_erosion_radius = 3
    contour_recall_weight = 0.0
    lesion_moment_weight = 0.0
    moment_std_weight = 0.5
    moment_eps = 1e-8
    lesion_min_pixels = 1
    under_enhancement_weight = 0.0
    segmentation_use_mirroring = True
    # Set by run_training_mamasynth.py from --seg-model-dir; None here means the
    # trainer was started without it (see _load_segmentation_model).
    segmentation_model_dir = os.environ.get("MAMASYNTH_SEGMENTATION_MODEL_DIR")
    segmentation_checkpoint_name = os.environ.get(
        "MAMASYNTH_SEGMENTATION_CHECKPOINT", "checkpoint_final.pth"
    )
    _segmentation_model = None

    def _ensure_translation_logger_keys(self):
        """Add the segmentation series, and the optional contour series when enabled."""
        super()._ensure_translation_logger_keys()
        local_logger = getattr(self.logger, "local_logger", None)
        if local_logger is None:
            return
        logging_dict = getattr(local_logger, "my_fantastic_logging", None)
        if logging_dict is None:
            return
        logging_dict.setdefault("train_dice_losses", [])
        logging_dict.setdefault("val_dice_losses", [])
        logging_dict.setdefault("train_tversky_losses", [])
        logging_dict.setdefault("val_tversky_losses", [])
        if self.contour_weight != 0:
            logging_dict.setdefault("train_contour_losses", [])
            logging_dict.setdefault("val_contour_losses", [])
        if self.contour_recall_weight != 0:
            logging_dict.setdefault("train_contour_recall_losses", [])
            logging_dict.setdefault("val_contour_recall_losses", [])

    def _get_segmentation_model(self):
        """Load the frozen lesion segmenter once, from its own plans and checkpoint.

        The architecture is rebuilt from the segmenter's stored plans rather than the
        translation trainer's, the checkpoint's distributed-training key prefixes are
        stripped, and every parameter is frozen.
        """
        if self._segmentation_model is not None:
            return self._segmentation_model

        from nnunetv2.utilities.label_handling.label_handling import (
            determine_num_input_channels,
        )
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

        if self.segmentation_model_dir is None:
            raise RuntimeError(
                f"{type(self).__name__} needs a chained segmentation model. Pass "
                "--seg-model-dir to nnUNetv2_train_mamasynth_translate, or set "
                "MAMASYNTH_SEGMENTATION_MODEL_DIR."
            )

        model_dir = Path(self.segmentation_model_dir)
        plans_path = model_dir / "plans.json"
        dataset_path = model_dir / "dataset.json"
        checkpoint_path = model_dir / "fold_0" / self.segmentation_checkpoint_name
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Segmentation checkpoint not found: {checkpoint_path}"
            )

        with plans_path.open("r", encoding="utf-8") as handle:
            seg_plans = json.load(handle)
        with dataset_path.open("r", encoding="utf-8") as handle:
            seg_dataset_json = json.load(handle)

        plans_manager = PlansManager(seg_plans)
        configuration_manager = plans_manager.get_configuration("2d")
        label_manager = plans_manager.get_label_manager(seg_dataset_json)
        num_input_channels = determine_num_input_channels(
            plans_manager, configuration_manager, seg_dataset_json
        )
        model = nnUNetTrainer.build_network_architecture(
            plans_manager,
            configuration_manager,
            num_input_channels,
            label_manager.num_segmentation_heads,
            enable_deep_supervision=False,
        )

        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        # Checkpoints written under distributed training carry a module prefix on every
        # key; strip it so the weights load into a plain network.
        state_dict = checkpoint["network_weights"]
        state_dict = {
            key[7:] if key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        # The segmenter is only a fixed critic: it never trains, but gradients still pass
        # through it into the generator.
        for param in model.parameters():
            param.requires_grad_(False)
        self._segmentation_model = model
        return model

    def _segmentation_logits(self, image: torch.Tensor) -> torch.Tensor:
        """Run the frozen segmenter, optionally averaging over mirrored inputs."""
        segmentation_model = self._get_segmentation_model()

        def forward(candidate: torch.Tensor) -> torch.Tensor:
            """One segmenter pass, taking the full-resolution head if several are returned."""
            logits = segmentation_model(candidate.float())
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            return logits

        logits = forward(image)
        if not self.segmentation_use_mirroring:
            return logits

        mirrored_logits = [logits]
        for dims in ((2,), (3,), (2, 3)):
            flipped_image = torch.flip(image, dims=dims)
            flipped_logits = forward(flipped_image)
            mirrored_logits.append(torch.flip(flipped_logits, dims=dims))
        return torch.stack(mirrored_logits, dim=0).mean(dim=0)

    def _segmentation_target(
        self,
        lesion_mask: torch.Tensor,
        acquired_image_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """The lesion mask as a soft target, optionally dilated, clipped to acquired pixels."""
        import torch.nn.functional as F

        target = lesion_mask.to(dtype=dtype)
        radius = int(self.lesion_mask_dilation_radius)
        if radius > 0:
            kernel_size = 2 * radius + 1
            target = F.max_pool2d(
                target, kernel_size=kernel_size, stride=1, padding=radius
            )
        return target * acquired_image_mask.to(dtype=dtype)

    def _segmentation_probabilities_and_target(
        self,
        prediction_post: torch.Tensor,
        lesion_mask: torch.Tensor,
        acquired_image_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the segmenter's lesion probability and the matching target.

        Probabilities are kept rather than thresholded so the loss stays differentiable.
        """
        logits = self._segmentation_logits(prediction_post)
        lesion_probability = torch.softmax(logits, dim=1)[:, 1:2]
        acquired_support = acquired_image_mask.to(dtype=lesion_probability.dtype)
        target = self._segmentation_target(
            lesion_mask, acquired_image_mask, lesion_probability.dtype
        )
        return lesion_probability * acquired_support, target

    def _segmentation_dice_loss(
        self,
        prediction_post: torch.Tensor,
        lesion_mask: torch.Tensor,
        acquired_image_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Soft Dice between the segmenter's lesion probability and the real mask."""
        prediction, target = self._segmentation_probabilities_and_target(
            prediction_post, lesion_mask, acquired_image_mask
        )
        axes = tuple(range(1, prediction.ndim))
        intersection = (prediction * target).sum(dim=axes)
        denominator = prediction.sum(dim=axes) + target.sum(dim=axes)
        dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
        return 1.0 - dice.mean()

    def _segmentation_losses(
        self,
        prediction_post: torch.Tensor,
        lesion_mask: torch.Tensor,
        acquired_image_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the soft Dice, the Tversky variant and the optional contour recall.

        All three read the same segmenter output, so they cost one forward pass in total.
        Tversky reweights false negatives against false positives; contour recall asks
        the lesion probability to stay high all along the boundary band.
        """
        prediction, target = self._segmentation_probabilities_and_target(
            prediction_post, lesion_mask, acquired_image_mask
        )
        axes = tuple(range(1, prediction.ndim))
        true_positive = (prediction * target).sum(dim=axes)
        prediction_sum = prediction.sum(dim=axes)
        target_sum = target.sum(dim=axes)
        dice_loss = (
            1.0
            - ((2.0 * true_positive + 1.0) / (prediction_sum + target_sum + 1.0)).mean()
        )
        if self.tversky_weight == 0:
            tversky_loss = dice_loss.new_zeros(())
        else:
            false_positive = prediction_sum - true_positive
            false_negative = target_sum - true_positive
            tversky_loss = (
                1.0
                - (
                    (true_positive + 1.0)
                    / (
                        true_positive
                        + self.tversky_alpha * false_positive
                        + self.tversky_beta * false_negative
                        + 1.0
                    )
                ).mean()
            )
        if self.contour_recall_weight != 0:
            band = self._lesion_contour_band(lesion_mask, acquired_image_mask).to(
                dtype=prediction.dtype
            )
            band_sum = band.sum(dim=axes)
            recall = (prediction * band).sum(dim=axes) / band_sum.clamp_min(1.0)
            has_band = (band_sum > 0).to(dtype=recall.dtype)
            contour_recall_loss = (
                has_band * (1.0 - recall)
            ).sum() / has_band.sum().clamp_min(1.0)
        else:
            contour_recall_loss = dice_loss.new_zeros(())
        return dice_loss, tversky_loss, contour_recall_loss

    def _lesion_contour_band(
        self, lesion_mask: torch.Tensor, acquired_image_mask: torch.Tensor
    ) -> torch.Tensor:
        """A band hugging the lesion boundary: a dilation minus an erosion of the mask.

        One inside the band, zero deep inside the lesion and far outside it, and empty
        for slices with no lesion. Dilation and erosion are max-pools (the erosion on the
        negated mask), so the band straddles the contour by about the configured radii.
        Restricted to acquired pixels.
        """
        import torch.nn.functional as F

        mask = (lesion_mask > 0.5).to(dtype=torch.float32)
        dilation_radius = int(self.contour_dilation_radius)
        erosion_radius = int(self.contour_erosion_radius)
        dilated = (
            F.max_pool2d(
                mask,
                kernel_size=2 * dilation_radius + 1,
                stride=1,
                padding=dilation_radius,
            )
            if dilation_radius > 0
            else mask
        )
        eroded = (
            -F.max_pool2d(
                -mask,
                kernel_size=2 * erosion_radius + 1,
                stride=1,
                padding=erosion_radius,
            )
            if erosion_radius > 0
            else mask
        )
        band = (dilated - eroded).clamp_(0.0, 1.0)
        return band * acquired_image_mask.to(dtype=band.dtype)

    def _lesion_contour_loss(
        self,
        prediction_post: torch.Tensor,
        target_post: torch.Tensor,
        lesion_mask: torch.Tensor,
        acquired_image_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Masked MSE over the lesion boundary band only.

        Concentrates reconstruction weight on the lesion margin rather than its interior.
        Returns zero for lesion-free slices.
        """
        band = self._lesion_contour_band(lesion_mask, acquired_image_mask).to(
            dtype=prediction_post.dtype
        )
        squared_error = (prediction_post - target_post).square() * band
        return squared_error.sum() / band.sum().clamp_min(1.0)

    @staticmethod
    def _masked_mean_std(image: torch.Tensor, region_mask: torch.Tensor, eps: float):
        """Per-image mean and standard deviation of an image inside a region mask."""
        axes = (1, 2, 3)
        count = region_mask.sum(dim=axes).clamp_min(1.0)
        mean = (image * region_mask).sum(dim=axes) / count
        mean_broadcast = mean.view(-1, 1, 1, 1)
        variance = (((image - mean_broadcast).square()) * region_mask).sum(
            dim=axes
        ) / count
        std = variance.clamp_min(eps).sqrt()
        return mean, std

    def _lesion_moment_loss(
        self,
        prediction_post: torch.Tensor,
        target_post: torch.Tensor,
        lesion_mask: torch.Tensor,
        acquired_image_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Match the lesion region's mean and spread to those of the real post image.

        A per-pixel squared error pulls each pixel toward the conditional mean of its
        target, which leaves the lesion flatter and dimmer than the real one. Matching
        the first two moments restores its overall level and its internal variation
        without dictating individual pixel values. Both statistics are taken from the
        real target, so the term cannot push the lesion past it. Restricted to acquired
        pixels, and gated on lesion presence so empty slices contribute nothing.
        """
        prediction = prediction_post.float()
        target = target_post.float()
        lesion_region = (lesion_mask.float() > 0.5).to(
            prediction.dtype
        ) * acquired_image_mask.float()
        mean_prediction, std_prediction = self._masked_mean_std(
            prediction, lesion_region, self.moment_eps
        )
        mean_target, std_target = self._masked_mean_std(
            target, lesion_region, self.moment_eps
        )
        per_sample = (mean_prediction - mean_target).square() + (
            self.moment_std_weight * (std_prediction - std_target).square()
        )
        present = lesion_region.sum(dim=(1, 2, 3)) >= float(self.lesion_min_pixels)
        if bool(present.any()):
            return per_sample[present].mean()
        return prediction.sum() * 0.0

    @staticmethod
    def _under_enhancement_hinge(
        prediction_post: torch.Tensor,
        target_post: torch.Tensor,
        lesion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Penalise only lesion pixels that are darker than the real post image.

        The mean of the squared shortfall, so pixels at or above the target cost nothing.
        Being one-sided it lifts the dim end of the lesion without capping the bright
        end, though it can also compress the lower tail and so reduce variation inside
        the lesion. Returns zero for lesion-free slices.
        """
        pred = prediction_post.float()
        tgt = target_post.float()
        roi = (lesion_mask.float() > 0.5).to(pred.dtype)
        lesion_sum = roi.sum()
        if lesion_sum < 1.0:
            return pred.sum() * 0.0
        deficit = torch.relu(tgt - pred)
        return (deficit.square() * roi).sum() / lesion_sum.clamp_min(1.0)

    def train_step(self, batch: dict) -> dict:
        """Score the prediction with every enabled term and take one optimizer step.

        The optional lesion-local terms are only evaluated when their weight is non-zero.
        """
        from nnunetv2.utilities.helpers import dummy_context

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        acquired_image_mask = batch["acquired_image_mask"].to(
            self.device, non_blocking=True
        )
        lesion_mask = batch["lesion_mask"].to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(data)
            prediction_post = data + output
            target_post = data + target
            mse_loss = self._masked_mse(
                prediction_post, target_post, acquired_image_mask
            )
            lpips_loss = self._lpips_loss(
                prediction_post, target_post, acquired_image_mask
            )
            ssim_loss = self._ssim_loss(prediction_post, target_post, lesion_mask)
            dice_loss, tversky_loss, contour_recall_loss = self._segmentation_losses(
                prediction_post, lesion_mask, acquired_image_mask
            )
            contour_loss = (
                self._lesion_contour_loss(
                    prediction_post, target_post, lesion_mask, acquired_image_mask
                )
                if self.contour_weight != 0
                else mse_loss.new_zeros(())
            )
            moment_loss = (
                self._lesion_moment_loss(
                    prediction_post, target_post, lesion_mask, acquired_image_mask
                )
                if self.lesion_moment_weight != 0
                else mse_loss.new_zeros(())
            )
            hinge_loss = (
                self._under_enhancement_hinge(prediction_post, target_post, lesion_mask)
                if self.under_enhancement_weight != 0
                else mse_loss.new_zeros(())
            )
            loss = (
                mse_loss
                + self.lpips_weight * lpips_loss
                + self.ssim_weight * ssim_loss
                + self.dice_weight * dice_loss
                + self.tversky_weight * tversky_loss
            )
            if self.contour_weight != 0:
                loss = loss + self.contour_weight * contour_loss
            if self.contour_recall_weight != 0:
                loss = loss + self.contour_recall_weight * contour_recall_loss
            if self.lesion_moment_weight != 0:
                loss = loss + self.lesion_moment_weight * moment_loss
            if self.under_enhancement_weight != 0:
                loss = loss + self.under_enhancement_weight * hinge_loss
            baseline_mse = self._masked_mse(data, target_post, acquired_image_mask)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {
            "loss": loss.detach().cpu().numpy(),
            "mse": mse_loss.detach().cpu().numpy(),
            "baseline_mse": baseline_mse.detach().cpu().numpy(),
            "lpips": lpips_loss.detach().cpu().numpy(),
            "ssim": ssim_loss.detach().cpu().numpy(),
            "dice": dice_loss.detach().cpu().numpy(),
            "tversky": tversky_loss.detach().cpu().numpy(),
            "contour": contour_loss.detach().cpu().numpy(),
            "contour_recall": contour_recall_loss.detach().cpu().numpy(),
            "lesion_moment": moment_loss.detach().cpu().numpy(),
            "under_enhancement": hinge_loss.detach().cpu().numpy(),
        }

    def validation_step(self, batch: dict) -> dict:
        """Score one validation batch without updating the weights."""
        from nnunetv2.utilities.helpers import dummy_context

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        acquired_image_mask = batch["acquired_image_mask"].to(
            self.device, non_blocking=True
        )
        lesion_mask = batch["lesion_mask"].to(self.device, non_blocking=True)
        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(data)
            prediction_post = data + output
            target_post = data + target
            mse_loss = self._masked_mse(
                prediction_post, target_post, acquired_image_mask
            )
            lpips_loss = self._lpips_loss(
                prediction_post, target_post, acquired_image_mask
            )
            ssim_loss = self._ssim_loss(prediction_post, target_post, lesion_mask)
            dice_loss, tversky_loss, contour_recall_loss = self._segmentation_losses(
                prediction_post, lesion_mask, acquired_image_mask
            )
            contour_loss = (
                self._lesion_contour_loss(
                    prediction_post, target_post, lesion_mask, acquired_image_mask
                )
                if self.contour_weight != 0
                else mse_loss.new_zeros(())
            )
            moment_loss = (
                self._lesion_moment_loss(
                    prediction_post, target_post, lesion_mask, acquired_image_mask
                )
                if self.lesion_moment_weight != 0
                else mse_loss.new_zeros(())
            )
            hinge_loss = (
                self._under_enhancement_hinge(prediction_post, target_post, lesion_mask)
                if self.under_enhancement_weight != 0
                else mse_loss.new_zeros(())
            )
            loss = (
                mse_loss
                + self.lpips_weight * lpips_loss
                + self.ssim_weight * ssim_loss
                + self.dice_weight * dice_loss
                + self.tversky_weight * tversky_loss
            )
            if self.contour_weight != 0:
                loss = loss + self.contour_weight * contour_loss
            if self.contour_recall_weight != 0:
                loss = loss + self.contour_recall_weight * contour_recall_loss
            if self.lesion_moment_weight != 0:
                loss = loss + self.lesion_moment_weight * moment_loss
            if self.under_enhancement_weight != 0:
                loss = loss + self.under_enhancement_weight * hinge_loss
            baseline_mse = self._masked_mse(data, target_post, acquired_image_mask)
        return {
            "loss": loss.detach().cpu().numpy(),
            "mse": mse_loss.detach().cpu().numpy(),
            "baseline_mse": baseline_mse.detach().cpu().numpy(),
            "lpips": lpips_loss.detach().cpu().numpy(),
            "ssim": ssim_loss.detach().cpu().numpy(),
            "dice": dice_loss.detach().cpu().numpy(),
            "tversky": tversky_loss.detach().cpu().numpy(),
            "contour": contour_loss.detach().cpu().numpy(),
            "contour_recall": contour_recall_loss.detach().cpu().numpy(),
            "lesion_moment": moment_loss.detach().cpu().numpy(),
            "under_enhancement": hinge_loss.detach().cpu().numpy(),
        }

    def on_train_epoch_end(self, train_outputs):
        """Average the training metrics across ranks, log them and refresh the preview."""
        self._ensure_translation_logger_keys()
        outputs = collate_outputs(train_outputs)
        tversky_outputs = outputs.get("tversky", np.zeros_like(outputs["dice"]))
        if self.is_ddp:
            mses = [None for _ in range(dist.get_world_size())]
            baselines = [None for _ in range(dist.get_world_size())]
            lpipses = [None for _ in range(dist.get_world_size())]
            ssims = [None for _ in range(dist.get_world_size())]
            dices = [None for _ in range(dist.get_world_size())]
            tverskys = [None for _ in range(dist.get_world_size())]
            objectives = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(mses, outputs["mse"])
            dist.all_gather_object(baselines, outputs["baseline_mse"])
            dist.all_gather_object(lpipses, outputs["lpips"])
            dist.all_gather_object(ssims, outputs["ssim"])
            dist.all_gather_object(dices, outputs["dice"])
            dist.all_gather_object(tverskys, tversky_outputs)
            dist.all_gather_object(objectives, outputs["loss"])
            train_mse = np.vstack(mses).mean()
            train_baseline_mse = np.vstack(baselines).mean()
            train_lpips = np.vstack(lpipses).mean()
            train_ssim = np.vstack(ssims).mean()
            train_dice = np.vstack(dices).mean()
            train_tversky = np.vstack(tverskys).mean()
            train_objective = np.vstack(objectives).mean()
        else:
            train_mse = np.mean(outputs["mse"])
            train_baseline_mse = np.mean(outputs["baseline_mse"])
            train_lpips = np.mean(outputs["lpips"])
            train_ssim = np.mean(outputs["ssim"])
            train_dice = np.mean(outputs["dice"])
            train_tversky = np.mean(tversky_outputs)
            train_objective = np.mean(outputs["loss"])
        self._log_translation_metric("train_losses", "train_post_mses", train_mse)
        self._log_translation_metric(
            "train_baseline_mses",
            "train_identity_baseline_post_mses",
            train_baseline_mse,
        )
        self._log_translation_metric(
            "train_lpips_losses", "train_lpips_distances", train_lpips
        )
        self._log_translation_metric(
            "train_ssim_losses", "train_lesion_roi_ssim_losses", train_ssim
        )
        self._log_translation_metric(
            "train_dice_losses", "train_frozen_segmenter_soft_dice_losses", train_dice
        )
        self.logger.log("train_tversky_losses", train_tversky, self.current_epoch)
        if self.contour_weight != 0:
            contour_outputs = outputs.get("contour", np.zeros_like(outputs["dice"]))
            if self.is_ddp:
                gathered = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(gathered, contour_outputs)
                train_contour = np.vstack(gathered).mean()
            else:
                train_contour = np.mean(contour_outputs)
            self.logger.log("train_contour_losses", train_contour, self.current_epoch)
        if self.contour_recall_weight != 0:
            recall_outputs = outputs.get(
                "contour_recall", np.zeros_like(outputs["dice"])
            )
            if self.is_ddp:
                gathered = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(gathered, recall_outputs)
                train_contour_recall = np.vstack(gathered).mean()
            else:
                train_contour_recall = np.mean(recall_outputs)
            self.logger.log(
                "train_contour_recall_losses", train_contour_recall, self.current_epoch
            )
        self._log_translation_metric(
            "train_objective_losses", "train_total_objectives", train_objective
        )
        if self.local_rank == 0:
            try:
                self._save_example_batch_png()
            except Exception as exc:
                self.print_to_log_file(
                    f"Could not save LPIPS+SSIM+Dice translation example png: {exc}"
                )

    def on_validation_epoch_end(self, val_outputs):
        """Average the validation metrics across ranks and log them."""
        self._ensure_translation_logger_keys()
        outputs = collate_outputs(val_outputs)
        tversky_outputs = outputs.get("tversky", np.zeros_like(outputs["dice"]))
        if self.is_ddp:
            mses = [None for _ in range(dist.get_world_size())]
            baselines = [None for _ in range(dist.get_world_size())]
            lpipses = [None for _ in range(dist.get_world_size())]
            ssims = [None for _ in range(dist.get_world_size())]
            dices = [None for _ in range(dist.get_world_size())]
            tverskys = [None for _ in range(dist.get_world_size())]
            objectives = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(mses, outputs["mse"])
            dist.all_gather_object(baselines, outputs["baseline_mse"])
            dist.all_gather_object(lpipses, outputs["lpips"])
            dist.all_gather_object(ssims, outputs["ssim"])
            dist.all_gather_object(dices, outputs["dice"])
            dist.all_gather_object(tverskys, tversky_outputs)
            dist.all_gather_object(objectives, outputs["loss"])
            val_mse = np.vstack(mses).mean()
            val_baseline_mse = np.vstack(baselines).mean()
            val_lpips = np.vstack(lpipses).mean()
            val_ssim = np.vstack(ssims).mean()
            val_dice = np.vstack(dices).mean()
            val_tversky = np.vstack(tverskys).mean()
            val_objective = np.vstack(objectives).mean()
        else:
            val_mse = np.mean(outputs["mse"])
            val_baseline_mse = np.mean(outputs["baseline_mse"])
            val_lpips = np.mean(outputs["lpips"])
            val_ssim = np.mean(outputs["ssim"])
            val_dice = np.mean(outputs["dice"])
            val_tversky = np.mean(tversky_outputs)
            val_objective = np.mean(outputs["loss"])
        self._log_translation_metric("val_losses", "val_post_mses", val_mse)
        self._log_translation_metric(
            "val_baseline_mses", "val_identity_baseline_post_mses", val_baseline_mse
        )
        self._log_translation_metric(
            "val_lpips_losses", "val_lpips_distances", val_lpips
        )
        self._log_translation_metric(
            "val_ssim_losses", "val_lesion_roi_ssim_losses", val_ssim
        )
        self._log_translation_metric(
            "val_dice_losses", "val_frozen_segmenter_soft_dice_losses", val_dice
        )
        self.logger.log("val_tversky_losses", val_tversky, self.current_epoch)
        if self.contour_weight != 0:
            contour_outputs = outputs.get("contour", np.zeros_like(outputs["dice"]))
            if self.is_ddp:
                gathered = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(gathered, contour_outputs)
                val_contour = np.vstack(gathered).mean()
            else:
                val_contour = np.mean(contour_outputs)
            self.logger.log("val_contour_losses", val_contour, self.current_epoch)
        if self.contour_recall_weight != 0:
            recall_outputs = outputs.get(
                "contour_recall", np.zeros_like(outputs["dice"])
            )
            if self.is_ddp:
                gathered = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(gathered, recall_outputs)
                val_contour_recall = np.vstack(gathered).mean()
            else:
                val_contour_recall = np.mean(recall_outputs)
            self.logger.log(
                "val_contour_recall_losses", val_contour_recall, self.current_epoch
            )
        self._log_translation_metric(
            "val_objective_losses", "val_total_objectives", val_objective
        )

    def on_epoch_end(self):
        """Print the epoch summary, roll the checkpoints and redraw the progress plots.

        The segmentation term is passed to the checkpoint selector as well, so a
        separate best-by-segmentation checkpoint can be kept alongside the best overall.
        """
        from time import time

        from batchgenerators.utilities.file_and_folder_operations import join

        self.logger.log("epoch_end_timestamps", time(), self.current_epoch)
        self._ensure_translation_logger_keys()
        train_mse = float(self.logger.get_value("train_losses", step=-1))
        val_mse = float(self.logger.get_value("val_losses", step=-1))
        train_baseline_mse = float(
            self.logger.get_value("train_baseline_mses", step=-1)
        )
        val_baseline_mse = float(self.logger.get_value("val_baseline_mses", step=-1))
        train_lpips = float(self.logger.get_value("train_lpips_losses", step=-1))
        val_lpips = float(self.logger.get_value("val_lpips_losses", step=-1))
        train_ssim = float(self.logger.get_value("train_ssim_losses", step=-1))
        val_ssim = float(self.logger.get_value("val_ssim_losses", step=-1))
        train_dice = float(self.logger.get_value("train_dice_losses", step=-1))
        val_dice = float(self.logger.get_value("val_dice_losses", step=-1))
        train_tversky = float(self.logger.get_value("train_tversky_losses", step=-1))
        val_tversky = float(self.logger.get_value("val_tversky_losses", step=-1))
        train_objective = float(
            self.logger.get_value("train_objective_losses", step=-1)
        )
        val_objective = float(self.logger.get_value("val_objective_losses", step=-1))
        train_ratio = train_mse / max(train_baseline_mse, 1e-8)
        val_ratio = val_mse / max(val_baseline_mse, 1e-8)
        self.print_to_log_file(
            "train_total_objective",
            np.round(train_objective, decimals=6),
            "train_post_mse_valid_pixels",
            np.round(train_mse, decimals=6),
            "train_lpips_distance",
            np.round(train_lpips, decimals=6),
            "train_lesion_roi_ssim_loss",
            np.round(train_ssim, decimals=6),
            "train_frozen_segmenter_soft_dice_loss",
            np.round(train_dice, decimals=6),
            "train_frozen_segmenter_tversky_loss",
            np.round(train_tversky, decimals=6),
        )
        self.print_to_log_file(
            "train_identity_baseline_post_mse_valid_pixels",
            np.round(train_baseline_mse, decimals=6),
            "train_post_mse_over_identity_baseline",
            np.round(train_ratio, decimals=4),
        )
        self.print_to_log_file(
            "val_total_objective",
            np.round(val_objective, decimals=6),
            "val_post_mse_valid_pixels",
            np.round(val_mse, decimals=6),
            "val_lpips_distance",
            np.round(val_lpips, decimals=6),
            "val_lesion_roi_ssim_loss",
            np.round(val_ssim, decimals=6),
            "val_frozen_segmenter_soft_dice_loss",
            np.round(val_dice, decimals=6),
            "val_frozen_segmenter_tversky_loss",
            np.round(val_tversky, decimals=6),
        )
        self.print_to_log_file(
            "val_identity_baseline_post_mse_valid_pixels",
            np.round(val_baseline_mse, decimals=6),
            "val_post_mse_over_identity_baseline",
            np.round(val_ratio, decimals=4),
        )
        if self.contour_weight != 0:
            train_contour = float(
                self.logger.get_value("train_contour_losses", step=-1)
            )
            val_contour = float(self.logger.get_value("val_contour_losses", step=-1))
            self.print_to_log_file(
                "train_lesion_contour_mse",
                np.round(train_contour, decimals=6),
                "val_lesion_contour_mse",
                np.round(val_contour, decimals=6),
            )
        if self.contour_recall_weight != 0:
            train_contour_recall = float(
                self.logger.get_value("train_contour_recall_losses", step=-1)
            )
            val_contour_recall = float(
                self.logger.get_value("val_contour_recall_losses", step=-1)
            )
            self.print_to_log_file(
                "train_lesion_contour_recall_loss",
                np.round(train_contour_recall, decimals=6),
                "val_lesion_contour_recall_loss",
                np.round(val_contour_recall, decimals=6),
            )
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.get_value('epoch_end_timestamps', step=-1) - self.logger.get_value('epoch_start_timestamps', step=-1), decimals=2)} s"
        )

        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (
            self.num_epochs - 1
        ):
            self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))

        self._update_best_translation_checkpoints(
            val_objective,
            validation_mse=val_mse,
            validation_dice=val_dice,
        )

        if self.local_rank == 0:
            self._plot_mse_progress_png()
        self.current_epoch += 1

    def _plot_mse_progress_png(self):
        """Add a plot of the segmentation terms."""
        super()._plot_mse_progress_png()
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from batchgenerators.utilities.file_and_folder_operations import join

        epochs = list(range(len(self.logger.get_value("train_dice_losses", step=None))))
        if not epochs:
            return
        fig, ax = plt.subplots(1, 1, figsize=(14, 5))
        ax.plot(
            epochs,
            self.logger.get_value("train_dice_losses", step=None),
            label="train frozen-segmenter soft Dice loss",
            linewidth=2,
        )
        ax.plot(
            epochs,
            self.logger.get_value("val_dice_losses", step=None),
            label="validation frozen-segmenter soft Dice loss",
            linewidth=2,
        )
        ax.set_xlabel("epoch")
        ax.set_ylabel("1 - soft Dice")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            join(self.output_folder, "progress_frozen_segmenter_soft_dice_loss.png"),
            dpi=120,
        )
        plt.close(fig)


class nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuria(
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDice
):
    """Adds a feature-matching term against a frozen vision-transformer encoder.

    The generated and the real post image are both tokenized by the encoder and the
    per-token features compared, with extra weight on tokens covering the lesion.
    """

    batch_size_override = 1
    curia_weight = 0.2
    curia_lesion_weight = 0.5
    curia_microbatch_size = 2
    _curia_feature_model = None

    def __init__(
        self, plans, configuration, fold, dataset_json, device=torch.device("cuda")
    ):
        """Shorten the run and checkpoint every epoch."""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
        self.save_every = 1

    def _ensure_translation_logger_keys(self):
        """Add the feature-distance series to the logger."""
        super()._ensure_translation_logger_keys()
        local_logger = getattr(self.logger, "local_logger", None)
        if local_logger is None:
            return
        logging_dict = getattr(local_logger, "my_fantastic_logging", None)
        if logging_dict is None:
            return
        logging_dict.setdefault("train_curia_losses", [])
        logging_dict.setdefault("val_curia_losses", [])

    def _get_curia_feature_model(self):
        """Load the frozen encoder once and keep it in eval mode."""
        if self._curia_feature_model is not None:
            return self._curia_feature_model
        from nnunetv2.architectures.curia.common import load_curia_encoder

        model = load_curia_encoder().to(self.device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._curia_feature_model = model
        return model

    @staticmethod
    def _curia_zscore(
        image: torch.Tensor, acquired_image_mask: torch.Tensor
    ) -> torch.Tensor:
        """Per-slice z-score over the acquired region, with padding left at zero.

        The encoder expects standardised input. Statistics are taken over acquired pixels
        only so the amount of padding on the canvas does not shift them. Kept
        differentiable, since the prediction passes through it.
        """
        image_float = image.float()
        acquired_support = acquired_image_mask.float()
        count = acquired_support.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        mean = (image_float * acquired_support).sum(dim=(-2, -1), keepdim=True) / count
        variance = ((image_float - mean).square() * acquired_support).sum(
            dim=(-2, -1), keepdim=True
        ) / count
        normalized = (image_float - mean) / variance.clamp_min(1e-8).sqrt()
        return normalized * acquired_support

    @staticmethod
    def _curia_patch_tokens(model, image: torch.Tensor) -> torch.Tensor:
        """Return the encoder's last-layer patch tokens, dropping any prefix tokens."""
        outputs = model(pixel_values=image, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        patch_size = int(model.config.patch_size)
        num_patches = (image.shape[-2] // patch_size) * (image.shape[-1] // patch_size)
        return hidden[:, -num_patches:]

    @staticmethod
    def _mean_token_loss(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Average per-token values inside a token mask, per image then over the batch.

        Images whose mask selects no token are left out entirely rather than contributing
        a zero.
        """
        mask_float = mask.to(dtype=values.dtype)
        per_sample = (values * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(
            1.0
        )
        present = mask.any(dim=1)
        if present.any():
            return per_sample[present].mean()
        return values.sum() * 0.0

    def _curia_feature_loss(
        self,
        prediction_post: torch.Tensor,
        target_post: torch.Tensor,
        lesion_mask: torch.Tensor,
        acquired_image_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Cosine distance between the encoder's tokens for the prediction and the target.

        Only tokens whose patch lies fully inside the acquired region count, so no token
        is scored on canvas padding. The whole-image distance is added to a lesion-token
        distance, which puts extra weight on the region of interest. The batch is fed
        through the encoder in small chunks to bound activation memory, and the target
        branch runs without gradients.
        """
        from contextlib import nullcontext

        import torch.nn.functional as F

        model = self._get_curia_feature_model()
        patch_size = int(model.config.patch_size)
        prediction = self._curia_zscore(prediction_post, acquired_image_mask)
        target = self._curia_zscore(target_post, acquired_image_mask)
        losses = []
        batch_size = prediction.shape[0]
        # Feed the encoder in small chunks: its activations dominate memory at this input
        # size.
        for start in range(0, batch_size, self.curia_microbatch_size):
            end = min(start + self.curia_microbatch_size, batch_size)
            prediction_chunk = prediction[start:end]
            target_chunk = target[start:end]
            lesion_chunk = lesion_mask[start:end].float()
            acquired_image_chunk = acquired_image_mask[start:end].float()

            autocast_context = (
                torch.autocast(self.device.type, dtype=torch.bfloat16)
                if self.device.type == "cuda"
                else nullcontext()
            )
            with autocast_context:
                prediction_tokens = self._curia_patch_tokens(model, prediction_chunk)
                with torch.no_grad():
                    target_tokens = self._curia_patch_tokens(model, target_chunk)
            prediction_tokens = F.normalize(prediction_tokens.float(), dim=-1)
            target_tokens = F.normalize(target_tokens.float(), dim=-1)
            token_cosine_distance = 1.0 - (prediction_tokens * target_tokens).sum(
                dim=-1
            )

            lesion_tokens = (
                F.max_pool2d(
                    lesion_chunk, kernel_size=patch_size, stride=patch_size
                ).flatten(1)
                > 0
            )
            valid_fraction = F.avg_pool2d(
                acquired_image_chunk, kernel_size=patch_size, stride=patch_size
            ).flatten(1)
            # Keep only tokens whose patch lies entirely inside the acquired region, so no
            # token is scored partly on canvas padding.
            acquired_image_tokens = valid_fraction > 0.999
            lesion_tokens &= acquired_image_tokens
            whole_loss = self._mean_token_loss(
                token_cosine_distance, acquired_image_tokens
            )
            lesion_loss = self._mean_token_loss(token_cosine_distance, lesion_tokens)
            losses.append(
                (end - start) * (whole_loss + self.curia_lesion_weight * lesion_loss)
            )
        return torch.stack(losses).sum() / batch_size

    def train_step(self, batch: dict) -> dict:
        """Score the prediction with every term and take one optimizer step."""
        from nnunetv2.utilities.helpers import dummy_context

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        acquired_image_mask = batch["acquired_image_mask"].to(
            self.device, non_blocking=True
        )
        lesion_mask = batch["lesion_mask"].to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(data)
            prediction_post = data + output
            target_post = data + target
            mse_loss = self._masked_mse(
                prediction_post, target_post, acquired_image_mask
            )
            lpips_loss = self._lpips_loss(
                prediction_post, target_post, acquired_image_mask
            )
            ssim_loss = self._ssim_loss(prediction_post, target_post, lesion_mask)
            dice_loss = self._segmentation_dice_loss(
                prediction_post, lesion_mask, acquired_image_mask
            )
            curia_loss = self._curia_feature_loss(
                prediction_post, target_post, lesion_mask, acquired_image_mask
            )
            loss = (
                mse_loss
                + self.lpips_weight * lpips_loss
                + self.ssim_weight * ssim_loss
                + self.dice_weight * dice_loss
                + self.curia_weight * curia_loss
            )
            baseline_mse = self._masked_mse(data, target_post, acquired_image_mask)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {
            "loss": loss.detach().cpu().numpy(),
            "mse": mse_loss.detach().cpu().numpy(),
            "baseline_mse": baseline_mse.detach().cpu().numpy(),
            "lpips": lpips_loss.detach().cpu().numpy(),
            "ssim": ssim_loss.detach().cpu().numpy(),
            "dice": dice_loss.detach().cpu().numpy(),
            "curia": curia_loss.detach().cpu().numpy(),
        }

    def validation_step(self, batch: dict) -> dict:
        """Score one validation batch without updating the weights."""
        from nnunetv2.utilities.helpers import dummy_context

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        acquired_image_mask = batch["acquired_image_mask"].to(
            self.device, non_blocking=True
        )
        lesion_mask = batch["lesion_mask"].to(self.device, non_blocking=True)
        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(data)
            prediction_post = data + output
            target_post = data + target
            mse_loss = self._masked_mse(
                prediction_post, target_post, acquired_image_mask
            )
            lpips_loss = self._lpips_loss(
                prediction_post, target_post, acquired_image_mask
            )
            ssim_loss = self._ssim_loss(prediction_post, target_post, lesion_mask)
            dice_loss = self._segmentation_dice_loss(
                prediction_post, lesion_mask, acquired_image_mask
            )
            curia_loss = self._curia_feature_loss(
                prediction_post, target_post, lesion_mask, acquired_image_mask
            )
            loss = (
                mse_loss
                + self.lpips_weight * lpips_loss
                + self.ssim_weight * ssim_loss
                + self.dice_weight * dice_loss
                + self.curia_weight * curia_loss
            )
            baseline_mse = self._masked_mse(data, target_post, acquired_image_mask)
        return {
            "loss": loss.detach().cpu().numpy(),
            "mse": mse_loss.detach().cpu().numpy(),
            "baseline_mse": baseline_mse.detach().cpu().numpy(),
            "lpips": lpips_loss.detach().cpu().numpy(),
            "ssim": ssim_loss.detach().cpu().numpy(),
            "dice": dice_loss.detach().cpu().numpy(),
            "curia": curia_loss.detach().cpu().numpy(),
        }

    def _mean_epoch_output(self, outputs: dict, key: str) -> float:
        """Mean of one output key over the epoch, gathered across ranks when distributed."""
        if not self.is_ddp:
            return float(np.mean(outputs[key]))
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, outputs[key])
        return float(np.vstack(gathered).mean())

    def on_train_epoch_end(self, train_outputs):
        """Log the feature distance on top of the inherited training metrics."""
        super().on_train_epoch_end(train_outputs)
        outputs = collate_outputs(train_outputs)
        train_curia = self._mean_epoch_output(outputs, "curia")
        self._log_translation_metric(
            "train_curia_losses", "train_curia2_token_cosine_distances", train_curia
        )

    def on_validation_epoch_end(self, val_outputs):
        """Log the feature distance on top of the inherited validation metrics."""
        super().on_validation_epoch_end(val_outputs)
        outputs = collate_outputs(val_outputs)
        val_curia = self._mean_epoch_output(outputs, "curia")
        self._log_translation_metric(
            "val_curia_losses", "val_curia2_token_cosine_distances", val_curia
        )

    def on_epoch_end(self):
        """Print the feature distance, then run the inherited end-of-epoch handling."""
        self._ensure_translation_logger_keys()
        train_curia = float(self.logger.get_value("train_curia_losses", step=-1))
        val_curia = float(self.logger.get_value("val_curia_losses", step=-1))
        self.print_to_log_file(
            "train_curia2_token_cosine_distance",
            np.round(train_curia, decimals=6),
            "val_curia2_token_cosine_distance",
            np.round(val_curia, decimals=6),
            "curia_weight",
            self.curia_weight,
        )
        super().on_epoch_end()

    def _plot_mse_progress_png(self):
        """Add a plot of the encoder feature distance."""
        super()._plot_mse_progress_png()
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from batchgenerators.utilities.file_and_folder_operations import join

        epochs = list(
            range(len(self.logger.get_value("train_curia_losses", step=None)))
        )
        if not epochs:
            return
        fig, ax = plt.subplots(1, 1, figsize=(14, 5))
        ax.plot(
            epochs,
            self.logger.get_value("train_curia_losses", step=None),
            label="train Curia-2 token cosine distance",
            linewidth=2,
        )
        ax.plot(
            epochs,
            self.logger.get_value("val_curia_losses", step=None),
            label="validation Curia-2 token cosine distance",
            linewidth=2,
        )
        ax.set_xlabel("epoch")
        ax.set_ylabel("whole acquired image + 0.5 x lesion cosine distance")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            join(self.output_folder, "progress_curia2_feature_distance.png"), dpi=120
        )
        plt.close(fig)


class nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuriaGAN(
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuria
):
    """Adds a lightly weighted adversarial term from a patch discriminator.

    The discriminator is trained alongside the generator on real against generated
    post-contrast images, and the generator gets a small extra reward for fooling it.
    """

    gan_weight = 0.01
    discriminator_lr_ratio = 0.25
    # One-sided label smoothing: a real target just below 1 keeps the discriminator
    # from saturating and starving the generator of gradient.
    discriminator_real_label = 0.9
    discriminator_noise_std = 0.1
    discriminator_noise_epochs = 25

    def __init__(
        self, plans, configuration, fold, dataset_json, device=torch.device("cuda")
    ):
        """Declare the discriminator slots; they are filled during initialization."""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.discriminator = None
        self.discriminator_optimizer = None
        self.discriminator_lr_scheduler = None
        self.discriminator_grad_scaler = None
        self.adversarial_loss = torch.nn.BCEWithLogitsLoss()

    @staticmethod
    def _new_grad_scaler():
        """Build a gradient scaler, tolerating either torch API spelling."""
        try:
            return torch.GradScaler("cuda")
        except AttributeError:
            return torch.cuda.amp.GradScaler()

    def initialize(self):
        """Build the discriminator with its own optimizer, schedule and scaler.

        It runs at a fraction of the generator's learning rate, with the low first
        momentum that adversarial training usually wants, and is wrapped for distributed
        training alongside the generator.
        """
        super().initialize()
        from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler

        self.discriminator = MamaSynthPatchDiscriminator().to(self.device)
        discriminator_lr = self.initial_lr * self.discriminator_lr_ratio
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=discriminator_lr,
            betas=(0.5, 0.999),
            weight_decay=0.0,
        )
        self.discriminator_lr_scheduler = PolyLRScheduler(
            self.discriminator_optimizer, discriminator_lr, self.num_epochs
        )
        if self.device.type == "cuda":
            self.discriminator_grad_scaler = self._new_grad_scaler()
        if self.is_ddp:
            from torch.nn.parallel import DistributedDataParallel as DDP

            self.discriminator = DDP(self.discriminator, device_ids=[self.local_rank])

    def _ensure_translation_logger_keys(self):
        """Add the adversarial and discriminator-accuracy series to the logger."""
        super()._ensure_translation_logger_keys()
        local_logger = getattr(self.logger, "local_logger", None)
        if local_logger is None:
            return
        logging_dict = getattr(local_logger, "my_fantastic_logging", None)
        if logging_dict is None:
            return
        for key in (
            "train_gan_losses",
            "val_gan_losses",
            "train_discriminator_losses",
            "val_discriminator_losses",
            "train_discriminator_real_accuracies",
            "val_discriminator_real_accuracies",
            "train_discriminator_fake_accuracies",
            "val_discriminator_fake_accuracies",
            "train_discriminator_accuracies",
            "val_discriminator_accuracies",
            "discriminator_lrs",
            "discriminator_noise_stds",
        ):
            logging_dict.setdefault(key, [])

    def _discriminator_noise_level(self) -> float:
        """Noise added to the discriminator's input, decaying to zero over early epochs.

        Blurring the two distributions at the start keeps the discriminator from winning
        outright before the generator has learned anything.
        """
        progress = min(
            max(float(self.current_epoch), 0.0) / self.discriminator_noise_epochs, 1.0
        )
        return self.discriminator_noise_std * (1.0 - progress)

    @staticmethod
    def _set_requires_grad(module: torch.nn.Module, enabled: bool):
        """Turn gradient tracking on or off for every parameter of a module."""
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    def _discriminator_input(
        self,
        post_image: torch.Tensor,
        acquired_image_mask: torch.Tensor,
        noise_std: float = 0.0,
    ) -> torch.Tensor:
        """Stack the masked post image with its support mask, optionally adding noise.

        The support channel tells the discriminator which pixels are real acquisition, so
        it cannot simply read the padding outline as a tell.
        """
        acquired_support = acquired_image_mask.to(dtype=post_image.dtype)
        image = post_image * acquired_support
        if noise_std > 0:
            image = image + torch.randn_like(image) * noise_std * acquired_support
        return torch.cat((image, acquired_support), dim=1)

    @staticmethod
    def _discriminator_accuracy(
        real_logits: torch.Tensor, fake_logits: torch.Tensor
    ) -> tuple:
        """Fraction of real and of generated patches classified correctly, plus their mean."""
        real_accuracy = (real_logits > 0).float().mean()
        fake_accuracy = (fake_logits <= 0).float().mean()
        return real_accuracy, fake_accuracy, 0.5 * (real_accuracy + fake_accuracy)

    def _discriminator_step(
        self,
        prediction_post: torch.Tensor,
        target_post: torch.Tensor,
        acquired_image_mask: torch.Tensor,
    ) -> tuple:
        """Train the discriminator for one step on real and detached generated images.

        Detaching the prediction keeps this update out of the generator's graph. The real
        target is slightly below one, which softens the discriminator's confidence.
        """
        from nnunetv2.utilities.helpers import dummy_context

        self._set_requires_grad(self.discriminator, True)
        self.discriminator_optimizer.zero_grad(set_to_none=True)
        noise_std = self._discriminator_noise_level()
        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            real_logits = self.discriminator(
                self._discriminator_input(target_post, acquired_image_mask, noise_std)
            )
            fake_logits = self.discriminator(
                self._discriminator_input(
                    # Detach, so the discriminator update never reaches the generator's parameters.
                    prediction_post.detach(),
                    acquired_image_mask,
                    noise_std,
                )
            )
            discriminator_loss = 0.5 * (
                self.adversarial_loss(
                    real_logits,
                    torch.full_like(real_logits, self.discriminator_real_label),
                )
                + self.adversarial_loss(fake_logits, torch.zeros_like(fake_logits))
            )
        if self.discriminator_grad_scaler is not None:
            self.discriminator_grad_scaler.scale(discriminator_loss).backward()
            self.discriminator_grad_scaler.step(self.discriminator_optimizer)
            self.discriminator_grad_scaler.update()
        else:
            discriminator_loss.backward()
            self.discriminator_optimizer.step()
        return (
            discriminator_loss,
            *self._discriminator_accuracy(real_logits, fake_logits),
        )

    def _generator_gan_loss(
        self,
        prediction_post: torch.Tensor,
        acquired_image_mask: torch.Tensor,
        noise_std: float = 0.0,
    ) -> torch.Tensor:
        """The generator's cost for being recognised, with the discriminator in eval mode."""
        was_training = self.discriminator.training
        self.discriminator.eval()
        try:
            fake_logits = self.discriminator(
                self._discriminator_input(
                    prediction_post, acquired_image_mask, noise_std
                )
            )
        finally:
            if was_training:
                self.discriminator.train()
        return self.adversarial_loss(fake_logits, torch.ones_like(fake_logits))

    def train_step(self, batch: dict) -> dict:
        """Update the discriminator, then the generator on all terms including the adversarial one.

        The discriminator's parameters are frozen while the generator's gradient flows
        through it, and unfrozen again afterwards even if the step raises.
        """
        from nnunetv2.utilities.helpers import dummy_context

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        acquired_image_mask = batch["acquired_image_mask"].to(
            self.device, non_blocking=True
        )
        lesion_mask = batch["lesion_mask"].to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(data)
            prediction_post = data + output
            target_post = data + target

        discriminator_loss, real_accuracy, fake_accuracy, discriminator_accuracy = (
            self._discriminator_step(prediction_post, target_post, acquired_image_mask)
        )
        # The generator's gradient flows through the discriminator but must not accumulate
        # in it; the finally block restores the flags even if the step raises.
        self._set_requires_grad(self.discriminator, False)
        try:
            with (
                torch.autocast(self.device.type, enabled=True)
                if self.device.type == "cuda"
                else dummy_context()
            ):
                mse_loss = self._masked_mse(
                    prediction_post, target_post, acquired_image_mask
                )
                lpips_loss = self._lpips_loss(
                    prediction_post, target_post, acquired_image_mask
                )
                ssim_loss = self._ssim_loss(prediction_post, target_post, lesion_mask)
                dice_loss = self._segmentation_dice_loss(
                    prediction_post, lesion_mask, acquired_image_mask
                )
                curia_loss = self._curia_feature_loss(
                    prediction_post, target_post, lesion_mask, acquired_image_mask
                )
                gan_loss = self._generator_gan_loss(
                    prediction_post,
                    acquired_image_mask,
                    self._discriminator_noise_level(),
                )
                loss = (
                    mse_loss
                    + self.lpips_weight * lpips_loss
                    + self.ssim_weight * ssim_loss
                    + self.dice_weight * dice_loss
                    + self.curia_weight * curia_loss
                    + self.gan_weight * gan_loss
                )
                baseline_mse = self._masked_mse(data, target_post, acquired_image_mask)

            if self.grad_scaler is not None:
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()
        finally:
            self._set_requires_grad(self.discriminator, True)

        return {
            "loss": loss.detach().cpu().numpy(),
            "mse": mse_loss.detach().cpu().numpy(),
            "baseline_mse": baseline_mse.detach().cpu().numpy(),
            "lpips": lpips_loss.detach().cpu().numpy(),
            "ssim": ssim_loss.detach().cpu().numpy(),
            "dice": dice_loss.detach().cpu().numpy(),
            "curia": curia_loss.detach().cpu().numpy(),
            "gan": gan_loss.detach().cpu().numpy(),
            "discriminator": discriminator_loss.detach().cpu().numpy(),
            "discriminator_real_accuracy": real_accuracy.detach().cpu().numpy(),
            "discriminator_fake_accuracy": fake_accuracy.detach().cpu().numpy(),
            "discriminator_accuracy": discriminator_accuracy.detach().cpu().numpy(),
        }

    def validation_step(self, batch: dict) -> dict:
        """Score one validation batch, including both adversarial figures, without updating."""
        from nnunetv2.utilities.helpers import dummy_context

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        acquired_image_mask = batch["acquired_image_mask"].to(
            self.device, non_blocking=True
        )
        lesion_mask = batch["lesion_mask"].to(self.device, non_blocking=True)
        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(data)
            prediction_post = data + output
            target_post = data + target
            mse_loss = self._masked_mse(
                prediction_post, target_post, acquired_image_mask
            )
            lpips_loss = self._lpips_loss(
                prediction_post, target_post, acquired_image_mask
            )
            ssim_loss = self._ssim_loss(prediction_post, target_post, lesion_mask)
            dice_loss = self._segmentation_dice_loss(
                prediction_post, lesion_mask, acquired_image_mask
            )
            curia_loss = self._curia_feature_loss(
                prediction_post, target_post, lesion_mask, acquired_image_mask
            )
            real_logits = self.discriminator(
                self._discriminator_input(target_post, acquired_image_mask)
            )
            fake_logits = self.discriminator(
                self._discriminator_input(prediction_post, acquired_image_mask)
            )
            discriminator_loss = 0.5 * (
                self.adversarial_loss(
                    real_logits,
                    torch.full_like(real_logits, self.discriminator_real_label),
                )
                + self.adversarial_loss(fake_logits, torch.zeros_like(fake_logits))
            )
            gan_loss = self.adversarial_loss(fake_logits, torch.ones_like(fake_logits))
            real_accuracy, fake_accuracy, discriminator_accuracy = (
                self._discriminator_accuracy(real_logits, fake_logits)
            )
            loss = (
                mse_loss
                + self.lpips_weight * lpips_loss
                + self.ssim_weight * ssim_loss
                + self.dice_weight * dice_loss
                + self.curia_weight * curia_loss
                + self.gan_weight * gan_loss
            )
            baseline_mse = self._masked_mse(data, target_post, acquired_image_mask)
        return {
            "loss": loss.detach().cpu().numpy(),
            "mse": mse_loss.detach().cpu().numpy(),
            "baseline_mse": baseline_mse.detach().cpu().numpy(),
            "lpips": lpips_loss.detach().cpu().numpy(),
            "ssim": ssim_loss.detach().cpu().numpy(),
            "dice": dice_loss.detach().cpu().numpy(),
            "curia": curia_loss.detach().cpu().numpy(),
            "gan": gan_loss.detach().cpu().numpy(),
            "discriminator": discriminator_loss.detach().cpu().numpy(),
            "discriminator_real_accuracy": real_accuracy.detach().cpu().numpy(),
            "discriminator_fake_accuracy": fake_accuracy.detach().cpu().numpy(),
            "discriminator_accuracy": discriminator_accuracy.detach().cpu().numpy(),
        }

    def _log_gan_epoch_outputs(self, outputs: list, prefix: str):
        """Average and log the adversarial losses and discriminator accuracies."""
        collated = collate_outputs(outputs)
        for output_key, legacy_suffix, canonical_suffix in (
            ("gan", "gan_losses", "generator_adversarial_bces"),
            ("discriminator", "discriminator_losses", "discriminator_bces"),
            (
                "discriminator_real_accuracy",
                "discriminator_real_accuracies",
                "discriminator_real_patch_accuracies",
            ),
            (
                "discriminator_fake_accuracy",
                "discriminator_fake_accuracies",
                "discriminator_fake_patch_accuracies",
            ),
            (
                "discriminator_accuracy",
                "discriminator_accuracies",
                "discriminator_balanced_patch_accuracies",
            ),
        ):
            self._log_translation_metric(
                f"{prefix}_{legacy_suffix}",
                f"{prefix}_{canonical_suffix}",
                self._mean_epoch_output(collated, output_key),
            )

    def on_train_epoch_start(self):
        """Advance the discriminator's schedule and log its learning rate and noise level."""
        super().on_train_epoch_start()
        self.discriminator.train()
        self.discriminator_lr_scheduler.step(self.current_epoch)
        self._ensure_translation_logger_keys()
        discriminator_lr = self.discriminator_optimizer.param_groups[0]["lr"]
        noise_std = self._discriminator_noise_level()
        self.logger.log("discriminator_lrs", discriminator_lr, self.current_epoch)
        self.logger.log("discriminator_noise_stds", noise_std, self.current_epoch)
        self.print_to_log_file(
            f"Discriminator learning rate: {discriminator_lr:.7f}; input noise std: {noise_std:.4f}"
        )

    def on_validation_epoch_start(self):
        """Put the discriminator in eval mode for validation."""
        super().on_validation_epoch_start()
        self.discriminator.eval()

    def on_train_epoch_end(self, train_outputs):
        """Log the adversarial training figures on top of the inherited metrics."""
        super().on_train_epoch_end(train_outputs)
        self._log_gan_epoch_outputs(train_outputs, "train")

    def on_validation_epoch_end(self, val_outputs):
        """Log the adversarial validation figures on top of the inherited metrics."""
        super().on_validation_epoch_end(val_outputs)
        self._log_gan_epoch_outputs(val_outputs, "val")

    def on_epoch_end(self):
        """Print the adversarial summary, then run the inherited end-of-epoch handling."""
        self._ensure_translation_logger_keys()
        self.print_to_log_file(
            "train_generator_adversarial_bce_unweighted",
            np.round(self.logger.get_value("train_gan_losses", step=-1), 6),
            "train_discriminator_bce",
            np.round(self.logger.get_value("train_discriminator_losses", step=-1), 6),
            "train_discriminator_balanced_patch_accuracy",
            np.round(
                self.logger.get_value("train_discriminator_accuracies", step=-1), 4
            ),
            "train_discriminator_real_patch_accuracy",
            np.round(
                self.logger.get_value("train_discriminator_real_accuracies", step=-1), 4
            ),
            "train_discriminator_fake_patch_accuracy",
            np.round(
                self.logger.get_value("train_discriminator_fake_accuracies", step=-1), 4
            ),
        )
        self.print_to_log_file(
            "val_generator_adversarial_bce_unweighted",
            np.round(self.logger.get_value("val_gan_losses", step=-1), 6),
            "val_discriminator_bce",
            np.round(self.logger.get_value("val_discriminator_losses", step=-1), 6),
            "val_discriminator_balanced_patch_accuracy",
            np.round(self.logger.get_value("val_discriminator_accuracies", step=-1), 4),
            "val_discriminator_real_patch_accuracy",
            np.round(
                self.logger.get_value("val_discriminator_real_accuracies", step=-1), 4
            ),
            "val_discriminator_fake_patch_accuracy",
            np.round(
                self.logger.get_value("val_discriminator_fake_accuracies", step=-1), 4
            ),
            "gan_weight",
            self.gan_weight,
        )
        super().on_epoch_end()

    @staticmethod
    def _checkpoint_module(module: torch.nn.Module) -> torch.nn.Module:
        """Unwrap distributed and compiled wrappers to reach the plain module."""
        from torch._dynamo import OptimizedModule

        if hasattr(module, "module"):
            module = module.module
        if isinstance(module, OptimizedModule):
            module = module._orig_mod
        return module

    def save_checkpoint(self, filename: str) -> None:
        """Write a checkpoint carrying the discriminator's state as well.

        The base implementation knows nothing about the second network, so the whole
        checkpoint dictionary is rebuilt here to include its weights, optimizer, schedule
        and scaler; without them a resumed run would restart adversarial training from
        scratch against a partly trained generator.
        """
        if self.local_rank != 0:
            return
        if self.disable_checkpointing:
            self.print_to_log_file("No checkpoint written, checkpointing is disabled")
            return
        checkpoint = {
            "network_weights": self._checkpoint_module(self.network).state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "grad_scaler_state": (
                self.grad_scaler.state_dict() if self.grad_scaler is not None else None
            ),
            "discriminator_weights": self._checkpoint_module(
                self.discriminator
            ).state_dict(),
            "discriminator_optimizer_state": self.discriminator_optimizer.state_dict(),
            "discriminator_lr_scheduler_state": self.discriminator_lr_scheduler.state_dict(),
            "discriminator_grad_scaler_state": (
                self.discriminator_grad_scaler.state_dict()
                if self.discriminator_grad_scaler is not None
                else None
            ),
            "logging": self.logger.get_checkpoint(),
            "_best_ema": self._best_ema,
            "best_raw_validation_objective": getattr(
                self, "best_raw_validation_objective", None
            ),
            "validation_objective_ema": getattr(self, "validation_objective_ema", None),
            "validation_objective_ema_decay": getattr(
                self, "validation_objective_ema_decay", 0.8
            ),
            "current_epoch": self.current_epoch + 1,
            "init_args": self.my_init_kwargs,
            "trainer_name": self.__class__.__name__,
            "inference_allowed_mirroring_axes": self.inference_allowed_mirroring_axes,
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        """Restore both networks and all optimizer, schedule and selection state."""
        if not self.was_initialized:
            self.initialize()
        checkpoint = (
            torch.load(
                filename_or_checkpoint, map_location=self.device, weights_only=False
            )
            if isinstance(filename_or_checkpoint, str)
            else filename_or_checkpoint
        )
        network = self._checkpoint_module(self.network)
        state_dict = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in checkpoint["network_weights"].items()
        }
        network.load_state_dict(state_dict)
        self.my_init_kwargs = checkpoint["init_args"]
        self.current_epoch = checkpoint["current_epoch"]
        self.logger.load_checkpoint(checkpoint["logging"])
        self.validation_objective_ema_decay = checkpoint.get(
            "validation_objective_ema_decay",
            getattr(self, "validation_objective_ema_decay", 0.8),
        )
        self._ensure_translation_logger_keys()
        self._restore_validation_objective_selection_state()
        self.inference_allowed_mirroring_axes = checkpoint.get(
            "inference_allowed_mirroring_axes", self.inference_allowed_mirroring_axes
        )
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if (
            self.grad_scaler is not None
            and checkpoint.get("grad_scaler_state") is not None
        ):
            self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state"])
        if "discriminator_weights" not in checkpoint:
            self.print_to_log_file(
                "Checkpoint has no GAN state; initialized discriminator and its optimizer from scratch."
            )
            return
        self._checkpoint_module(self.discriminator).load_state_dict(
            checkpoint["discriminator_weights"]
        )
        self.discriminator_optimizer.load_state_dict(
            checkpoint["discriminator_optimizer_state"]
        )
        if "discriminator_lr_scheduler_state" in checkpoint:
            self.discriminator_lr_scheduler.load_state_dict(
                checkpoint["discriminator_lr_scheduler_state"]
            )
        if (
            self.discriminator_grad_scaler is not None
            and checkpoint.get("discriminator_grad_scaler_state") is not None
        ):
            self.discriminator_grad_scaler.load_state_dict(
                checkpoint["discriminator_grad_scaler_state"]
            )

    def _plot_mse_progress_png(self):
        """Add plots of the adversarial losses and the discriminator's accuracy."""
        super()._plot_mse_progress_png()
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from batchgenerators.utilities.file_and_folder_operations import join

        epochs = list(range(len(self.logger.get_value("train_gan_losses", step=None))))
        if not epochs:
            return
        fig, axes = plt.subplots(3, 1, figsize=(14, 14))
        axes[0].plot(
            epochs,
            self.logger.get_value("train_gan_losses", step=None),
            label="train generator adversarial BCE (unweighted)",
        )
        axes[0].plot(
            epochs,
            self.logger.get_value("val_gan_losses", step=None),
            label="validation generator adversarial BCE (unweighted)",
        )
        axes[1].plot(
            epochs,
            self.logger.get_value("train_discriminator_losses", step=None),
            label="train discriminator BCE",
        )
        axes[1].plot(
            epochs,
            self.logger.get_value("val_discriminator_losses", step=None),
            label="validation discriminator BCE",
        )
        axes[2].plot(
            epochs,
            self.logger.get_value("train_discriminator_accuracies", step=None),
            label="train balanced patch accuracy",
        )
        axes[2].plot(
            epochs,
            self.logger.get_value("train_discriminator_real_accuracies", step=None),
            label="train real-patch accuracy",
            linestyle="--",
        )
        axes[2].plot(
            epochs,
            self.logger.get_value("train_discriminator_fake_accuracies", step=None),
            label="train fake-patch accuracy",
            linestyle=":",
        )
        axes[2].plot(
            epochs,
            self.logger.get_value("val_discriminator_accuracies", step=None),
            label="validation balanced patch accuracy",
        )
        axes[2].plot(
            epochs,
            self.logger.get_value("val_discriminator_real_accuracies", step=None),
            label="validation real-patch accuracy",
            linestyle="--",
        )
        axes[2].plot(
            epochs,
            self.logger.get_value("val_discriminator_fake_accuracies", step=None),
            label="validation fake-patch accuracy",
            linestyle=":",
        )
        for axis in axes:
            axis.set_xlabel("epoch")
            axis.legend()
            axis.grid(True, alpha=0.3)
        axes[0].set_ylabel("BCE")
        axes[1].set_ylabel("BCE")
        axes[2].set_ylabel("accuracy")
        fig.tight_layout()
        fig.savefig(
            join(self.output_folder, "progress_adversarial_training.png"), dpi=120
        )
        plt.close(fig)


class nnUNetTrainerCuriaDPTMamaSynth(MamaSynthDataMixin, AbstractnnUNetTrainerCuria):
    """Segmentation with a frozen transformer encoder and a dense-prediction head."""

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=True,
    ):
        """Build the dense-prediction head on the transformer encoder."""
        from nnunetv2.architectures.curia.dpt import CuriaDPT

        return CuriaDPT(
            num_input_channels, num_output_channels, enable_deep_supervision
        )


class nnUNetTrainerCuriaDPTMamaSynth_BS_16(nnUNetTrainerCuriaDPTMamaSynth):
    """Dense-prediction-head segmenter at batch size 16."""

    batch_size_override = 16


class nnUNetTrainerCuriaUPerNetMamaSynth(
    MamaSynthDataMixin, AbstractnnUNetTrainerCuria
):
    """Segmentation with a frozen transformer encoder and a pyramid-pooling head."""

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=True,
    ):
        """Build the pyramid-pooling head on the transformer encoder."""
        from nnunetv2.architectures.curia.upernet import CuriaUPerNet

        return CuriaUPerNet(
            num_input_channels, num_output_channels, enable_deep_supervision
        )


class nnUNetTrainerCuriaUPerNetMamaSynth_BS_16(nnUNetTrainerCuriaUPerNetMamaSynth):
    """Pyramid-pooling-head segmenter at batch size 16."""

    batch_size_override = 16


class nnUNetTrainerMamaSynthTranslationLPIPSSSIMDice_BS_32(
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDice
):
    """Pixel, perceptual, structural and segmentation terms at batch size 32."""

    batch_size_override = 32


class nnUNetTrainerMamaSynthTranslationLPIPSSSIMDice_BS_48(
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDice
):
    """Pixel, perceptual, structural and segmentation terms at batch size 48."""

    batch_size_override = 48


class nnUNetTrainerMamaSynthTranslationLPIPSSSIMDice_BS_64(
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDice
):
    """Pixel, perceptual, structural and segmentation terms at batch size 64."""

    batch_size_override = 64


class nnUNetTrainerMamaSynthTranslationLPIPSSSIMDice_LPIPS05_BS_48(
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDice
):
    """Batch size 48 with the perceptual term at half weight and no feature term."""

    batch_size_override = 48
    lpips_weight = 0.5

    def _update_best_translation_checkpoints(
        self,
        validation_objective: float,
        validation_mse: float = None,
        validation_dice: float = None,
    ):
        """Keep only the averaged-best checkpoint for this trainer and its subclasses.

        The periodic latest and final checkpoints are still written elsewhere. The
        best-raw and per-sub-metric checkpoints are deliberately dropped to save disk; the
        running average is still tracked and logged, so selection and the plots are
        unaffected.
        """
        from batchgenerators.utilities.file_and_folder_operations import join

        if not np.isfinite(validation_objective):
            return
        decay = self.validation_objective_ema_decay
        if self.validation_objective_ema is None:
            self.validation_objective_ema = validation_objective
        else:
            self.validation_objective_ema = (
                decay * self.validation_objective_ema
                + (1.0 - decay) * validation_objective
            )
        self.logger.log(
            "val_total_objective_emas",
            self.validation_objective_ema,
            self.current_epoch,
        )
        self.print_to_log_file(
            "val_total_objective_raw",
            np.round(validation_objective, decimals=6),
            "val_total_objective_ema",
            np.round(self.validation_objective_ema, decimals=6),
        )
        if (
            self.best_raw_validation_objective is None
            or validation_objective < self.best_raw_validation_objective
        ):
            self.best_raw_validation_objective = validation_objective
        if self._best_ema is None or self.validation_objective_ema < self._best_ema:
            self._best_ema = self.validation_objective_ema
            self.print_to_log_file(
                f"New best EMA validation total objective: {self._best_ema:.6f} "
                f"(decay={self.validation_objective_ema_decay:.2f})"
            )
            self.save_checkpoint(join(self.output_folder, "checkpoint_best.pth"))


class nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceTverskyA02B08_LPIPS05_BS_48(
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDice_LPIPS05_BS_48
):
    """Adds the asymmetric Tversky term alongside the frozen segmenter's Dice.

    Weighting false negatives above false positives makes the segmenter's objective
    respond more strongly to lesion tissue the generator failed to render.
    """

    tversky_weight = 1.0

    def _update_best_translation_checkpoints(
        self,
        validation_objective: float,
        validation_mse: float = None,
        validation_dice: float = None,
    ):
        """Keep the averaged-best objective and averaged-best segmentation checkpoints.

        The periodic latest and final checkpoints are still written elsewhere; the
        best-raw and per-pixel-error checkpoints are not kept.
        """
        super()._update_best_translation_checkpoints(validation_objective)
        if validation_dice is not None:
            self._update_best_ema_metric_checkpoint(
                validation_dice,
                "soft_dice_loss",
                "val_frozen_segmenter_soft_dice_loss_emas",
                "checkpoint_best_dice.pth",
                "validation_dice_ema",
                "best_validation_dice_ema",
            )


class nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceTverskyA02B08DilatedGT_LPIPS05_BS_48(
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceTverskyA02B08_LPIPS05_BS_48
):
    """Dilates the lesion target by one pixel for the segmentation terms."""

    lesion_mask_dilation_radius = 1


class nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuria_BS_48(
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuria
):
    """Full objective including the encoder feature term, at batch size 48."""

    batch_size_override = 48


class nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuria_LPIPS05_Curia05_BS_48(
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuria
):
    """Batch size 48 with the perceptual and feature terms both at half weight."""

    batch_size_override = 48
    lpips_weight = 0.5
    curia_weight = 0.5


class nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuria_LPIPS05_Curia05_ResizeConv_BS_48(
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuria_LPIPS05_Curia05_BS_48
):
    """Architecture variant: resize-and-convolve upsampling instead of transposed convolutions.

    Transposed convolutions can leave a regular checkerboard pattern in the output;
    separating the upsampling from the filtering avoids it.
    """

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=True,
    ):
        """Build the standard network, then swap its decoder for the resize-convolution one."""
        network = nnUNetTrainer.build_network_architecture(
            plans_manager,
            configuration_manager,
            num_input_channels,
            1,
            enable_deep_supervision=False,
        )
        if not hasattr(network, "decoder") or not hasattr(
            network.decoder, "transpconvs"
        ):
            raise TypeError(
                "ResizeConv trainer requires a standard nnU-Net encoder-decoder architecture."
            )
        network.decoder = MamaSynthResizeConvUNetDecoder(network.decoder)
        if hasattr(network, "initialize"):
            network.decoder.upsample_convs.apply(network.initialize)
        return network


class nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuria_LPIPS05_Curia05_DS256128_BS_48(
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuria_LPIPS05_Curia05_BS_48
):
    """Adds auxiliary reconstruction supervision at two reduced resolutions.

    The decoder's intermediate heads are supervised against downsampled targets, so
    coarse structure is constrained before the full-resolution head refines it.
    """

    auxiliary_mse_256_weight = 0.25
    auxiliary_mse_128_weight = 0.125

    def __init__(
        self, plans, configuration, fold, dataset_json, device=torch.device("cuda")
    ):
        """Enable deep supervision, which the other translation trainers switch off."""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = True

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=True,
    ):
        """Build the single-output network with the deep-supervision heads retained."""
        return nnUNetTrainer.build_network_architecture(
            plans_manager,
            configuration_manager,
            num_input_channels,
            1,
            enable_deep_supervision=enable_deep_supervision,
        )

    def _ensure_translation_logger_keys(self):
        """Add the auxiliary-resolution series to the logger."""
        super()._ensure_translation_logger_keys()
        local_logger = getattr(self.logger, "local_logger", None)
        if local_logger is None:
            return
        logging_dict = getattr(local_logger, "my_fantastic_logging", None)
        if logging_dict is None:
            return
        logging_dict.setdefault("train_auxiliary_mse_256", [])
        logging_dict.setdefault("val_auxiliary_mse_256", [])
        logging_dict.setdefault("train_auxiliary_mse_128", [])
        logging_dict.setdefault("val_auxiliary_mse_128", [])

    def _multiscale_mse_losses(
        self,
        outputs,
        data: torch.Tensor,
        target_post: torch.Tensor,
        acquired_image_mask: torch.Tensor,
    ):
        """Score the two reduced-resolution heads against area-downsampled targets.

        Input, target and mask are all reduced with area averaging so the comparison stays
        consistent at each scale. Returns the full-resolution output together with the two
        auxiliary errors.
        """
        import torch.nn.functional as F

        if not isinstance(outputs, (list, tuple)) or len(outputs) < 3:
            raise RuntimeError(
                "Expected deep-supervision outputs at 512, 256, and 128 pixels."
            )
        auxiliary_losses = []
        for output_delta in outputs[1:3]:
            size = output_delta.shape[-2:]
            data_low = F.interpolate(data, size=size, mode="area")
            target_post_low = F.interpolate(target_post, size=size, mode="area")
            acquired_image_mask_low = F.interpolate(
                acquired_image_mask.float(), size=size, mode="area"
            )
            prediction_post_low = data_low + output_delta
            auxiliary_losses.append(
                self._masked_mse(
                    prediction_post_low, target_post_low, acquired_image_mask_low
                )
            )
        return outputs[0], auxiliary_losses[0], auxiliary_losses[1]

    def _forward_multiscale_losses(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        acquired_image_mask: torch.Tensor,
        lesion_mask: torch.Tensor,
    ):
        """Shared forward pass returning every loss term, full-resolution and auxiliary."""
        outputs = self.network(data)
        target_post = data + target
        output, auxiliary_mse_256, auxiliary_mse_128 = self._multiscale_mse_losses(
            outputs, data, target_post, acquired_image_mask
        )
        prediction_post = data + output
        mse_loss = self._masked_mse(prediction_post, target_post, acquired_image_mask)
        lpips_loss = self._lpips_loss(prediction_post, target_post, acquired_image_mask)
        ssim_loss = self._ssim_loss(prediction_post, target_post, lesion_mask)
        dice_loss = self._segmentation_dice_loss(
            prediction_post, lesion_mask, acquired_image_mask
        )
        curia_loss = self._curia_feature_loss(
            prediction_post, target_post, lesion_mask, acquired_image_mask
        )
        loss = (
            mse_loss
            + self.lpips_weight * lpips_loss
            + self.ssim_weight * ssim_loss
            + self.dice_weight * dice_loss
            + self.curia_weight * curia_loss
            + self.auxiliary_mse_256_weight * auxiliary_mse_256
            + self.auxiliary_mse_128_weight * auxiliary_mse_128
        )
        baseline_mse = self._masked_mse(data, target_post, acquired_image_mask)
        return {
            "loss": loss,
            "mse": mse_loss,
            "baseline_mse": baseline_mse,
            "lpips": lpips_loss,
            "ssim": ssim_loss,
            "dice": dice_loss,
            "curia": curia_loss,
            "auxiliary_mse_256": auxiliary_mse_256,
            "auxiliary_mse_128": auxiliary_mse_128,
        }

    @staticmethod
    def _detach_step_outputs(losses: dict) -> dict:
        """Convert a dictionary of loss tensors into plain arrays for logging."""
        return {key: value.detach().cpu().numpy() for key, value in losses.items()}

    def train_step(self, batch: dict) -> dict:
        """Run the shared forward pass and take one optimizer step."""
        from nnunetv2.utilities.helpers import dummy_context

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        acquired_image_mask = batch["acquired_image_mask"].to(
            self.device, non_blocking=True
        )
        lesion_mask = batch["lesion_mask"].to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            losses = self._forward_multiscale_losses(
                data, target, acquired_image_mask, lesion_mask
            )

        if self.grad_scaler is not None:
            self.grad_scaler.scale(losses["loss"]).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return self._detach_step_outputs(losses)

    def validation_step(self, batch: dict) -> dict:
        """Run the shared forward pass without updating the weights."""
        from nnunetv2.utilities.helpers import dummy_context

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        acquired_image_mask = batch["acquired_image_mask"].to(
            self.device, non_blocking=True
        )
        lesion_mask = batch["lesion_mask"].to(self.device, non_blocking=True)
        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            losses = self._forward_multiscale_losses(
                data, target, acquired_image_mask, lesion_mask
            )
        return self._detach_step_outputs(losses)

    def _log_auxiliary_epoch_outputs(self, outputs, prefix: str):
        """Average and log the two auxiliary-resolution errors."""
        collated = collate_outputs(outputs)
        auxiliary_mse_256 = self._mean_epoch_output(collated, "auxiliary_mse_256")
        auxiliary_mse_128 = self._mean_epoch_output(collated, "auxiliary_mse_128")
        self.logger.log(
            f"{prefix}_auxiliary_mse_256", auxiliary_mse_256, self.current_epoch
        )
        self.logger.log(
            f"{prefix}_auxiliary_mse_128", auxiliary_mse_128, self.current_epoch
        )

    def on_train_epoch_end(self, train_outputs):
        """Log the auxiliary errors on top of the inherited training metrics."""
        super().on_train_epoch_end(train_outputs)
        self._log_auxiliary_epoch_outputs(train_outputs, "train")

    def on_validation_epoch_end(self, val_outputs):
        """Log the auxiliary errors on top of the inherited validation metrics."""
        super().on_validation_epoch_end(val_outputs)
        self._log_auxiliary_epoch_outputs(val_outputs, "val")

    def on_epoch_end(self):
        """Print the auxiliary errors, then run the inherited end-of-epoch handling."""
        self._ensure_translation_logger_keys()
        self.print_to_log_file(
            "train_auxiliary_mse_256",
            np.round(self.logger.get_value("train_auxiliary_mse_256", step=-1), 6),
            "val_auxiliary_mse_256",
            np.round(self.logger.get_value("val_auxiliary_mse_256", step=-1), 6),
            "auxiliary_mse_256_weight",
            self.auxiliary_mse_256_weight,
        )
        self.print_to_log_file(
            "train_auxiliary_mse_128",
            np.round(self.logger.get_value("train_auxiliary_mse_128", step=-1), 6),
            "val_auxiliary_mse_128",
            np.round(self.logger.get_value("val_auxiliary_mse_128", step=-1), 6),
            "auxiliary_mse_128_weight",
            self.auxiliary_mse_128_weight,
        )
        super().on_epoch_end()


class nnUNetTrainerMamaSynthTranslationLPIPSSSIMDice_LPIPS05_BS_48_100_ep(
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDice
):
    """Half-weight perceptual term at batch size 48, run for 100 epochs."""

    batch_size_override = 48
    lpips_weight = 0.5

    def __init__(
        self, plans, configuration, fold, dataset_json, device=torch.device("cuda")
    ):
        """Shorten the run to 100 epochs."""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuria_LPIPS10_Curia05_BS_48(
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuria
):
    """Full-weight perceptual term with the feature term at half weight."""

    batch_size_override = 48
    lpips_weight = 1.0
    curia_weight = 0.5


class nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuria_BS_48_mama_mia(
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuria
):
    """Feature-term trainer at batch size 48 for the alternative dataset."""

    batch_size_override = 48


class nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuriaGAN_LPIPS05_Curia05_GAN02_BS_48(
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuriaGAN
):
    """Adversarial variant with a weaker perceptual term and stronger feature and adversarial terms."""

    batch_size_override = 48
    lpips_weight = 0.5
    curia_weight = 0.5
    gan_weight = 0.02


class nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuriaGAN_BS_48(
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuriaGAN
):
    """Adversarial trainer at batch size 48."""

    batch_size_override = 48


class nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuria_LPIPS05_Curia05_ResizeConv_BS_1(
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuria_LPIPS05_Curia05_ResizeConv_BS_48
):
    """The resize-convolution variant at batch size 1, to isolate batch-size effects."""

    batch_size_override = 1


class WarmupThenPolyLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warm-up to a peak learning rate, then polynomial decay to zero.

    ``step`` accepts an absolute step index, which is how the training loop calls it
    once per epoch, so the schedule lands in the right place after a resume rather
    than restarting from the beginning of the warm-up.
    """

    def __init__(
        self,
        optimizer,
        peak_lr: float,
        max_steps: int,
        warmup_steps: int,
        exponent: float = 0.9,
        current_step: int = None,
    ):
        """Record the peak rate and the warm-up and total lengths."""
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.max_steps = max_steps
        self.warmup_steps = max(1, int(warmup_steps))
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def step(self, current_step=None):
        """Set the learning rate for the given step, ramping up then decaying."""
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        if current_step < self.warmup_steps:
            new_lr = self.peak_lr * (current_step + 1) / self.warmup_steps
        else:
            decay_steps = max(1, self.max_steps - self.warmup_steps)
            progress = min(1.0, (current_step - self.warmup_steps) / decay_steps)
            new_lr = self.peak_lr * (1 - progress) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def get_last_lr(self):
        """Return the most recently applied learning rate per parameter group."""
        return self._last_lr


class MamaSynthMAEFinetuneMixin:
    """Fine-tuning schedule for translation trainers starting from MAE weights.

    The pretrained network and the translation network share an architecture, so the
    standard weight loader transfers both encoder and decoder directly and leaves only
    the task-specific output head randomly initialised. Pass the checkpoint with
    ``-pretrained_weights``.

    The schedule differs from training from scratch in three ways: a long linear
    warm-up precedes the decay, the peak learning rate is whatever the host trainer
    sets rather than the framework default, and the encoder is trained throughout
    instead of being frozen.
    """

    warmup_iterations = 12_500
    freeze_encoder_during_warmup = False

    def __init__(
        self, plans, configuration, fold, dataset_json, device=torch.device("cuda")
    ):
        """Convert the warm-up length from iterations into whole epochs."""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.warmup_epochs = max(
            1, round(self.warmup_iterations / self.num_iterations_per_epoch)
        )

    def configure_optimizers(self):
        """AdamW with linear warm-up followed by polynomial decay.

        The warm-up is capped at half the run so short schedules still spend time decaying.
        """
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
        )
        warmup_epochs = min(self.warmup_epochs, max(1, self.num_epochs // 2))
        scheduler = WarmupThenPolyLRScheduler(
            optimizer, self.initial_lr, self.num_epochs, warmup_epochs
        )
        return optimizer, scheduler

    def _set_encoder_trainable(self, trainable: bool):
        """Turn gradient tracking on or off for the encoder only."""
        network = self.network.module if self.is_ddp else self.network
        encoder = getattr(network, "encoder", None)
        if encoder is None:
            raise AttributeError(
                "Expected an encoder-decoder network with a `.encoder` attribute to (un)freeze."
            )
        for parameter in encoder.parameters():
            parameter.requires_grad = trainable

    def on_train_epoch_start(self):
        """Unfreeze the encoder once warm-up ends, if freezing was requested."""
        if self.freeze_encoder_during_warmup:
            self._set_encoder_trainable(self.current_epoch >= self.warmup_epochs)
        super().on_train_epoch_start()

    def on_train_start(self):
        """Record the fine-tuning schedule in the training log."""
        super().on_train_start()
        self.print_to_log_file(
            f"MAE fine-tune schedule: linear LR warm-up over {self.warmup_epochs} epochs "
            f"(~{self.warmup_iterations} steps) to peak LR {self.initial_lr}, then PolyLR decay over "
            f"{self.num_epochs} total epochs. freeze_encoder_during_warmup="
            f"{self.freeze_encoder_during_warmup}. Pass the MAE checkpoint via -pretrained_weights to "
            f"transfer encoder+decoder."
        )


class nnUNetTrainerMamaSynthTranslationLPIPSSSIMDice_LPIPS05_BS_48_MAEFinetune(
    MamaSynthMAEFinetuneMixin,
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDice_LPIPS05_BS_48,
):
    """Fine-tune of the half-weight perceptual trainer from MAE weights."""


class nnUNetTrainerMamaSynthTranslationLPIPSSSIMDice_LPIPS05_BS_48_MAEFinetune_NoRot(
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDice_LPIPS05_BS_48_MAEFinetune
):
    """Same fine-tune with random rotation dropped from the augmentation."""

    @staticmethod
    def get_translation_training_transforms(patch_size, rotation_for_DA, mirror_axes):
        """Spatial augmentation with mirroring only; rotation probability is zero."""
        from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
        from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
        from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms

        transforms = [
            SpatialTransform(
                patch_size,
                patch_center_dist_from_border=0,
                random_crop=False,
                p_elastic_deform=0,
                p_rotation=0.0,
                rotation=rotation_for_DA,
                p_scaling=0.0,
                scaling=(1.0, 1.0),
                p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False,
                mode_seg="nearest",
                border_mode_seg="zeros",
            )
        ]
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(MirrorTransform(allowed_axes=mirror_axes))
        return ComposeTransforms(transforms)


class nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceTverskyA02B08_LPIPS05_BS_48_MAEFinetune(
    MamaSynthMAEFinetuneMixin,
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceTverskyA02B08_LPIPS05_BS_48,
):
    """Fine-tune of the Tversky trainer from MAE weights."""


class nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceTverskyA02B08_LPIPS05_BS_48_MAEFinetune_LesionContour(
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceTverskyA02B08_LPIPS05_BS_48_MAEFinetune
):
    """Tversky fine-tune with extra reconstruction weight on the lesion boundary band.

    An additional masked error over a ring around the lesion outline, leaving the rest
    of the objective and the checkpoint policy unchanged.
    """

    contour_weight = 1.0
    contour_dilation_radius = 3
    contour_erosion_radius = 3


class nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceTverskyA02B08_LPIPS05_BS_48_MAEFinetune_LesionContourSkelRecall(
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceTverskyA02B08_LPIPS05_BS_48_MAEFinetune_LesionContour
):
    """Boundary-band variant that also asks the segmenter to fire along the whole margin.

    Adds a soft recall of the frozen segmenter's lesion probability over the same band.
    It reuses the segmenter pass the Dice term already performs, so it costs no extra
    inference.
    """

    contour_recall_weight = 0.5


class nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceTverskyA02B08_LPIPS05_BS_48_MAEFinetune_LesionMomentMatch(
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceTverskyA02B08_LPIPS05_BS_48_MAEFinetune
):
    """Tversky fine-tune that also matches the lesion's mean and spread to the real post.

    A per-pixel error pulls the lesion toward the average of its plausible
    appearances, which comes out flatter and dimmer than any real one. Matching the
    first two moments to the real target restores its level and internal variation
    while leaving the rest of the image alone.
    """

    lesion_moment_weight = 0.5
    moment_std_weight = 0.5


class nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceTverskyA02B08_LPIPS05_BS_48_MAEFinetune_LesionMomentMatchMeanOnly(
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceTverskyA02B08_LPIPS05_BS_48_MAEFinetune_LesionMomentMatch
):
    """Moment-matching variant that constrains only the lesion mean.

    Dropping the spread term separates the effect of raising the lesion's overall
    level from that of matching its internal variation.
    """

    moment_std_weight = 0.0


class nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceTverskyA02B08_LPIPS05_BS_48_MAEFinetune_LesionUnderEnhanceHinge(
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceTverskyA02B08_LPIPS05_BS_48_MAEFinetune
):
    """Tversky fine-tune with a one-sided penalty on lesion pixels darker than the target.

    Only shortfalls are penalised, never excess brightness. The asymmetry can compress
    the dim end of the lesion and so reduce its internal variation, which is why the
    weight is kept modest.
    """

    under_enhancement_weight = 0.5


class MamaSynthLesionSegChannelMixin:
    """Feeds the lesion mask to the network as a second input channel."""

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=True,
    ):
        """Build the network with one extra input channel for the mask."""
        return nnUNetTrainer.build_network_architecture(
            plans_manager,
            configuration_manager,
            num_input_channels + 1,
            1,
            enable_deep_supervision=False,
        )

    def _run_step_with_lesion_seg_channel(self, step, batch: dict) -> dict:
        """Run an inherited step with the mask concatenated onto the network input.

        A temporary forward hook appends the channel just before the network runs, so the
        inherited train and validation steps work unchanged instead of being duplicated.
        The hook is always removed afterwards.
        """
        lesion_mask = batch["lesion_mask"].to(self.device, non_blocking=True)
        lesion_mask = (lesion_mask > 0.5).to(batch["data"].dtype)

        def append_lesion_mask(_module, inputs):
            """Concatenate the mask onto the incoming image tensor."""
            if len(inputs) != 1:
                raise RuntimeError(
                    "LesionSegChannel expects the translation network to receive one image tensor."
                )
            image = inputs[0]
            if (
                image.shape[0] != lesion_mask.shape[0]
                or image.shape[2:] != lesion_mask.shape[2:]
            ):
                raise RuntimeError(
                    f"Image shape {tuple(image.shape)} and lesion-mask shape "
                    f"{tuple(lesion_mask.shape)} are incompatible."
                )
            return (torch.cat((image, lesion_mask.to(image.dtype)), dim=1),)

        hook = self.network.register_forward_pre_hook(append_lesion_mask)
        try:
            return step(batch)
        finally:
            hook.remove()

    def train_step(self, batch: dict) -> dict:
        """Run the inherited training step with the extra input channel."""
        return self._run_step_with_lesion_seg_channel(super().train_step, batch)

    def validation_step(self, batch: dict) -> dict:
        """Run the inherited validation step with the extra input channel."""
        return self._run_step_with_lesion_seg_channel(super().validation_step, batch)

    def _example_network_forward(self, data: torch.Tensor, batch: dict) -> torch.Tensor:
        """Forward for the preview image, with the mask channel appended as in training."""
        lesion_mask = batch["lesion_mask"].to(self.device, non_blocking=True)
        lesion_mask = (lesion_mask > 0.5).to(data.dtype)
        return self.network(torch.cat((data, lesion_mask), dim=1))


class nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceTverskyA02B08_LPIPS05_BS_48_MAEFinetune_LesionSegChannel(
    MamaSynthLesionSegChannelMixin,
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceTverskyA02B08_LPIPS05_BS_48_MAEFinetune,
):
    """Tversky fine-tune taking the lesion mask as a second input channel."""


class nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceTverskyA02B08_LPIPS05_BS_48_MAEFinetune_LesionMomentMatch_LesionSegChannel(
    MamaSynthLesionSegChannelMixin,
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceTverskyA02B08_LPIPS05_BS_48_MAEFinetune_LesionMomentMatch,
):
    """Mean-and-spread moment matching, with the lesion mask as a second input channel."""


class nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceTverskyA02B08_LPIPS05_BS_48_MAEFinetune_LesionMomentMatchMeanOnly_LesionSegChannel(
    MamaSynthLesionSegChannelMixin,
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceTverskyA02B08_LPIPS05_BS_48_MAEFinetune_LesionMomentMatchMeanOnly,
):
    """Mean-only moment matching, with the lesion mask as a second input channel."""


class nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceTverskyA02B08_LPIPS05_BS_48_MAEFinetune_LesionUnderEnhanceHinge_LesionSegChannel(
    MamaSynthLesionSegChannelMixin,
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceTverskyA02B08_LPIPS05_BS_48_MAEFinetune_LesionUnderEnhanceHinge,
):
    """One-sided shortfall penalty, with the lesion mask as a second input channel."""


class nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceTverskyA02B08_LPIPS05_BS_48_MAEFinetune_NoRot(
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceTverskyA02B08_LPIPS05_BS_48_MAEFinetune
):
    """Tversky fine-tune with random rotation dropped from the augmentation."""

    @staticmethod
    def get_translation_training_transforms(patch_size, rotation_for_DA, mirror_axes):
        """Spatial augmentation with mirroring only; rotation probability is zero."""
        from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
        from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
        from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms

        transforms = [
            SpatialTransform(
                patch_size,
                patch_center_dist_from_border=0,
                random_crop=False,
                p_elastic_deform=0,
                p_rotation=0.0,
                rotation=rotation_for_DA,
                p_scaling=0.0,
                scaling=(1.0, 1.0),
                p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False,
                mode_seg="nearest",
                border_mode_seg="zeros",
            )
        ]
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(MirrorTransform(allowed_axes=mirror_axes))
        return ComposeTransforms(transforms)


class nnUNetTrainerMamaSynthTranslationLPIPSSSIMDice_LPIPS05_BS_48_MAEFinetune_LR1e4(
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDice_LPIPS05_BS_48_MAEFinetune
):
    """Fine-tune at a ten-times lower peak learning rate."""

    def __init__(
        self, plans, configuration, fold, dataset_json, device=torch.device("cuda")
    ):
        """Lower the peak learning rate."""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-4


class nnUNetTrainerMamaSynthTranslationLPIPSSSIMDice_LPIPS05_BS_48_MAEFinetune_FrozenWarmup(
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDice_LPIPS05_BS_48_MAEFinetune
):
    """Fine-tune variant that holds the encoder frozen until warm-up ends."""

    freeze_encoder_during_warmup = True


class nnUNetTrainerMamaSynthTranslationLPIPSCuriaTokenMSE_LPIPS05_Curia05_BS_48(
    nnUNetTrainerMamaSynthTranslationLPIPS
):
    """Whole-image objective only: pixel error, perceptual distance, encoder token error.

    Every term is global. There is no lesion structural term and no segmentation term.
    The encoder term is a plain squared error between the prediction's and the
    target's patch tokens, averaged over feature dimensions and over tokens that lie
    fully inside the acquired region. The features are compared unnormalised: scaling
    them to unit length would reduce the term to a rescaled cosine distance and
    discard the magnitude information.
    """

    batch_size_override = 48
    lpips_weight = 0.5
    curia_weight = 0.5
    curia_microbatch_size = 2
    _curia_feature_model = None

    def _ensure_translation_logger_keys(self):
        """Add the token-error series to the logger."""
        super()._ensure_translation_logger_keys()
        local_logger = getattr(self.logger, "local_logger", None)
        if local_logger is None:
            return
        logging_dict = getattr(local_logger, "my_fantastic_logging", None)
        if logging_dict is None:
            return
        logging_dict.setdefault("train_curia_losses", [])
        logging_dict.setdefault("val_curia_losses", [])

    def _get_curia_feature_model(self):
        """Load the frozen encoder once and keep it in eval mode."""
        if self._curia_feature_model is not None:
            return self._curia_feature_model
        from nnunetv2.architectures.curia.common import load_curia_encoder

        model = load_curia_encoder().to(self.device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._curia_feature_model = model
        return model

    @staticmethod
    def _curia_zscore(
        image: torch.Tensor, acquired_image_mask: torch.Tensor
    ) -> torch.Tensor:
        """Per-slice z-score over the acquired region, with padding left at zero."""
        image_float = image.float()
        acquired_support = acquired_image_mask.float()
        count = acquired_support.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        mean = (image_float * acquired_support).sum(dim=(-2, -1), keepdim=True) / count
        variance = ((image_float - mean).square() * acquired_support).sum(
            dim=(-2, -1), keepdim=True
        ) / count
        normalized = (image_float - mean) / variance.clamp_min(1e-8).sqrt()
        return normalized * acquired_support

    @staticmethod
    def _curia_patch_tokens(model, image: torch.Tensor) -> torch.Tensor:
        """Return the encoder's last-layer patch tokens, dropping any prefix tokens."""
        outputs = model(pixel_values=image, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        patch_size = int(model.config.patch_size)
        num_patches = (image.shape[-2] // patch_size) * (image.shape[-1] // patch_size)
        return hidden[:, -num_patches:]

    @staticmethod
    def _masked_token_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Average per-token values inside a token mask, per image then over the batch.

        Images whose mask selects no token are left out rather than contributing a zero.
        """
        mask_float = mask.to(dtype=values.dtype)
        per_sample = (values * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(
            1.0
        )
        present = mask.any(dim=1)
        if present.any():
            return per_sample[present].mean()
        return values.sum() * 0.0

    def _curia_token_mse_loss(
        self,
        prediction_post: torch.Tensor,
        target_post: torch.Tensor,
        acquired_image_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Squared error between the encoder's tokens for the prediction and the target.

        Only tokens whose patch lies fully inside the acquired region are scored. The
        batch goes through the encoder in small chunks to bound activation memory, and the
        target branch runs without gradients.
        """
        from contextlib import nullcontext

        import torch.nn.functional as F

        model = self._get_curia_feature_model()
        patch_size = int(model.config.patch_size)
        prediction = self._curia_zscore(prediction_post, acquired_image_mask)
        target = self._curia_zscore(target_post, acquired_image_mask)
        losses = []
        batch_size = prediction.shape[0]
        for start in range(0, batch_size, self.curia_microbatch_size):
            end = min(start + self.curia_microbatch_size, batch_size)
            prediction_chunk = prediction[start:end]
            target_chunk = target[start:end]
            acquired_image_chunk = acquired_image_mask[start:end].float()

            autocast_context = (
                torch.autocast(self.device.type, dtype=torch.bfloat16)
                if self.device.type == "cuda"
                else nullcontext()
            )
            with autocast_context:
                prediction_tokens = self._curia_patch_tokens(model, prediction_chunk)
                with torch.no_grad():
                    target_tokens = self._curia_patch_tokens(model, target_chunk)
            prediction_tokens = prediction_tokens.float()
            target_tokens = target_tokens.float()
            valid_fraction = F.avg_pool2d(
                acquired_image_chunk, kernel_size=patch_size, stride=patch_size
            ).flatten(1)
            acquired_image_tokens = valid_fraction > 0.999
            # Compared without normalising the features to unit length, which would collapse
            # this into a rescaled cosine distance and throw away magnitude.
            per_token_mse = (prediction_tokens - target_tokens).square().mean(dim=-1)
            losses.append(
                (end - start)
                * self._masked_token_mean(per_token_mse, acquired_image_tokens)
            )
        return torch.stack(losses).sum() / batch_size

    def train_step(self, batch: dict) -> dict:
        """Score the prediction with all three whole-image terms and take one optimizer step."""
        from nnunetv2.utilities.helpers import dummy_context

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        acquired_image_mask = batch["acquired_image_mask"].to(
            self.device, non_blocking=True
        )
        self.optimizer.zero_grad(set_to_none=True)
        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(data)
            prediction_post = data + output
            target_post = data + target
            mse_loss = self._masked_mse(
                prediction_post, target_post, acquired_image_mask
            )
            lpips_loss = self._lpips_loss(
                prediction_post, target_post, acquired_image_mask
            )
            curia_loss = self._curia_token_mse_loss(
                prediction_post, target_post, acquired_image_mask
            )
            loss = (
                mse_loss
                + self.lpips_weight * lpips_loss
                + self.curia_weight * curia_loss
            )
            baseline_mse = self._masked_mse(data, target_post, acquired_image_mask)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {
            "loss": loss.detach().cpu().numpy(),
            "mse": mse_loss.detach().cpu().numpy(),
            "baseline_mse": baseline_mse.detach().cpu().numpy(),
            "lpips": lpips_loss.detach().cpu().numpy(),
            "curia": curia_loss.detach().cpu().numpy(),
        }

    def validation_step(self, batch: dict) -> dict:
        """Score one validation batch without updating the weights."""
        from nnunetv2.utilities.helpers import dummy_context

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        acquired_image_mask = batch["acquired_image_mask"].to(
            self.device, non_blocking=True
        )
        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(data)
            prediction_post = data + output
            target_post = data + target
            mse_loss = self._masked_mse(
                prediction_post, target_post, acquired_image_mask
            )
            lpips_loss = self._lpips_loss(
                prediction_post, target_post, acquired_image_mask
            )
            curia_loss = self._curia_token_mse_loss(
                prediction_post, target_post, acquired_image_mask
            )
            loss = (
                mse_loss
                + self.lpips_weight * lpips_loss
                + self.curia_weight * curia_loss
            )
            baseline_mse = self._masked_mse(data, target_post, acquired_image_mask)
        return {
            "loss": loss.detach().cpu().numpy(),
            "mse": mse_loss.detach().cpu().numpy(),
            "baseline_mse": baseline_mse.detach().cpu().numpy(),
            "lpips": lpips_loss.detach().cpu().numpy(),
            "curia": curia_loss.detach().cpu().numpy(),
        }

    def _mean_epoch_output(self, outputs: dict, key: str) -> float:
        """Mean of one output key over the epoch, gathered across ranks when distributed."""
        if not self.is_ddp:
            return float(np.mean(outputs[key]))
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, outputs[key])
        return float(np.vstack(gathered).mean())

    def on_train_epoch_end(self, train_outputs):
        """Log the token error on top of the inherited training metrics."""
        super().on_train_epoch_end(train_outputs)
        outputs = collate_outputs(train_outputs)
        train_curia = self._mean_epoch_output(outputs, "curia")
        self._log_translation_metric(
            "train_curia_losses", "train_curia2_token_mses", train_curia
        )

    def on_validation_epoch_end(self, val_outputs):
        """Log the token error on top of the inherited validation metrics."""
        super().on_validation_epoch_end(val_outputs)
        outputs = collate_outputs(val_outputs)
        val_curia = self._mean_epoch_output(outputs, "curia")
        self._log_translation_metric(
            "val_curia_losses", "val_curia2_token_mses", val_curia
        )

    def on_epoch_end(self):
        """Print the token error, then run the inherited end-of-epoch handling."""
        self._ensure_translation_logger_keys()
        train_curia = float(self.logger.get_value("train_curia_losses", step=-1))
        val_curia = float(self.logger.get_value("val_curia_losses", step=-1))
        self.print_to_log_file(
            "train_curia2_token_mse",
            np.round(train_curia, decimals=6),
            "val_curia2_token_mse",
            np.round(val_curia, decimals=6),
            "curia_weight",
            self.curia_weight,
        )
        super().on_epoch_end()

    def _plot_mse_progress_png(self):
        """Add a plot of the encoder token error."""
        super()._plot_mse_progress_png()
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from batchgenerators.utilities.file_and_folder_operations import join

        epochs = list(
            range(len(self.logger.get_value("train_curia_losses", step=None)))
        )
        if not epochs:
            return
        fig, ax = plt.subplots(1, 1, figsize=(14, 5))
        ax.plot(
            epochs,
            self.logger.get_value("train_curia_losses", step=None),
            label="train Curia-2 token MSE",
            linewidth=2,
        )
        ax.plot(
            epochs,
            self.logger.get_value("val_curia_losses", step=None),
            label="validation Curia-2 token MSE",
            linewidth=2,
        )
        ax.set_xlabel("epoch")
        ax.set_ylabel("whole acquired-image token MSE")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(join(self.output_folder, "progress_curia2_token_mse.png"), dpi=120)
        plt.close(fig)


class nnUNetTrainerMamaSynthTranslationLPIPSCuriaTokenMSE_LPIPS05_Curia05_BS_48_MAEFinetune(
    MamaSynthMAEFinetuneMixin,
    nnUNetTrainerMamaSynthTranslationLPIPSCuriaTokenMSE_LPIPS05_Curia05_BS_48,
):
    """Fine-tune of the whole-image-only trainer from MAE weights."""


class nnUNetTrainerMamaSynthTranslationLPIPSCuriaTokenMSE_LPIPS05_Curia05_BS_48_MAEFinetune_NoRot(
    nnUNetTrainerMamaSynthTranslationLPIPSCuriaTokenMSE_LPIPS05_Curia05_BS_48_MAEFinetune
):
    """Same fine-tune with rotation dropped and the encoder fed a cropped, resized frame.

    Rather than pushing the whole canvas, most of which is padding, through the
    encoder and then discarding the padding tokens, each slice is cropped to its
    acquired bounding box and stretched to fill the encoder's input. Every token then
    carries real anatomy. The encoder was itself trained with aspect-changing resizes,
    so the stretch is nothing unusual for it.
    """

    @staticmethod
    def get_translation_training_transforms(patch_size, rotation_for_DA, mirror_axes):
        """Spatial augmentation with mirroring only; rotation probability is zero."""
        from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
        from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
        from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms

        transforms = [
            SpatialTransform(
                patch_size,
                patch_center_dist_from_border=0,
                random_crop=False,
                p_elastic_deform=0,
                p_rotation=0.0,
                rotation=rotation_for_DA,
                p_scaling=0.0,
                scaling=(1.0, 1.0),
                p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False,
                mode_seg="nearest",
                border_mode_seg="zeros",
            )
        ]
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(MirrorTransform(allowed_axes=mirror_axes))
        return ComposeTransforms(transforms)

    def _curia_token_mse_loss(
        self,
        prediction_post: torch.Tensor,
        target_post: torch.Tensor,
        acquired_image_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Token error computed on the acquired bounding box, resized to the full frame.

        Each slice is cropped to its acquired region and resized independently, since the
        bounding box differs per slice. Normalisation and the token validity mask are then
        computed on the resized frame, where almost every token is valid.
        """
        from contextlib import nullcontext

        import torch.nn.functional as F

        model = self._get_curia_feature_model()
        patch_size = int(model.config.patch_size)
        target_h, target_w = prediction_post.shape[-2], prediction_post.shape[-1]
        batch_size = prediction_post.shape[0]

        prediction_frames = []
        target_frames = []
        mask_frames = []
        for index in range(batch_size):
            # Each slice has its own acquired bounding box, so the crop and resize have to be
            # done one sample at a time.
            present = acquired_image_mask[index].any(dim=0)
            if bool(present.any()):
                rows = torch.nonzero(present.any(dim=1), as_tuple=False)
                cols = torch.nonzero(present.any(dim=0), as_tuple=False)
                row_start, row_end = int(rows[0]), int(rows[-1]) + 1
                col_start, col_end = int(cols[0]), int(cols[-1]) + 1
            else:
                row_start, row_end, col_start, col_end = 0, target_h, 0, target_w

            sample = slice(index, index + 1)
            box = (
                sample,
                slice(None),
                slice(row_start, row_end),
                slice(col_start, col_end),
            )
            prediction_frames.append(
                F.interpolate(
                    prediction_post[box],
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
            )
            target_frames.append(
                F.interpolate(
                    target_post[box],
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
            )
            mask_frames.append(
                F.interpolate(
                    acquired_image_mask[box].float(),
                    size=(target_h, target_w),
                    mode="nearest",
                )
            )

        prediction_resized = torch.cat(prediction_frames, dim=0)
        target_resized = torch.cat(target_frames, dim=0)
        mask_resized = torch.cat(mask_frames, dim=0)

        prediction = self._curia_zscore(prediction_resized, mask_resized)
        target = self._curia_zscore(target_resized, mask_resized)
        losses = []
        for start in range(0, batch_size, self.curia_microbatch_size):
            end = min(start + self.curia_microbatch_size, batch_size)
            prediction_chunk = prediction[start:end]
            target_chunk = target[start:end]
            acquired_image_chunk = mask_resized[start:end].float()

            autocast_context = (
                torch.autocast(self.device.type, dtype=torch.bfloat16)
                if self.device.type == "cuda"
                else nullcontext()
            )
            with autocast_context:
                prediction_tokens = self._curia_patch_tokens(model, prediction_chunk)
                with torch.no_grad():
                    target_tokens = self._curia_patch_tokens(model, target_chunk)
            prediction_tokens = prediction_tokens.float()
            target_tokens = target_tokens.float()
            valid_fraction = F.avg_pool2d(
                acquired_image_chunk, kernel_size=patch_size, stride=patch_size
            ).flatten(1)
            acquired_image_tokens = valid_fraction > 0.999
            per_token_mse = (prediction_tokens - target_tokens).square().mean(dim=-1)
            losses.append(
                (end - start)
                * self._masked_token_mean(per_token_mse, acquired_image_tokens)
            )
        return torch.stack(losses).sum() / batch_size


class nnUNetTrainerMamaSynthTranslationCuriaTokenMSEOnly_BS_48_NoRot(
    nnUNetTrainerMamaSynthTranslationLPIPSCuriaTokenMSE_LPIPS05_Curia05_BS_48
):
    """Translation driven by the encoder token error alone.

    Neither the pixel error nor the perceptual distance enters the optimized loss;
    both are still computed and logged every step so the inherited progress plots stay
    intact, but no gradient comes from them. The encoder is fed the full canvas, as in
    the parent, and rotation is dropped from the augmentation.
    """

    curia_weight = 1.0

    @staticmethod
    def get_translation_training_transforms(patch_size, rotation_for_DA, mirror_axes):
        """Spatial augmentation with mirroring only; rotation probability is zero."""
        from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
        from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
        from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms

        transforms = [
            SpatialTransform(
                patch_size,
                patch_center_dist_from_border=0,
                random_crop=False,
                p_elastic_deform=0,
                p_rotation=0.0,
                rotation=rotation_for_DA,
                p_scaling=0.0,
                scaling=(1.0, 1.0),
                p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False,
                mode_seg="nearest",
                border_mode_seg="zeros",
            )
        ]
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(MirrorTransform(allowed_axes=mirror_axes))
        return ComposeTransforms(transforms)

    def train_step(self, batch: dict) -> dict:
        """Optimize the token error alone, computing the other terms only for logging."""
        from nnunetv2.utilities.helpers import dummy_context

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        acquired_image_mask = batch["acquired_image_mask"].to(
            self.device, non_blocking=True
        )
        self.optimizer.zero_grad(set_to_none=True)
        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(data)
            prediction_post = data + output
            target_post = data + target
            curia_loss = self._curia_token_mse_loss(
                prediction_post, target_post, acquired_image_mask
            )
            loss = self.curia_weight * curia_loss
            with torch.no_grad():
                mse_loss = self._masked_mse(
                    prediction_post, target_post, acquired_image_mask
                )
                lpips_loss = self._lpips_loss(
                    prediction_post, target_post, acquired_image_mask
                )
                baseline_mse = self._masked_mse(data, target_post, acquired_image_mask)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {
            "loss": loss.detach().cpu().numpy(),
            "mse": mse_loss.detach().cpu().numpy(),
            "baseline_mse": baseline_mse.detach().cpu().numpy(),
            "lpips": lpips_loss.detach().cpu().numpy(),
            "curia": curia_loss.detach().cpu().numpy(),
        }

    def validation_step(self, batch: dict) -> dict:
        """Score one validation batch, with the monitored terms computed but not optimized."""
        from nnunetv2.utilities.helpers import dummy_context

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        acquired_image_mask = batch["acquired_image_mask"].to(
            self.device, non_blocking=True
        )
        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(data)
            prediction_post = data + output
            target_post = data + target
            curia_loss = self._curia_token_mse_loss(
                prediction_post, target_post, acquired_image_mask
            )
            loss = self.curia_weight * curia_loss
            mse_loss = self._masked_mse(
                prediction_post, target_post, acquired_image_mask
            )
            lpips_loss = self._lpips_loss(
                prediction_post, target_post, acquired_image_mask
            )
            baseline_mse = self._masked_mse(data, target_post, acquired_image_mask)
        return {
            "loss": loss.detach().cpu().numpy(),
            "mse": mse_loss.detach().cpu().numpy(),
            "baseline_mse": baseline_mse.detach().cpu().numpy(),
            "lpips": lpips_loss.detach().cpu().numpy(),
            "curia": curia_loss.detach().cpu().numpy(),
        }


class _AbstractnnUNetTrainerMamaSynthCuriaMultiLayerMSEOnly_BS_48_NoRot(
    nnUNetTrainerMamaSynthTranslationCuriaTokenMSEOnly_BS_48_NoRot
):
    """Matches encoder features at several depths rather than at the final layer alone.

    Information useful for comparing two images is spread through a transformer's
    hierarchy, so the term becomes a weighted sum of per-layer token errors. The taps
    are evenly spaced through the network and always include the final block.
    Subclasses supply the weights. Everything else is inherited: only the encoder term
    is optimized, rotation is off, and the full canvas is fed to the encoder.
    """

    num_curia_taps = 4
    curia_tap_layers = None

    def _curia_tap_layer_indices(self, model) -> tuple:
        """Which encoder blocks to tap: an explicit list, or evenly spaced ones."""
        if self.curia_tap_layers is not None:
            return tuple(int(layer) for layer in self.curia_tap_layers)
        num_layers = int(model.config.num_hidden_layers)
        return tuple(
            round(num_layers * k / self.num_curia_taps)
            for k in range(1, self.num_curia_taps + 1)
        )

    def _curia_layer_weights(self, layers: tuple) -> list:
        """Per-layer weights aligned with ``layers``, deepest last. Subclasses implement this."""
        raise NotImplementedError

    @staticmethod
    def _curia_patch_tokens_at_layers(
        model, image: torch.Tensor, layers: tuple
    ) -> list:
        """Patch tokens from several blocks in one forward pass.

        Index zero of the hidden states is the embedding output, so index i is the output
        of block i.
        """
        outputs = model(pixel_values=image, output_hidden_states=True)
        hidden = outputs.hidden_states
        patch_size = int(model.config.patch_size)
        num_patches = (image.shape[-2] // patch_size) * (image.shape[-1] // patch_size)
        return [hidden[layer][:, -num_patches:] for layer in layers]

    def _curia_token_mse_loss(
        self,
        prediction_post: torch.Tensor,
        target_post: torch.Tensor,
        acquired_image_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted sum of per-layer token errors over valid tokens."""
        from contextlib import nullcontext

        import torch.nn.functional as F

        model = self._get_curia_feature_model()
        patch_size = int(model.config.patch_size)
        layers = self._curia_tap_layer_indices(model)
        weights = self._curia_layer_weights(layers)
        prediction = self._curia_zscore(prediction_post, acquired_image_mask)
        target = self._curia_zscore(target_post, acquired_image_mask)
        losses = []
        batch_size = prediction.shape[0]
        for start in range(0, batch_size, self.curia_microbatch_size):
            end = min(start + self.curia_microbatch_size, batch_size)
            prediction_chunk = prediction[start:end]
            target_chunk = target[start:end]
            acquired_image_chunk = acquired_image_mask[start:end].float()

            autocast_context = (
                torch.autocast(self.device.type, dtype=torch.bfloat16)
                if self.device.type == "cuda"
                else nullcontext()
            )
            with autocast_context:
                prediction_layers = self._curia_patch_tokens_at_layers(
                    model, prediction_chunk, layers
                )
                with torch.no_grad():
                    target_layers = self._curia_patch_tokens_at_layers(
                        model, target_chunk, layers
                    )
            valid_fraction = F.avg_pool2d(
                acquired_image_chunk, kernel_size=patch_size, stride=patch_size
            ).flatten(1)
            acquired_image_tokens = valid_fraction > 0.999
            chunk_loss = prediction_layers[0].sum() * 0.0
            for weight, prediction_tokens, target_tokens in zip(
                weights, prediction_layers, target_layers
            ):
                per_token_mse = (
                    (prediction_tokens.float() - target_tokens.float())
                    .square()
                    .mean(dim=-1)
                )
                chunk_loss = chunk_loss + weight * self._masked_token_mean(
                    per_token_mse, acquired_image_tokens
                )
            losses.append((end - start) * chunk_loss)
        return torch.stack(losses).sum() / batch_size


class nnUNetTrainerMamaSynthTranslationCuriaDepthWeightedMSEOnly_BS_48_NoRot(
    _AbstractnnUNetTrainerMamaSynthCuriaMultiLayerMSEOnly_BS_48_NoRot
):
    """Multi-layer variant weighting each tap in proportion to its depth.

    Deeper taps count for more, with the weights normalised to sum to one.
    """

    def _curia_layer_weights(self, layers: tuple) -> list:
        """Weights proportional to depth index, normalised to sum to one."""
        total = float(sum(layers))
        return [layer / total for layer in layers]


class nnUNetTrainerMamaSynthTranslationCuriaHalfFinalHalfInterMSEOnly_BS_48_NoRot(
    _AbstractnnUNetTrainerMamaSynthCuriaMultiLayerMSEOnly_BS_48_NoRot
):
    """Multi-layer variant splitting the term evenly between final and intermediate taps.

    Half the weight goes to the final block, matching the single-layer trainer, and
    the other half is shared equally among the intermediate taps.
    """

    def _curia_layer_weights(self, layers: tuple) -> list:
        """Half the weight on the final tap, the rest split evenly over the others."""
        num_intermediate = len(layers) - 1
        if num_intermediate <= 0:
            return [1.0]
        intermediate_weight = 0.5 / num_intermediate
        return [intermediate_weight] * num_intermediate + [0.5]


class nnUNetTrainerMamaSynthTranslationCuriaTokenMSEOnly_BS_48_NoRot_Jitter(
    nnUNetTrainerMamaSynthTranslationCuriaTokenMSEOnly_BS_48_NoRot
):
    """Final-layer token trainer with the patch grid randomly shifted during training.

    The encoder tokenizes on a fixed lattice, so an unshifted token error hands the
    network one target per patch at the same pixel offset every step, and it learns
    block-uniform corrections locked to that grid, which show up as rectangular
    artifacts. Shifting prediction, target and mask together by a random offset
    smaller than one patch, before normalisation, moves the patch boundaries around
    over the course of training and decorrelates the loss grid from absolute position.

    The shift pads and crops rather than wrapping, so anatomy is translated instead of
    being torn across the border, and the new edge is padding the token validity mask
    already discards. Jitter applies to training steps only, so the logged validation
    figure stays comparable across epochs. Inference is unaffected.
    """

    curia_grid_jitter = True
    curia_jitter_min_acquired_fraction = 0.5

    def _curia_grid_jitter(
        self,
        prediction_post: torch.Tensor,
        target_post: torch.Tensor,
        acquired_image_mask: torch.Tensor,
    ):
        """Shift prediction, target and mask together by a random sub-patch offset.

        A shift that would push too much of the acquired region off the canvas is skipped
        for that sample, which keeps small or edge-adjacent anatomy from being cropped
        away.
        """
        import torch.nn.functional as F

        patch_size = int(self._get_curia_feature_model().config.patch_size)
        dy = int(torch.randint(0, patch_size, (1,)).item())
        dx = int(torch.randint(0, patch_size, (1,)).item())
        if dy == 0 and dx == 0:
            return prediction_post, target_post, acquired_image_mask
        height, width = prediction_post.shape[-2], prediction_post.shape[-1]

        def shift(tensor: torch.Tensor) -> torch.Tensor:
            """Pad on the trailing edges, then crop back to the original size."""
            padded = F.pad(tensor, (0, dx, 0, dy))
            return padded[..., dy : dy + height, dx : dx + width]

        shifted_prediction = shift(prediction_post)
        shifted_target = shift(target_post)
        shifted_mask = shift(acquired_image_mask)
        original_count = acquired_image_mask.flatten(1).sum(dim=1)
        shifted_count = shifted_mask.flatten(1).sum(dim=1)
        # Skip the shift for samples where it would push too much anatomy off the canvas.
        keep_shifted = (
            shifted_count >= self.curia_jitter_min_acquired_fraction * original_count
        )
        keep_shifted = keep_shifted.view(-1, *([1] * (acquired_image_mask.dim() - 1)))
        return (
            torch.where(keep_shifted, shifted_prediction, prediction_post),
            torch.where(keep_shifted, shifted_target, target_post),
            torch.where(keep_shifted, shifted_mask, acquired_image_mask),
        )

    def _curia_token_mse_loss(
        self,
        prediction_post: torch.Tensor,
        target_post: torch.Tensor,
        acquired_image_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the grid jitter during training, then compute the inherited token error."""
        # Training only: validation uses the unshifted loss so the logged value stays
        # comparable from epoch to epoch.
        if self.curia_grid_jitter and self.network.training:
            prediction_post, target_post, acquired_image_mask = self._curia_grid_jitter(
                prediction_post, target_post, acquired_image_mask
            )
        # Run the token error in full precision; the per-slice normalisation that precedes
        # it is numerically fragile in half precision.
        with torch.autocast(self.device.type, enabled=False):
            return super()._curia_token_mse_loss(
                prediction_post.float(),
                target_post.float(),
                acquired_image_mask.float(),
            )


class nnUNetTrainerMamaSynthTranslationCuriaDiceTverskyA02B08Only_BS_48_NoRot_Jitter(
    nnUNetTrainerMamaSynthTranslationCuriaTokenMSEOnly_BS_48_NoRot_Jitter,
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceTverskyA02B08_LPIPS05_BS_48,
):
    """Optimizes the jittered token error together with the frozen segmenter's terms.

    The pixel error, perceptual distance and lesion structural term are computed and
    logged but contribute no gradient. Both parent branches take part: one supplies
    the encoder term, the jitter and the rotation-free loader, the other the frozen
    segmenter and the multi-term logging. Requires the frozen segmentation checkpoint.
    """

    curia_weight = 1.0
    dice_weight = 1.0
    tversky_weight = 1.0

    def train_step(self, batch: dict) -> dict:
        """Optimize the encoder and segmentation terms; compute the rest for logging only."""
        from nnunetv2.utilities.helpers import dummy_context

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        acquired_image_mask = batch["acquired_image_mask"].to(
            self.device, non_blocking=True
        )
        lesion_mask = batch["lesion_mask"].to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(data)
            prediction_post = data + output
            target_post = data + target
            curia_loss = self._curia_token_mse_loss(
                prediction_post, target_post, acquired_image_mask
            )
            dice_loss, tversky_loss, _ = self._segmentation_losses(
                prediction_post, lesion_mask, acquired_image_mask
            )
            loss = (
                self.curia_weight * curia_loss
                + self.dice_weight * dice_loss
                + self.tversky_weight * tversky_loss
            )
            with torch.no_grad():
                mse_loss = self._masked_mse(
                    prediction_post, target_post, acquired_image_mask
                )
                lpips_loss = self._lpips_loss(
                    prediction_post, target_post, acquired_image_mask
                )
                ssim_loss = self._ssim_loss(prediction_post, target_post, lesion_mask)
                baseline_mse = self._masked_mse(data, target_post, acquired_image_mask)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        zero = mse_loss.new_zeros(())
        return {
            "loss": loss.detach().cpu().numpy(),
            "mse": mse_loss.detach().cpu().numpy(),
            "baseline_mse": baseline_mse.detach().cpu().numpy(),
            "lpips": lpips_loss.detach().cpu().numpy(),
            "ssim": ssim_loss.detach().cpu().numpy(),
            "dice": dice_loss.detach().cpu().numpy(),
            "tversky": tversky_loss.detach().cpu().numpy(),
            "contour": zero.cpu().numpy(),
            "contour_recall": zero.cpu().numpy(),
            "curia": curia_loss.detach().cpu().numpy(),
        }

    def validation_step(self, batch: dict) -> dict:
        """Score one validation batch, with the monitored terms computed but not optimized."""
        from nnunetv2.utilities.helpers import dummy_context

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        acquired_image_mask = batch["acquired_image_mask"].to(
            self.device, non_blocking=True
        )
        lesion_mask = batch["lesion_mask"].to(self.device, non_blocking=True)
        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(data)
            prediction_post = data + output
            target_post = data + target
            curia_loss = self._curia_token_mse_loss(
                prediction_post, target_post, acquired_image_mask
            )
            dice_loss, tversky_loss, _ = self._segmentation_losses(
                prediction_post, lesion_mask, acquired_image_mask
            )
            loss = (
                self.curia_weight * curia_loss
                + self.dice_weight * dice_loss
                + self.tversky_weight * tversky_loss
            )
            mse_loss = self._masked_mse(
                prediction_post, target_post, acquired_image_mask
            )
            lpips_loss = self._lpips_loss(
                prediction_post, target_post, acquired_image_mask
            )
            ssim_loss = self._ssim_loss(prediction_post, target_post, lesion_mask)
            baseline_mse = self._masked_mse(data, target_post, acquired_image_mask)
        zero = mse_loss.new_zeros(())
        return {
            "loss": loss.detach().cpu().numpy(),
            "mse": mse_loss.detach().cpu().numpy(),
            "baseline_mse": baseline_mse.detach().cpu().numpy(),
            "lpips": lpips_loss.detach().cpu().numpy(),
            "ssim": ssim_loss.detach().cpu().numpy(),
            "dice": dice_loss.detach().cpu().numpy(),
            "tversky": tversky_loss.detach().cpu().numpy(),
            "contour": zero.cpu().numpy(),
            "contour_recall": zero.cpu().numpy(),
            "curia": curia_loss.detach().cpu().numpy(),
        }


class nnUNetTrainerMamaSynthTranslationCuriaTokenMSEOnly_BS_48_NoRot_Jitter_MomentMatch(
    nnUNetTrainerMamaSynthTranslationCuriaTokenMSEOnly_BS_48_NoRot_Jitter
):
    """Adds moment matching to pin the brightness and contrast the encoder ignores.

    Each slice is z-scored before it reaches the encoder, so the token error is blind
    to any affine rescaling of the prediction's intensities. The encoder therefore
    constrains structure and texture while the overall brightness and contrast are
    free to drift, which ruins every intensity-based comparison. A pixel error would
    pin the level but also smooth the image, because it pulls each pixel toward the
    mean of its plausible values.

    Moment matching instead constrains exactly the two numbers the encoder discards.
    Per slice, over a region, it matches the prediction's mean and standard deviation
    to the target's. Its gradient is a uniform brightness shift plus a global contrast
    rescale, an affine correction with no per-pixel averaging, so it restores the
    intensity scale without blurring what the encoder built.

    The statistics over the whole acquired region are dominated by ordinary tissue and
    say little about the lesion, which is small and bright, so a second term matches
    moments inside the lesion, gated on lesion presence.
    """

    moment_match_weight = 1.0
    lesion_moment_weight = 1.0
    moment_std_weight = 1.0
    moment_eps = 1e-8
    lesion_min_pixels = 1

    @staticmethod
    def _masked_mean_std(image: torch.Tensor, region_mask: torch.Tensor, eps: float):
        """Per-image mean and standard deviation of an image inside a region mask."""
        axes = (1, 2, 3)
        count = region_mask.sum(dim=axes).clamp_min(1.0)
        mean = (image * region_mask).sum(dim=axes) / count
        mean_broadcast = mean.view(-1, 1, 1, 1)
        variance = (((image - mean_broadcast).square()) * region_mask).sum(
            dim=axes
        ) / count
        std = variance.clamp_min(eps).sqrt()
        return mean, std

    def _region_moment_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        region_mask: torch.Tensor,
        min_pixels: int,
    ) -> torch.Tensor:
        """Squared difference of mean and of spread between prediction and target in a region.

        Images whose region is smaller than ``min_pixels`` are excluded, so a region that
        is absent or nearly so cannot contribute a meaningless statistic.
        """
        mean_prediction, std_prediction = self._masked_mean_std(
            prediction, region_mask, self.moment_eps
        )
        mean_target, std_target = self._masked_mean_std(
            target, region_mask, self.moment_eps
        )
        per_sample = (mean_prediction - mean_target).square() + (
            self.moment_std_weight * (std_prediction - std_target).square()
        )
        present = region_mask.sum(dim=(1, 2, 3)) >= float(min_pixels)
        if bool(present.any()):
            return per_sample[present].mean()
        return prediction.sum() * 0.0

    def _moment_match_loss(
        self,
        prediction_post: torch.Tensor,
        target_post: torch.Tensor,
        acquired_image_mask: torch.Tensor,
        lesion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Moment match over the whole acquired region, plus the lesion when weighted."""
        prediction = prediction_post.float()
        target = target_post.float()
        bulk_mask = acquired_image_mask.float()
        bulk_moment = self._region_moment_loss(prediction, target, bulk_mask, 1)
        if self.lesion_moment_weight != 0:
            lesion_region = (lesion_mask.float() > 0.5).to(prediction.dtype) * bulk_mask
            lesion_moment = self._region_moment_loss(
                prediction, target, lesion_region, self.lesion_min_pixels
            )
        else:
            lesion_moment = bulk_moment.new_zeros(())
        return bulk_moment + self.lesion_moment_weight * lesion_moment

    def _ensure_translation_logger_keys(self):
        """Add the moment-matching series to the logger."""
        super()._ensure_translation_logger_keys()
        local_logger = getattr(self.logger, "local_logger", None)
        if local_logger is None:
            return
        logging_dict = getattr(local_logger, "my_fantastic_logging", None)
        if logging_dict is None:
            return
        logging_dict.setdefault("train_moment_losses", [])
        logging_dict.setdefault("val_moment_losses", [])
        if "train_moment_match_losses" not in logging_dict:
            logging_dict["train_moment_match_losses"] = list(
                logging_dict["train_moment_losses"]
            )
        if "val_moment_match_losses" not in logging_dict:
            logging_dict["val_moment_match_losses"] = list(
                logging_dict["val_moment_losses"]
            )

    def _translation_forward(self, data: torch.Tensor) -> torch.Tensor:
        """Run the network on the input and return the residual.

        A hook for subclasses that add input channels before the forward pass; here it is
        a plain call on the single-channel image.
        """
        return self.network(data)

    def train_step(self, batch: dict) -> dict:
        """Optimize the encoder and moment terms and take one optimizer step."""
        from nnunetv2.utilities.helpers import dummy_context

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        acquired_image_mask = batch["acquired_image_mask"].to(
            self.device, non_blocking=True
        )
        lesion_mask = batch["lesion_mask"].to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self._translation_forward(data)
            prediction_post = data + output
            target_post = data + target
            curia_loss = self._curia_token_mse_loss(
                prediction_post, target_post, acquired_image_mask
            )
            moment_loss = self._moment_match_loss(
                prediction_post, target_post, acquired_image_mask, lesion_mask
            )
            loss = (
                self.curia_weight * curia_loss + self.moment_match_weight * moment_loss
            )
            with torch.no_grad():
                mse_loss = self._masked_mse(
                    prediction_post, target_post, acquired_image_mask
                )
                lpips_loss = self._lpips_loss(
                    prediction_post, target_post, acquired_image_mask
                )
                baseline_mse = self._masked_mse(data, target_post, acquired_image_mask)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {
            "loss": loss.detach().cpu().numpy(),
            "mse": mse_loss.detach().cpu().numpy(),
            "baseline_mse": baseline_mse.detach().cpu().numpy(),
            "lpips": lpips_loss.detach().cpu().numpy(),
            "curia": curia_loss.detach().cpu().numpy(),
            "moment": moment_loss.detach().cpu().numpy(),
        }

    def validation_step(self, batch: dict) -> dict:
        """Score one validation batch without updating the weights."""
        from nnunetv2.utilities.helpers import dummy_context

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        acquired_image_mask = batch["acquired_image_mask"].to(
            self.device, non_blocking=True
        )
        lesion_mask = batch["lesion_mask"].to(self.device, non_blocking=True)
        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self._translation_forward(data)
            prediction_post = data + output
            target_post = data + target
            curia_loss = self._curia_token_mse_loss(
                prediction_post, target_post, acquired_image_mask
            )
            moment_loss = self._moment_match_loss(
                prediction_post, target_post, acquired_image_mask, lesion_mask
            )
            loss = (
                self.curia_weight * curia_loss + self.moment_match_weight * moment_loss
            )
            mse_loss = self._masked_mse(
                prediction_post, target_post, acquired_image_mask
            )
            lpips_loss = self._lpips_loss(
                prediction_post, target_post, acquired_image_mask
            )
            baseline_mse = self._masked_mse(data, target_post, acquired_image_mask)
        return {
            "loss": loss.detach().cpu().numpy(),
            "mse": mse_loss.detach().cpu().numpy(),
            "baseline_mse": baseline_mse.detach().cpu().numpy(),
            "lpips": lpips_loss.detach().cpu().numpy(),
            "curia": curia_loss.detach().cpu().numpy(),
            "moment": moment_loss.detach().cpu().numpy(),
        }

    def on_train_epoch_end(self, train_outputs):
        """Log the moment term on top of the inherited training metrics."""
        super().on_train_epoch_end(train_outputs)
        outputs = collate_outputs(train_outputs)
        train_moment = self._mean_epoch_output(outputs, "moment")
        self._log_translation_metric(
            "train_moment_losses", "train_moment_match_losses", train_moment
        )

    def on_validation_epoch_end(self, val_outputs):
        """Log the moment term on top of the inherited validation metrics."""
        super().on_validation_epoch_end(val_outputs)
        outputs = collate_outputs(val_outputs)
        val_moment = self._mean_epoch_output(outputs, "moment")
        self._log_translation_metric(
            "val_moment_losses", "val_moment_match_losses", val_moment
        )

    def on_epoch_end(self):
        """Print the moment term, then run the inherited end-of-epoch handling."""
        self._ensure_translation_logger_keys()
        train_moment = float(self.logger.get_value("train_moment_losses", step=-1))
        val_moment = float(self.logger.get_value("val_moment_losses", step=-1))
        self.print_to_log_file(
            "train_moment_match",
            np.round(train_moment, decimals=6),
            "val_moment_match",
            np.round(val_moment, decimals=6),
            "moment_match_weight",
            self.moment_match_weight,
        )
        super().on_epoch_end()

    def _plot_mse_progress_png(self):
        """Add a plot of the moment-matching term."""
        super()._plot_mse_progress_png()
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from batchgenerators.utilities.file_and_folder_operations import join

        epochs = list(
            range(len(self.logger.get_value("train_moment_losses", step=None)))
        )
        if not epochs:
            return
        fig, ax = plt.subplots(1, 1, figsize=(14, 5))
        ax.plot(
            epochs,
            self.logger.get_value("train_moment_losses", step=None),
            label="train moment match",
            linewidth=2,
        )
        ax.plot(
            epochs,
            self.logger.get_value("val_moment_losses", step=None),
            label="validation moment match",
            linewidth=2,
        )
        ax.set_xlabel("epoch")
        ax.set_ylabel("per-slice mean/std match (bulk + lesion)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(join(self.output_folder, "progress_moment_match.png"), dpi=120)
        plt.close(fig)


class nnUNetTrainerMamaSynthTranslationCuriaTokenMSEOnly_BS_96_NoRot_Jitter_MomentMatch(
    nnUNetTrainerMamaSynthTranslationCuriaTokenMSEOnly_BS_48_NoRot_Jitter_MomentMatch
):
    """Region-stratified moment matching at batch size 96."""

    batch_size_override = 96


class nnUNetTrainerMamaSynthTranslationCuriaTokenMSEOnly_BS_48_NoRot_Jitter_MomentMatchPlain(
    nnUNetTrainerMamaSynthTranslationCuriaTokenMSEOnly_BS_48_NoRot_Jitter_MomentMatch
):
    """Moment matching over the acquired region only, with no separate lesion term.

    One global anchor per slice for the brightness and contrast the encoder discards.
    Use this when the drift is uniform across the image; the stratified parent is for
    when the lesion needs its own anchor.
    """

    lesion_moment_weight = 0.0


class nnUNetTrainerMamaSynthTranslationCuriaTokenMSEOnly_BS_32_NoRot(
    nnUNetTrainerMamaSynthTranslationCuriaTokenMSEOnly_BS_48_NoRot
):
    """Encoder-only trainer at batch size 32."""

    batch_size_override = 32


class nnUNetTrainerMamaSynthTranslationCuriaTokenMSEOnly_BS_32_NoRot_Jitter(
    nnUNetTrainerMamaSynthTranslationCuriaTokenMSEOnly_BS_48_NoRot_Jitter
):
    """Encoder-only trainer with grid jitter, at batch size 32."""

    batch_size_override = 32


class nnUNetTrainerMamaSynthTranslationCuriaDiceTverskyA02B08Only_BS_32_NoRot_Jitter(
    nnUNetTrainerMamaSynthTranslationCuriaDiceTverskyA02B08Only_BS_48_NoRot_Jitter
):
    """Encoder and segmentation terms with grid jitter, at batch size 32."""

    batch_size_override = 32


class nnUNetTrainerMamaSynthTranslationCuriaTokenMSEOnly_BS_32_NoRot_Jitter_MomentMatch(
    nnUNetTrainerMamaSynthTranslationCuriaTokenMSEOnly_BS_48_NoRot_Jitter_MomentMatch
):
    """Region-stratified moment matching at batch size 32."""

    batch_size_override = 32


class nnUNetTrainerMamaSynthTranslationCuriaTokenMSEOnly_BS_32_NoRot_Jitter_MomentMatchPlain(
    nnUNetTrainerMamaSynthTranslationCuriaTokenMSEOnly_BS_48_NoRot_Jitter_MomentMatchPlain
):
    """Global-only moment matching at batch size 32."""

    batch_size_override = 32


class nnUNetTrainerMamaSynthTranslationCuriaTokenMSEOnly_BS_32_NoRot_Jitter_MomentMatchPlain_SegChannel(
    nnUNetTrainerMamaSynthTranslationCuriaTokenMSEOnly_BS_32_NoRot_Jitter_MomentMatchPlain
):
    """Same objective as its parent, with a lesion probability map as a second input.

    Each step runs a frozen segmenter on the pre-contrast slice and concatenates its
    lesion probability to the image, so the network sees two channels and still
    predicts a one-channel residual. The segmenter never trains and the probability is
    detached, making it pure input guidance that leaves the loss untouched. It runs
    directly on the training slice, at the resolution and normalisation it was trained
    for, so no resizing is needed.

    The first convolution has one extra input channel, which the framework's weight
    loader fills in when a single-channel pretrained checkpoint is transferred.

    The segmenter directory is resolved at runtime, preferring a dedicated setting and
    falling back to the shared one.

    Inference has to repeat the same segment-and-concatenate step: an ordinary
    single-channel predictor cannot drive this checkpoint.
    """

    segmentation_use_mirroring = False
    pre_seg_prob_label = 1
    _pre_seg_model = None

    @staticmethod
    def _resolve_pre_seg_model_dir() -> Path:
        """Find the segmenter directory, preferring the dedicated setting over the shared one."""
        directory = os.environ.get("MAMASYNTH_PRE_SEG_MODEL_DIR") or os.environ.get(
            "MAMASYNTH_SEGMENTATION_MODEL_DIR"
        )
        if not directory:
            raise RuntimeError(
                "SegChannel trainer needs a pre-segmentation model. Pass --pre-seg-model-dir "
                "(or --seg-model-dir); neither MAMASYNTH_PRE_SEG_MODEL_DIR nor "
                "MAMASYNTH_SEGMENTATION_MODEL_DIR is set."
            )
        return Path(directory)

    @staticmethod
    def _resolve_pre_seg_checkpoint_name() -> str:
        """Which checkpoint file to load from that directory."""
        return os.environ.get("MAMASYNTH_PRE_SEG_CHECKPOINT") or os.environ.get(
            "MAMASYNTH_SEGMENTATION_CHECKPOINT", "checkpoint_final.pth"
        )

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=True,
    ):
        """Build the network with one extra input channel for the probability map."""
        return nnUNetTrainer.build_network_architecture(
            plans_manager,
            configuration_manager,
            num_input_channels + 1,
            1,
            enable_deep_supervision=False,
        )

    def _get_pre_seg_model(self):
        """Load the frozen pre-contrast segmenter once, from its own plans and checkpoint."""
        if self._pre_seg_model is not None:
            return self._pre_seg_model

        from nnunetv2.utilities.label_handling.label_handling import (
            determine_num_input_channels,
        )
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

        model_dir = self._resolve_pre_seg_model_dir()
        if (model_dir / "segmentation").is_dir() and not (
            model_dir / "plans.json"
        ).is_file():
            model_dir = model_dir / "segmentation"
        plans_path = model_dir / "plans.json"
        dataset_path = model_dir / "dataset.json"
        checkpoint_path = model_dir / "fold_0" / self._resolve_pre_seg_checkpoint_name()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Pre-segmentation checkpoint not found: {checkpoint_path}"
            )

        with plans_path.open("r", encoding="utf-8") as handle:
            seg_plans = json.load(handle)
        with dataset_path.open("r", encoding="utf-8") as handle:
            seg_dataset_json = json.load(handle)

        plans_manager = PlansManager(seg_plans)
        configuration_manager = plans_manager.get_configuration("2d")
        label_manager = plans_manager.get_label_manager(seg_dataset_json)
        num_input_channels = determine_num_input_channels(
            plans_manager, configuration_manager, seg_dataset_json
        )
        model = nnUNetTrainer.build_network_architecture(
            plans_manager,
            configuration_manager,
            num_input_channels,
            label_manager.num_segmentation_heads,
            enable_deep_supervision=False,
        )

        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["network_weights"]
        state_dict = {
            key[7:] if key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._pre_seg_model = model
        return model

    def _pre_seg_probability(self, data: torch.Tensor) -> torch.Tensor:
        """Lesion probability for the input slice, without gradients.

        Optionally averaged over mirrored inputs, which steadies the map at no cost to the
        generator's gradient since the whole thing runs detached.
        """
        model = self._get_pre_seg_model()
        with torch.no_grad():
            logits = model(data.float())
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            if self.segmentation_use_mirroring:
                mirrored = [logits]
                for dims in ((2,), (3,), (2, 3)):
                    flipped = model(torch.flip(data.float(), dims=dims))
                    if isinstance(flipped, (list, tuple)):
                        flipped = flipped[0]
                    mirrored.append(torch.flip(flipped, dims=dims))
                # Averaging over mirrored inputs steadies the probability map; it costs nothing in
                # gradient terms because the whole block runs detached.
                logits = torch.stack(mirrored, dim=0).mean(dim=0)
            probability = torch.softmax(logits.float(), dim=1)[
                :, self.pre_seg_prob_label : self.pre_seg_prob_label + 1
            ]
        return probability.to(data.dtype)

    def _translation_forward(self, data: torch.Tensor) -> torch.Tensor:
        """Concatenate the lesion probability onto the input and run the network."""
        seg_probability = self._pre_seg_probability(data)
        network_input = torch.cat([data, seg_probability], dim=1)
        return self.network(network_input)


class nnUNetTrainerMamaSynthTranslationCuriaDiceTverskyA02B08MomentMatchPlain_BS_32_NoRot_Jitter_SegChannel(
    nnUNetTrainerMamaSynthTranslationCuriaDiceTverskyA02B08Only_BS_48_NoRot_Jitter,
    nnUNetTrainerMamaSynthTranslationCuriaTokenMSEOnly_BS_32_NoRot_Jitter_MomentMatchPlain_SegChannel,
):
    """Everything enabled: encoder, segmentation and moment terms, plus the guidance channel.

    The optimized loss sums the jittered token error, the frozen segmenter's Dice and
    Tversky terms and the global moment match, and the network additionally receives
    the frozen segmenter's lesion probability as a second input channel. One parent
    branch supplies the segmentation terms and their logging, the other the moment
    term, the extra input channel and the runtime-resolved segmenter. The steps below
    sum all four terms and route the forward pass through the channel-appending hook;
    everything else is inherited.
    """

    batch_size_override = 32
    moment_match_weight = 1.0

    def train_step(self, batch: dict) -> dict:
        """Sum all four terms and take one optimizer step, with the guidance channel appended."""
        from nnunetv2.utilities.helpers import dummy_context

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        acquired_image_mask = batch["acquired_image_mask"].to(
            self.device, non_blocking=True
        )
        lesion_mask = batch["lesion_mask"].to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self._translation_forward(data)
            prediction_post = data + output
            target_post = data + target
            curia_loss = self._curia_token_mse_loss(
                prediction_post, target_post, acquired_image_mask
            )
            dice_loss, tversky_loss, _ = self._segmentation_losses(
                prediction_post, lesion_mask, acquired_image_mask
            )
            moment_loss = self._moment_match_loss(
                prediction_post, target_post, acquired_image_mask, lesion_mask
            )
            loss = (
                self.curia_weight * curia_loss
                + self.dice_weight * dice_loss
                + self.tversky_weight * tversky_loss
                + self.moment_match_weight * moment_loss
            )
            with torch.no_grad():
                mse_loss = self._masked_mse(
                    prediction_post, target_post, acquired_image_mask
                )
                lpips_loss = self._lpips_loss(
                    prediction_post, target_post, acquired_image_mask
                )
                ssim_loss = self._ssim_loss(prediction_post, target_post, lesion_mask)
                baseline_mse = self._masked_mse(data, target_post, acquired_image_mask)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        zero = mse_loss.new_zeros(())
        return {
            "loss": loss.detach().cpu().numpy(),
            "mse": mse_loss.detach().cpu().numpy(),
            "baseline_mse": baseline_mse.detach().cpu().numpy(),
            "lpips": lpips_loss.detach().cpu().numpy(),
            "ssim": ssim_loss.detach().cpu().numpy(),
            "dice": dice_loss.detach().cpu().numpy(),
            "tversky": tversky_loss.detach().cpu().numpy(),
            "contour": zero.cpu().numpy(),
            "contour_recall": zero.cpu().numpy(),
            "curia": curia_loss.detach().cpu().numpy(),
            "moment": moment_loss.detach().cpu().numpy(),
        }

    def validation_step(self, batch: dict) -> dict:
        """Score one validation batch on all four terms without updating the weights."""
        from nnunetv2.utilities.helpers import dummy_context

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        acquired_image_mask = batch["acquired_image_mask"].to(
            self.device, non_blocking=True
        )
        lesion_mask = batch["lesion_mask"].to(self.device, non_blocking=True)
        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self._translation_forward(data)
            prediction_post = data + output
            target_post = data + target
            curia_loss = self._curia_token_mse_loss(
                prediction_post, target_post, acquired_image_mask
            )
            dice_loss, tversky_loss, _ = self._segmentation_losses(
                prediction_post, lesion_mask, acquired_image_mask
            )
            moment_loss = self._moment_match_loss(
                prediction_post, target_post, acquired_image_mask, lesion_mask
            )
            loss = (
                self.curia_weight * curia_loss
                + self.dice_weight * dice_loss
                + self.tversky_weight * tversky_loss
                + self.moment_match_weight * moment_loss
            )
            mse_loss = self._masked_mse(
                prediction_post, target_post, acquired_image_mask
            )
            lpips_loss = self._lpips_loss(
                prediction_post, target_post, acquired_image_mask
            )
            ssim_loss = self._ssim_loss(prediction_post, target_post, lesion_mask)
            baseline_mse = self._masked_mse(data, target_post, acquired_image_mask)
        zero = mse_loss.new_zeros(())
        return {
            "loss": loss.detach().cpu().numpy(),
            "mse": mse_loss.detach().cpu().numpy(),
            "baseline_mse": baseline_mse.detach().cpu().numpy(),
            "lpips": lpips_loss.detach().cpu().numpy(),
            "ssim": ssim_loss.detach().cpu().numpy(),
            "dice": dice_loss.detach().cpu().numpy(),
            "tversky": tversky_loss.detach().cpu().numpy(),
            "contour": zero.cpu().numpy(),
            "contour_recall": zero.cpu().numpy(),
            "curia": curia_loss.detach().cpu().numpy(),
            "moment": moment_loss.detach().cpu().numpy(),
        }


class nnUNetTrainerMamaSynthTranslationCuriaDiceTverskyA02B08MomentMatchPlain_BS_48_NoRot_Jitter_SegChannel(
    nnUNetTrainerMamaSynthTranslationCuriaDiceTverskyA02B08MomentMatchPlain_BS_32_NoRot_Jitter_SegChannel
):
    """The everything-enabled trainer at batch size 48."""

    batch_size_override = 48


class nnUNetTrainerMamaSynthTranslationCuriaTokenMSEOnly_BS_32_NoRot_Jitter_MomentMatchPlain_39GB(
    nnUNetTrainerMamaSynthTranslationCuriaTokenMSEOnly_BS_32_NoRot_Jitter_MomentMatchPlain
):
    """Global-only moment matching at batch size 32, sized for a smaller memory budget."""

    batch_size_override = 32


class nnUNetTrainerMamaSynthTranslationCuriaTokenMSEOnly_BS_32_NoRot_Jitter_MomentMatchPlain_MAEFinetune(
    MamaSynthMAEFinetuneMixin,
    nnUNetTrainerMamaSynthTranslationCuriaTokenMSEOnly_BS_32_NoRot_Jitter_MomentMatchPlain_39GB,
):
    """Fine-tune of the global moment-matching encoder-only trainer from MAE weights."""


class nnUNetTrainerMamaSynthTranslationLPIPS_BS_32(
    nnUNetTrainerMamaSynthTranslationLPIPS
):
    """Pixel error plus perceptual distance at batch size 32.

    The loss, the training and validation steps and the logging are all inherited;
    only the batch size differs.
    """

    batch_size_override = 32


class nnUNetTrainerMamaSynthTranslationLPIPSMomentMatchPlain_BS_32(
    nnUNetTrainerMamaSynthTranslationLPIPS
):
    """Perceptual distance plus moment matching, with the pixel error dropped.

    The perceptual term pins structure but, after its fixed clipping and rescaling,
    says little about overall brightness and contrast. The moment term anchors exactly
    those two numbers, per slice over the acquired region, with a gradient that shifts
    and rescales rather than averaging pixels together. Unlike the encoder-based
    moment trainers this keeps the ordinary data pipeline, with no second model, no
    grid jitter and no lesion stratification.

    The pixel error and the identity baseline are still computed without gradients and
    logged, but they do not enter the objective.
    """

    batch_size_override = 32
    lpips_weight = 1.0
    moment_match_weight = 1.0
    moment_std_weight = 1.0
    moment_eps = 1e-8

    @staticmethod
    def _masked_mean_std(image: torch.Tensor, region_mask: torch.Tensor, eps: float):
        """Per-image mean and standard deviation of an image inside a region mask."""
        axes = (1, 2, 3)
        count = region_mask.sum(dim=axes).clamp_min(1.0)
        mean = (image * region_mask).sum(dim=axes) / count
        mean_broadcast = mean.view(-1, 1, 1, 1)
        variance = (((image - mean_broadcast).square()) * region_mask).sum(
            dim=axes
        ) / count
        std = variance.clamp_min(eps).sqrt()
        return mean, std

    def _moment_match_loss(
        self,
        prediction_post: torch.Tensor,
        target_post: torch.Tensor,
        acquired_image_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Match the prediction's mean and spread to the target's over the acquired region."""
        prediction = prediction_post.float()
        target = target_post.float()
        region_mask = acquired_image_mask.float()
        mean_prediction, std_prediction = self._masked_mean_std(
            prediction, region_mask, self.moment_eps
        )
        mean_target, std_target = self._masked_mean_std(
            target, region_mask, self.moment_eps
        )
        per_sample = (mean_prediction - mean_target).square() + (
            self.moment_std_weight * (std_prediction - std_target).square()
        )
        present = region_mask.sum(dim=(1, 2, 3)) >= 1.0
        if bool(present.any()):
            return per_sample[present].mean()
        return prediction.sum() * 0.0

    def _mean_epoch_output(self, outputs: dict, key: str) -> float:
        """Mean of one output key over the epoch, gathered across ranks when distributed."""
        if not self.is_ddp:
            return float(np.mean(outputs[key]))
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, outputs[key])
        return float(np.vstack(gathered).mean())

    def _ensure_translation_logger_keys(self):
        """Add the moment-matching series to the logger."""
        super()._ensure_translation_logger_keys()
        local_logger = getattr(self.logger, "local_logger", None)
        if local_logger is None:
            return
        logging_dict = getattr(local_logger, "my_fantastic_logging", None)
        if logging_dict is None:
            return
        logging_dict.setdefault("train_moment_losses", [])
        logging_dict.setdefault("val_moment_losses", [])
        if "train_moment_match_losses" not in logging_dict:
            logging_dict["train_moment_match_losses"] = list(
                logging_dict["train_moment_losses"]
            )
        if "val_moment_match_losses" not in logging_dict:
            logging_dict["val_moment_match_losses"] = list(
                logging_dict["val_moment_losses"]
            )

    def train_step(self, batch: dict) -> dict:
        """Optimize the perceptual and moment terms, computing the pixel error for logging."""
        from nnunetv2.utilities.helpers import dummy_context

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        acquired_image_mask = batch["acquired_image_mask"].to(
            self.device, non_blocking=True
        )
        self.optimizer.zero_grad(set_to_none=True)
        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(data)
            prediction_post = data + output
            target_post = data + target
            lpips_loss = self._lpips_loss(
                prediction_post, target_post, acquired_image_mask
            )
            moment_loss = self._moment_match_loss(
                prediction_post, target_post, acquired_image_mask
            )
            loss = (
                self.lpips_weight * lpips_loss + self.moment_match_weight * moment_loss
            )
            with torch.no_grad():
                mse_loss = self._masked_mse(
                    prediction_post, target_post, acquired_image_mask
                )
                baseline_mse = self._masked_mse(data, target_post, acquired_image_mask)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {
            "loss": loss.detach().cpu().numpy(),
            "mse": mse_loss.detach().cpu().numpy(),
            "baseline_mse": baseline_mse.detach().cpu().numpy(),
            "lpips": lpips_loss.detach().cpu().numpy(),
            "moment": moment_loss.detach().cpu().numpy(),
        }

    def validation_step(self, batch: dict) -> dict:
        """Score one validation batch, with the pixel error computed but not optimized."""
        from nnunetv2.utilities.helpers import dummy_context

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        acquired_image_mask = batch["acquired_image_mask"].to(
            self.device, non_blocking=True
        )
        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(data)
            prediction_post = data + output
            target_post = data + target
            lpips_loss = self._lpips_loss(
                prediction_post, target_post, acquired_image_mask
            )
            moment_loss = self._moment_match_loss(
                prediction_post, target_post, acquired_image_mask
            )
            loss = (
                self.lpips_weight * lpips_loss + self.moment_match_weight * moment_loss
            )
            mse_loss = self._masked_mse(
                prediction_post, target_post, acquired_image_mask
            )
            baseline_mse = self._masked_mse(data, target_post, acquired_image_mask)
        return {
            "loss": loss.detach().cpu().numpy(),
            "mse": mse_loss.detach().cpu().numpy(),
            "baseline_mse": baseline_mse.detach().cpu().numpy(),
            "lpips": lpips_loss.detach().cpu().numpy(),
            "moment": moment_loss.detach().cpu().numpy(),
        }

    def on_train_epoch_end(self, train_outputs):
        """Log the moment term on top of the inherited training metrics."""
        super().on_train_epoch_end(train_outputs)
        outputs = collate_outputs(train_outputs)
        train_moment = self._mean_epoch_output(outputs, "moment")
        self._log_translation_metric(
            "train_moment_losses", "train_moment_match_losses", train_moment
        )

    def on_validation_epoch_end(self, val_outputs):
        """Log the moment term on top of the inherited validation metrics."""
        super().on_validation_epoch_end(val_outputs)
        outputs = collate_outputs(val_outputs)
        val_moment = self._mean_epoch_output(outputs, "moment")
        self._log_translation_metric(
            "val_moment_losses", "val_moment_match_losses", val_moment
        )

    def on_epoch_end(self):
        """Print the moment term, then run the inherited end-of-epoch handling."""
        self._ensure_translation_logger_keys()
        train_moment = float(self.logger.get_value("train_moment_losses", step=-1))
        val_moment = float(self.logger.get_value("val_moment_losses", step=-1))
        self.print_to_log_file(
            "train_moment_match",
            np.round(train_moment, decimals=6),
            "val_moment_match",
            np.round(val_moment, decimals=6),
            "moment_match_weight",
            self.moment_match_weight,
        )
        super().on_epoch_end()

    def _plot_mse_progress_png(self):
        """Add a plot of the moment-matching term."""
        super()._plot_mse_progress_png()
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from batchgenerators.utilities.file_and_folder_operations import join

        epochs = list(
            range(len(self.logger.get_value("train_moment_losses", step=None)))
        )
        if not epochs:
            return
        fig, ax = plt.subplots(1, 1, figsize=(14, 5))
        ax.plot(
            epochs,
            self.logger.get_value("train_moment_losses", step=None),
            label="train moment match",
            linewidth=2,
        )
        ax.plot(
            epochs,
            self.logger.get_value("val_moment_losses", step=None),
            label="validation moment match",
            linewidth=2,
        )
        ax.set_xlabel("epoch")
        ax.set_ylabel("per-slice mean/std match (acquired region)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(join(self.output_folder, "progress_moment_match.png"), dpi=120)
        plt.close(fig)


class nnUNetTrainerMamaSynthTranslationLPIPS_BS_32_MAEFinetune(
    MamaSynthMAEFinetuneMixin,
    nnUNetTrainerMamaSynthTranslationLPIPS_BS_32,
):
    """Fine-tune of the pixel-plus-perceptual baseline from MAE weights."""


class nnUNetTrainerMamaSynthTranslationLPIPSMomentMatchPlain_BS_32_MAEFinetune(
    MamaSynthMAEFinetuneMixin,
    nnUNetTrainerMamaSynthTranslationLPIPSMomentMatchPlain_BS_32,
):
    """Fine-tune of the perceptual-plus-moment trainer from MAE weights."""


class nnUNetTrainerMamaSynthTranslationCuriaTokenMSEOnly_BS_32_NoRot_Jitter_MomentMatchPlain_SegChannel_MAEFinetune(
    MamaSynthMAEFinetuneMixin,
    nnUNetTrainerMamaSynthTranslationCuriaTokenMSEOnly_BS_32_NoRot_Jitter_MomentMatchPlain_SegChannel,
):
    """Fine-tune of the encoder-plus-moment trainer with the guidance channel.

    The single-channel pretrained weights are expanded into the two-channel first
    convolution by the framework's weight loader.
    """


class nnUNetTrainerMamaSynthTranslationCuriaDiceTverskyA02B08MomentMatchPlain_BS_32_NoRot_Jitter_SegChannel_MAEFinetune(
    MamaSynthMAEFinetuneMixin,
    nnUNetTrainerMamaSynthTranslationCuriaDiceTverskyA02B08MomentMatchPlain_BS_32_NoRot_Jitter_SegChannel,
):
    """Fine-tune of the everything-enabled trainer with the guidance channel.

    The single-channel pretrained weights are expanded into the two-channel first
    convolution by the framework's weight loader.
    """


class nnUNetTrainerMamaSynthTranslationLPIPS_BS_48_MAEFinetune(
    nnUNetTrainerMamaSynthTranslationLPIPS_BS_32_MAEFinetune
):
    """Pixel-plus-perceptual fine-tune at batch size 48."""

    batch_size_override = 48


class nnUNetTrainerMamaSynthTranslationLPIPS_BS_48_MAEFinetune_ep_200(
    nnUNetTrainerMamaSynthTranslationLPIPS_BS_48_MAEFinetune
):
    """Batch size 48 fine-tune run for 200 epochs."""

    def __init__(
        self, plans, configuration, fold, dataset_json, device=torch.device("cuda")
    ):
        """Extend the schedule to 200 epochs."""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200


class nnUNetTrainerMamaSynthTranslationLPIPS_BS_48_MAEFinetune_ep_300(
    nnUNetTrainerMamaSynthTranslationLPIPS_BS_48_MAEFinetune
):
    """Batch size 48 fine-tune run for 300 epochs."""

    def __init__(
        self, plans, configuration, fold, dataset_json, device=torch.device("cuda")
    ):
        """Extend the schedule to 300 epochs."""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 300


class nnUNetTrainerMamaSynthTranslationLPIPS_BS_64_MAEFinetune(
    nnUNetTrainerMamaSynthTranslationLPIPS_BS_32_MAEFinetune
):
    """Pixel-plus-perceptual fine-tune at batch size 64."""

    batch_size_override = 64


class nnUNetTrainerMamaSynthTranslationLPIPS_BS_64_MAEFinetune_ep_200(
    nnUNetTrainerMamaSynthTranslationLPIPS_BS_64_MAEFinetune
):
    """Batch size 64 fine-tune run for 200 epochs."""

    def __init__(
        self, plans, configuration, fold, dataset_json, device=torch.device("cuda")
    ):
        """Extend the schedule to 200 epochs."""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200


class nnUNetTrainerMamaSynthTranslationLPIPS_BS_64_MAEFinetune_ep_300(
    nnUNetTrainerMamaSynthTranslationLPIPS_BS_64_MAEFinetune
):
    """Batch size 64 fine-tune run for 300 epochs."""

    def __init__(
        self, plans, configuration, fold, dataset_json, device=torch.device("cuda")
    ):
        """Extend the schedule to 300 epochs."""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 300
