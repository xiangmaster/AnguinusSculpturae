"""Masked-autoencoder pretraining for the 2D ResEnc U-Net.

Trains the same architecture the translation trainers fine-tune, on unlabeled
slices, by blanking coarse blocks of the input and reconstructing them.
"""

from __future__ import annotations

import os
import zlib
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

from nnunetv2.training.mamasynth.dataset import TARGET_SIZE, load_split
from nnunetv2.training.mamasynth.mae_data_loader import (
    MamaSynthMAEFixedValDataLoader,
    MamaSynthMAESliceDataLoader,
)
from nnunetv2.training.mamasynth.mae_dataset import load_mae_cases
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerMamaSynth import (
    MAMASYNTH_DATASET_DIR_ENV,
)
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.helpers import dummy_context

try:
    from torch import autocast
except Exception:
    from torch.cuda.amp import autocast


class MamaSynthMAEDataMixin:
    """Feeds the trainer from blosc2 volumes instead of an nnU-Net preprocessed folder.

    The masked-image-modelling counterpart of ``MamaSynthDataMixin``: there are no
    segmentation masks, volumes are keyed by ``volume_id``, and slices are drawn
    uniformly across the whole dataset. Every override below exists so the base
    trainer never reaches for a preprocessed folder that does not exist.
    """

    batch_size_override = None

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
        self.cases = load_mae_cases(self.dataset_dir)

    def _do_i_compile(self):
        """Disable torch.compile for this trainer."""
        return False

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

    def do_split(self):
        """Return the train and validation volume ids for the current fold."""
        split = load_split(self.dataset_dir, self.fold)
        return split["train"], split["val"]


class nnUNetTrainerMamaSynthMAE(MamaSynthMAEDataMixin, nnUNetTrainer):
    """2D masked-autoencoder pretraining of the ResEnc U-Net.

    Coarse blocks are zeroed in the input image, the network reconstructs the full
    slice, and an L2 loss is taken only over pixels that were both masked and actually
    acquired. The architecture matches the translation fine-tune exactly so weights
    transfer one-to-one.
    """

    batch_size_override = None

    # Masking knobs, overridable by variants: the coarse-block edge in pixels (must
    # divide the canvas size) and the per-sample masked fraction, drawn uniformly from
    # this range.
    block_size = 32
    mask_ratio_range = (
        0.6,
        0.9,
    )

    def __init__(
        self, plans, configuration, fold, dataset_json, device=torch.device("cuda")
    ):
        """Set the schedule, the optimizer hyperparameters and the NaN-guard counters."""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 100
        self.num_epochs = 1000
        self.enable_deep_supervision = False
        # AdamW rather than the high-momentum SGD of the original recipe: reconstruction
        # MSE on globally z-scored, multi-modal inputs produces gradient spikes, and
        # momentum accumulates unclipped velocity from them until the run diverges.
        # Adaptive per-parameter scaling has no such failure mode.
        self.initial_lr = 1e-4
        self.weight_decay = 1e-4
        self._last_finite_train_loss = 0.0
        self._nan_skip_count = 0
        assert (
            TARGET_SIZE % self.block_size == 0
        ), "block_size must divide TARGET_SIZE (512)."

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=True,
    ):
        """Build the same U-Net as the translation trainers, with one output channel.

        Deep supervision is off, so the reconstruction comes from the full-resolution
        head alone.
        """
        return nnUNetTrainer.build_network_architecture(
            plans_manager,
            configuration_manager,
            num_input_channels,
            1,
            enable_deep_supervision=False,
        )

    def _build_loss(self):
        """Unused placeholder; the reconstruction loss is computed in the train/val steps."""
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

    @staticmethod
    def get_mae_training_transforms(patch_size, rotation_for_DA, mirror_axes):
        """Spatial-only augmentation: random rotation, scaling and mirroring.

        No intensity augmentation, since the reconstruction target is the input itself.
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
                p_scaling=0.2,
                scaling=(0.7, 1.4),
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

        Validation uses a fixed set of centred slices in a deterministic order, so the
        validation loss is comparable across epochs; its length sets the number of
        validation iterations per epoch.
        """
        patch_size = self.configuration_manager.patch_size
        rotation_for_DA, _, _, mirror_axes = (
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )
        tr_transforms = self.get_mae_training_transforms(
            patch_size, rotation_for_DA, mirror_axes
        )

        tr_keys, val_keys = self.do_split()
        tr_cases = [c for c in self.cases if c.volume_id in set(tr_keys)]
        val_cases = [c for c in self.cases if c.volume_id in set(val_keys)]

        dl_tr = MamaSynthMAESliceDataLoader(
            tr_cases, self.batch_size, transforms=tr_transforms
        )
        dl_val = MamaSynthMAEFixedValDataLoader(val_cases, self.batch_size)
        self.fixed_validation_loader = dl_val
        self.num_val_iterations_per_epoch = dl_val.num_batches
        msg = (
            f"MAE pretraining: {len(tr_cases)} train volumes "
            f"({dl_tr._total_slices} slices, uniform global sampling); "
            f"validation {len(dl_val.samples)} centered slices in {dl_val.num_batches} batches"
        )
        if dl_val.capped:
            msg += f" (capped from {dl_val.num_volumes} val volumes)"
        self.print_to_log_file(msg)

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
        # The fixed validation loader owns one ordered cursor, so priming it or spreading
        # it over workers would consume or duplicate batches.
        mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        if allowed_num_processes != 0:
            _ = next(mt_gen_train)
        return mt_gen_train, mt_gen_val

    def _make_visible_mask(self, batch_size: int, device, seeds=None) -> torch.Tensor:
        """Build a block mask upsampled to full resolution, where 1 is visible and 0 masked.

        A random subset of the coarse grid is zeroed and each block is expanded to
        ``block_size`` pixels. Passing ``seeds`` makes the mask reproducible per sample and
        pins the masked fraction to the midpoint of the configured range.
        """
        grid = TARGET_SIZE // self.block_size
        n_blocks = grid * grid
        masks = []
        for i in range(batch_size):
            if seeds is not None:
                generator = torch.Generator().manual_seed(int(seeds[i]))
                ratio = 0.5 * (self.mask_ratio_range[0] + self.mask_ratio_range[1])
                perm = torch.randperm(n_blocks, generator=generator)
            else:
                lo, hi = self.mask_ratio_range
                ratio = lo if hi <= lo else float(np.random.uniform(lo, hi))
                perm = torch.randperm(n_blocks)
            flat = torch.ones(n_blocks)
            flat[perm[: int(round(ratio * n_blocks))]] = 0.0
            masks.append(flat.reshape(grid, grid))
        mask = torch.stack(masks)[:, None]
        mask = mask.repeat_interleave(self.block_size, dim=2).repeat_interleave(
            self.block_size, dim=3
        )
        return mask.to(device, non_blocking=True)

    @staticmethod
    def _mae_loss(
        output: torch.Tensor,
        target: torch.Tensor,
        visible: torch.Tensor,
        acquired: torch.Tensor,
    ) -> torch.Tensor:
        """Mean squared reconstruction error over masked, acquired pixels.

        Pixels that stayed visible, and padding outside the acquired image, carry no
        weight.
        """
        weight = (1.0 - visible) * acquired.to(dtype=visible.dtype)
        squared_error = (output - target).square() * weight
        return squared_error.sum() / weight.sum().clamp_min(1.0)

    def train_step(self, batch: dict) -> dict:
        """Mask the input, reconstruct it, and take one guarded optimizer step."""
        data = batch["data"].to(self.device, non_blocking=True)
        acquired = batch["acquired_image_mask"].to(self.device, non_blocking=True)
        visible = self._make_visible_mask(data.shape[0], self.device)
        masked_data = data * visible

        self.optimizer.zero_grad(set_to_none=True)
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(masked_data)
            l = self._mae_loss(output, data, visible, acquired)

        # A non-finite loss must never reach the optimizer: it would poison the weights and
        # the optimizer state for the rest of the run. Skip the step and report the last
        # finite value so the loss curve stays readable.
        if not torch.isfinite(l):
            self.optimizer.zero_grad(set_to_none=True)
            self._nan_skip_count += 1
            return {"loss": np.array(self._last_finite_train_loss, dtype=np.float32)}

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            if torch.isfinite(grad_norm):
                self.grad_scaler.step(self.optimizer)
            else:
                self._nan_skip_count += 1
            self.grad_scaler.update()
        else:
            l.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            if torch.isfinite(grad_norm):
                self.optimizer.step()
            else:
                self._nan_skip_count += 1
        self._last_finite_train_loss = float(l.detach())
        return {"loss": np.array(self._last_finite_train_loss, dtype=np.float32)}

    def validation_step(self, batch: dict) -> dict:
        """Reconstruct one validation batch under a deterministic mask."""
        data = batch["data"].to(self.device, non_blocking=True)
        acquired = batch["acquired_image_mask"].to(self.device, non_blocking=True)
        # Derive the mask seed from the sample key so each validation slice keeps the same
        # mask across epochs and across resumes.
        seeds = [zlib.crc32(k.encode("utf-8")) for k in batch["keys"]]
        visible = self._make_visible_mask(data.shape[0], self.device, seeds=seeds)
        masked_data = data * visible

        with torch.no_grad():
            with (
                autocast(self.device.type, enabled=True)
                if self.device.type == "cuda"
                else dummy_context()
            ):
                output = self.network(masked_data)
                l = self._mae_loss(output, data, visible, acquired)
        return {"loss": l.detach().cpu().numpy()}

    def on_validation_epoch_end(self, val_outputs):
        """Average the validation loss across ranks and log it."""
        outputs = collate_outputs(val_outputs)
        if self.is_ddp:
            losses_val = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_val, outputs["loss"])
            loss_here = float(np.vstack(losses_val).mean())
        else:
            loss_here = float(np.mean(outputs["loss"]))

        self.logger.log("val_losses", loss_here, self.current_epoch)
        # The base class selects checkpoints and draws progress plots from a
        # higher-is-better quantity, so the negative reconstruction error is logged in the
        # slot it expects.
        negative_recon_mse = -loss_here
        self.logger.log("mean_fg_dice", negative_recon_mse, self.current_epoch)
        self.logger.log(
            "dice_per_class_or_region", [negative_recon_mse], self.current_epoch
        )

    def on_train_epoch_end(self, train_outputs):
        """Report skipped optimizer steps and refresh the reconstruction preview."""
        super().on_train_epoch_end(train_outputs)
        if self._nan_skip_count:
            self.print_to_log_file(
                f"WARNING: skipped {self._nan_skip_count} non-finite optimizer step(s) this epoch."
            )
            self._nan_skip_count = 0
        if self.local_rank == 0:
            try:
                self._save_mae_examples_png()
            # Preview rendering is best-effort and must not interrupt training.
            except Exception as exc:
                self.print_to_log_file(f"Could not save MAE example png: {exc}")

    def _save_mae_examples_png(self, num_examples: int = 4):
        """Write a masked input / target / reconstruction panel for a few training slices."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from batchgenerators.utilities.file_and_folder_operations import join

        batch = next(self.dataloader_train)
        data = batch["data"].to(self.device, non_blocking=True)
        acquired = batch["acquired_image_mask"].to(self.device, non_blocking=True)
        n = min(num_examples, data.shape[0])
        visible = self._make_visible_mask(data.shape[0], self.device)
        masked_data = data * visible

        was_training = self.network.training
        self.network.eval()
        with torch.no_grad():
            with (
                autocast(self.device.type, enabled=True)
                if self.device.type == "cuda"
                else dummy_context()
            ):
                output = self.network(masked_data)
        if was_training:
            self.network.train()

        data = data.detach().cpu().float().numpy()
        masked_np = masked_data.detach().cpu().float().numpy()
        recon = output.detach().cpu().float().numpy()
        keys = batch.get("keys", [""] * data.shape[0])

        fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
        titles = ["masked input", "target", "reconstruction"]
        for i in range(n):
            panels = [masked_np[i, 0], data[i, 0], recon[i, 0]]
            for col, (img, title) in enumerate(zip(panels, titles)):
                axes[i, col].imshow(img, cmap="gray")
                axes[i, col].set_title(str(keys[i]) if col == 0 else title, fontsize=7)
                axes[i, col].axis("off")
        fig.suptitle(f"epoch {self.current_epoch} - MAE reconstruction", fontsize=9)
        fig.tight_layout()
        fig.savefig(join(self.output_folder, "mae_examples.png"), dpi=100)
        plt.close(fig)


class nnUNetTrainerMamaSynthMAE_BS_32(nnUNetTrainerMamaSynthMAE):
    """MAE pretraining at batch size 32."""

    batch_size_override = 32


class nnUNetTrainerMamaSynthMAE_BS_64_ep_1000(nnUNetTrainerMamaSynthMAE):
    """MAE pretraining at batch size 64 for 1000 epochs."""

    batch_size_override = 64

    def __init__(
        self, plans, configuration, fold, dataset_json, device=torch.device("cuda")
    ):
        """Extend the schedule to 1000 epochs."""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000
