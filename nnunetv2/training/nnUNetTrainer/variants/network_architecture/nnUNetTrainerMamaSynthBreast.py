"""2D segmentation of the whole breast region from pre-contrast slices.

Provides a slice-level data loader that samples only slices containing breast
tissue, plus the nnU-Net trainers that use it. The target merges the raw lesion
and breast labels into a single foreground class.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np
import torch
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
    NonDetMultiThreadedAugmenter,
)
from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)
from threadpoolctl import threadpool_limits

from nnunetv2.training.mamasynth.dataset import (
    BACKGROUND_LABEL,
    BREAST_LABEL,
    LESION_LABEL,
    TARGET_SIZE,
    MamaSynthCase,
    _place_into_canvas,
    _random_placement,
    open_b2nd,
    resample_slice,
    zscore_per_slice,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerMamaSynth import (
    MamaSynthDataMixin,
)
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA


@lru_cache(maxsize=8192)
def breast_slice_indices(mask_path: str) -> tuple:
    """Return the indices of all slices whose mask contains breast tissue.

    Breast tissue is any voxel carrying the lesion or the breast label. Cached per
    mask path, since the same volume is queried repeatedly while building a split.
    """
    mask = np.asarray(open_b2nd(mask_path)[:])
    present = np.any((mask == LESION_LABEL) | (mask == BREAST_LABEL), axis=(1, 2))
    return tuple(int(index) for index in np.nonzero(present)[0])


def load_pre_and_breast_mask_slice(case: MamaSynthCase, slice_idx: int) -> tuple:
    """Load one pre-contrast slice and its binary breast mask, ready for training.

    The multi-label mask is collapsed to breast-vs-background, both arrays are
    resampled to the common pixel spacing, the image is z-scored, and the two are
    placed into the fixed-size canvas at the same random offset so they stay aligned.
    """
    pre = np.asarray(open_b2nd(case.pre_path)[slice_idx], dtype=np.float32)
    raw_mask = np.asarray(open_b2nd(case.mask_path)[slice_idx], dtype=np.uint8)
    breast_mask = ((raw_mask == LESION_LABEL) | (raw_mask == BREAST_LABEL)).astype(
        np.int16
    )

    pre = zscore_per_slice(
        resample_slice(pre, case.spacing_y, case.spacing_x, is_mask=False)
    ).astype(np.float32)
    breast_mask = resample_slice(
        breast_mask, case.spacing_y, case.spacing_x, is_mask=True
    ).astype(np.int16)

    placement_yx = (
        _random_placement(pre.shape[0], TARGET_SIZE),
        _random_placement(pre.shape[1], TARGET_SIZE),
    )
    pre = _place_into_canvas(pre, placement_yx, fill=0.0)
    breast_mask = _place_into_canvas(breast_mask, placement_yx, fill=BACKGROUND_LABEL)
    return np.ascontiguousarray(pre), np.ascontiguousarray(breast_mask)


class MamaSynthBreastSliceDataLoader(DataLoader):
    """Samples breast-positive pre-contrast slices in nnU-Net batch format."""

    def __init__(self, cases: List[MamaSynthCase], batch_size: int, transforms=None):
        """Build the flat list of (case, slice index) pairs that contain breast tissue.

        Raises ValueError if the split has no such slice.
        """
        super().__init__(cases, batch_size, 1, None, True, False, True, None)
        self.cases = cases
        self.transforms = transforms
        self.positive_slices = [
            (case, slice_idx)
            for case in cases
            for slice_idx in breast_slice_indices(str(case.mask_path))
        ]
        if not self.positive_slices:
            raise ValueError(
                "Split contains no slices with breast foreground (raw label 1 or 2)."
            )

    def _sample_foreground_example(self, max_tries: int = 32) -> tuple:
        """Draw one random breast-positive slice, retrying if the crop lost the mask.

        The random canvas placement can push the breast outside the visible area, so up
        to ``max_tries`` samples are drawn until one still has mask voxels.
        """
        for _ in range(max_tries):
            sample_index = int(np.random.randint(len(self.positive_slices)))
            case, slice_idx = self.positive_slices[sample_index]
            pre, mask = load_pre_and_breast_mask_slice(case, slice_idx)
            if mask.any():
                return pre, mask, f"{case.case_id}_s{slice_idx}"
        raise RuntimeError(
            f"Could not retain breast foreground after {max_tries} random crop attempts."
        )

    def generate_train_batch(self) -> dict:
        """Assemble one augmented batch of images and breast masks.

        Each example is transformed on its own and then stacked. Under deep supervision
        the transform returns one segmentation per resolution, so a separate stacked
        tensor is built for every level.
        """
        data_all = None
        seg_all = None
        keys: List[str] = []
        with torch.no_grad(), threadpool_limits(limits=1, user_api=None):
            for batch_index in range(self.batch_size):
                pre, mask, key = self._sample_foreground_example()
                keys.append(key)
                data = torch.from_numpy(pre)[None].float()
                seg = torch.from_numpy(mask)[None].to(torch.int16)

                if self.transforms is not None:
                    transformed = self.transforms(image=data, segmentation=seg)
                    data, seg = transformed["image"], transformed["segmentation"]

                if data_all is None:
                    data_all = torch.empty(
                        (self.batch_size, *data.shape), dtype=torch.float32
                    )
                data_all[batch_index] = data

                if isinstance(seg, list):
                    if seg_all is None:
                        seg_all = [
                            torch.empty(
                                (self.batch_size, *level.shape), dtype=level.dtype
                            )
                            for level in seg
                        ]
                    for level_index, level in enumerate(seg):
                        seg_all[level_index][batch_index] = level
                else:
                    if seg_all is None:
                        seg_all = torch.empty(
                            (self.batch_size, *seg.shape), dtype=seg.dtype
                        )
                    seg_all[batch_index] = seg

        return {"data": data_all, "target": seg_all, "keys": keys}


class nnUNetTrainerMamaSynthBreast(MamaSynthDataMixin, nnUNetTrainer):
    """Standard 2D nnU-Net trained to segment the whole breast region."""

    def __init__(
        self, plans, configuration, fold, dataset_json, device=torch.device("cuda")
    ):
        """Replace the label set with background/breast and fix the epoch schedule."""
        breast_dataset_json = dict(dataset_json)
        breast_dataset_json["labels"] = {"background": 0, "breast": 1}
        super().__init__(plans, configuration, fold, breast_dataset_json, device)
        self.num_iterations_per_epoch = 1000
        self.num_val_iterations_per_epoch = 200
        self.num_epochs = 100

    def get_dataloaders(self):
        """Build augmented train and validation loaders over breast-positive slices.

        Keeps the standard nnU-Net transform pipeline but swaps in the slice loader,
        which applies the transforms itself rather than leaving them to the augmenter.
        Training uses worker processes where the environment allows; validation stays
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
        train_ids = set(tr_keys)
        val_ids = set(val_keys)
        tr_cases = [case for case in self.cases if case.case_id in train_ids]
        val_cases = [case for case in self.cases if case.case_id in val_ids]

        dl_tr = MamaSynthBreastSliceDataLoader(
            tr_cases, self.batch_size, transforms=tr_transforms
        )
        dl_val = MamaSynthBreastSliceDataLoader(
            val_cases, self.batch_size, transforms=val_transforms
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

    def _save_example_batch_png(self, num_examples: int = 4):
        """Write a PNG of one augmented training batch with its breast masks overlaid."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from batchgenerators.utilities.file_and_folder_operations import join

        batch = next(self.dataloader_train)
        data = batch["data"]
        seg = (
            batch["target"][0] if isinstance(batch["target"], list) else batch["target"]
        )
        keys = batch.get("keys", [""] * data.shape[0])
        n = min(num_examples, data.shape[0])

        fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n), squeeze=False)
        for index in range(n):
            image = data[index, 0].detach().cpu().float().numpy()
            mask = seg[index, 0].detach().cpu().numpy()
            axes[index, 0].imshow(image, cmap="gray")
            axes[index, 0].set_title(str(keys[index]), fontsize=7)
            axes[index, 1].imshow(image, cmap="gray")
            axes[index, 1].imshow(
                np.ma.masked_where(mask == 0, mask),
                cmap="autumn",
                alpha=0.6,
                vmin=1,
                vmax=1,
            )
            axes[index, 1].set_title("image + breast mask", fontsize=8)
            for axis in axes[index]:
                axis.axis("off")
        fig.suptitle(
            f"epoch {self.current_epoch} - augmented breast training inputs", fontsize=9
        )
        fig.tight_layout()
        fig.savefig(join(self.output_folder, "train_examples.png"), dpi=100)
        plt.close(fig)


class nnUNetTrainerMamaSynthBreast_BS_32(nnUNetTrainerMamaSynthBreast):
    """Breast segmenter at batch size 32."""

    batch_size_override = 32
