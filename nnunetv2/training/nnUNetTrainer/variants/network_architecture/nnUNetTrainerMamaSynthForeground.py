"""2D segmentation of the acquired tissue support from pre-contrast slices.

Mirrors the breast segmenter, but targets the binary foreground mask and reads
its own dataset manifest.
"""

from __future__ import annotations

import ast
import csv
from functools import lru_cache
from pathlib import Path
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

FOREGROUND_LABEL = 1


def load_foreground_cases(dataset_dir: Path) -> List[MamaSynthCase]:
    """Read the case list for the foreground dataset from its manifest.

    Pre volumes and foreground masks are sibling blosc2 arrays named
    ``<case_id>_pre.b2nd`` and ``<case_id>_foreground_mask.b2nd``; where those are
    absent, the source and output paths recorded in the manifest are used instead.
    Rows the resumable converter marked ``skipped_existing`` are kept, because their
    outputs are already on disk. Rows with no slices or missing files are skipped.
    """
    manifest = dataset_dir / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Missing manifest.csv in dataset directory: {manifest}"
        )

    cases: List[MamaSynthCase] = []
    with manifest.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"status", "case_id", "n_slices", "spacing_3d"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"manifest.csv is missing required columns: {sorted(missing)}"
            )

        for row in reader:
            status = row["status"]
            if status not in {"ok", "skipped_existing"}:
                continue
            n_slices = int(row["n_slices"])
            if n_slices <= 0:
                continue

            spacing = ast.literal_eval(row["spacing_3d"])
            case_id = row["case_id"]
            pre_path = dataset_dir / f"{case_id}_pre.b2nd"
            foreground_path = dataset_dir / f"{case_id}_foreground_mask.b2nd"
            if not pre_path.is_file() and row.get("source_pre_b2nd"):
                pre_path = Path(row["source_pre_b2nd"])
            if not foreground_path.is_file() and row.get("output_path"):
                foreground_path = Path(row["output_path"])
            if not pre_path.is_file() or not foreground_path.is_file():
                continue

            cases.append(
                MamaSynthCase(
                    case_id=case_id,
                    n_slices=n_slices,
                    spacing_y=float(spacing[0]),
                    spacing_x=float(spacing[1]),
                    pre_path=pre_path,
                    post_path=pre_path,
                    mask_path=foreground_path,
                )
            )

    if not cases:
        raise ValueError(f"No usable foreground cases found in {manifest}")
    return cases


@lru_cache(maxsize=8192)
def foreground_slice_indices(mask_path: str) -> tuple:
    """Return the indices of all slices whose mask contains foreground.

    Cached per mask path, since the same volume is queried repeatedly while building
    a split.
    """
    mask = np.asarray(open_b2nd(Path(mask_path))[:])
    present = np.any(mask > 0, axis=(1, 2))
    return tuple(int(index) for index in np.nonzero(present)[0])


def load_pre_and_foreground_mask_slice(case: MamaSynthCase, slice_idx: int) -> tuple:
    """Load one pre-contrast slice and its foreground mask, ready for training.

    Both arrays are resampled to the common pixel spacing, the image is z-scored, and
    the two are placed into the fixed-size canvas at the same random offset so they
    stay aligned.
    """
    pre = np.asarray(open_b2nd(case.pre_path)[slice_idx], dtype=np.float32)
    foreground_mask = (
        np.asarray(open_b2nd(case.mask_path)[slice_idx], dtype=np.uint8) > 0
    ).astype(np.int16)

    pre = zscore_per_slice(
        resample_slice(pre, case.spacing_y, case.spacing_x, is_mask=False)
    ).astype(np.float32)
    foreground_mask = resample_slice(
        foreground_mask, case.spacing_y, case.spacing_x, is_mask=True
    ).astype(np.int16)

    placement_yx = (
        _random_placement(pre.shape[0], TARGET_SIZE),
        _random_placement(pre.shape[1], TARGET_SIZE),
    )
    pre = _place_into_canvas(pre, placement_yx, fill=0.0)
    foreground_mask = _place_into_canvas(
        foreground_mask, placement_yx, fill=BACKGROUND_LABEL
    )
    return np.ascontiguousarray(pre), np.ascontiguousarray(foreground_mask)


class MamaSynthForegroundSliceDataLoader(DataLoader):
    """Samples foreground-positive pre-contrast slices in nnU-Net batch format."""

    def __init__(self, cases: List[MamaSynthCase], batch_size: int, transforms=None):
        """Build the flat list of (case, slice index) pairs that contain foreground.

        Raises ValueError if the split has no such slice.
        """
        super().__init__(cases, batch_size, 1, None, True, False, True, None)
        self.cases = cases
        self.transforms = transforms
        self.positive_slices = [
            (case, slice_idx)
            for case in cases
            for slice_idx in foreground_slice_indices(str(case.mask_path))
        ]
        if not self.positive_slices:
            raise ValueError("Split contains no slices with foreground mask voxels.")

    def _sample_foreground_example(self, max_tries: int = 32) -> tuple:
        """Draw one random foreground-positive slice, retrying if the crop lost the mask.

        The random canvas placement can push the tissue outside the visible area, so up
        to ``max_tries`` samples are drawn until one still has mask voxels.
        """
        for _ in range(max_tries):
            sample_index = int(np.random.randint(len(self.positive_slices)))
            case, slice_idx = self.positive_slices[sample_index]
            pre, mask = load_pre_and_foreground_mask_slice(case, slice_idx)
            if mask.any():
                return pre, mask, f"{case.case_id}_s{slice_idx}"
        raise RuntimeError(
            f"Could not retain foreground after {max_tries} random crop attempts."
        )

    def generate_train_batch(self) -> dict:
        """Assemble one augmented batch of images and foreground masks.

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


class nnUNetTrainerMamaSynthForeground(MamaSynthDataMixin, nnUNetTrainer):
    """Standard 2D nnU-Net trained to segment the acquired tissue support."""

    @staticmethod
    def _load_mamasynth_cases(dataset_dir: Path):
        """Read cases from the foreground dataset manifest rather than the default one."""
        return load_foreground_cases(dataset_dir)

    def __init__(
        self, plans, configuration, fold, dataset_json, device=torch.device("cuda")
    ):
        """Replace the label set with background/foreground and fix the epoch schedule."""
        foreground_dataset_json = dict(dataset_json)
        foreground_dataset_json["labels"] = {
            "background": BACKGROUND_LABEL,
            "foreground": FOREGROUND_LABEL,
        }
        super().__init__(plans, configuration, fold, foreground_dataset_json, device)
        self.num_iterations_per_epoch = 1000
        self.num_val_iterations_per_epoch = 200
        self.num_epochs = 100

    def get_dataloaders(self):
        """Build augmented train and validation loaders over foreground-positive slices.

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

        dl_tr = MamaSynthForegroundSliceDataLoader(
            tr_cases, self.batch_size, transforms=tr_transforms
        )
        dl_val = MamaSynthForegroundSliceDataLoader(
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
        """Write a PNG of one augmented training batch with its foreground masks overlaid."""
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
                alpha=0.45,
                vmin=1,
                vmax=1,
            )
            axes[index, 1].set_title("image + foreground mask", fontsize=8)
            for axis in axes[index]:
                axis.axis("off")
        fig.suptitle(
            f"epoch {self.current_epoch} - augmented foreground training inputs",
            fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(join(self.output_folder, "train_examples.png"), dpi=100)
        plt.close(fig)


class nnUNetTrainerMamaSynthForeground_BS_32(nnUNetTrainerMamaSynthForeground):
    """Foreground segmenter at batch size 32."""

    batch_size_override = 32
