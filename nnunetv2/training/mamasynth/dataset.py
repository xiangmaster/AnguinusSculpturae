from __future__ import annotations

import ast
import csv
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import blosc2
import numpy as np
import torch
from torch.nn import functional as F

# Label convention of Dataset625_Pre_Seg masks.
BACKGROUND_LABEL = 0
LESION_LABEL = 1
BREAST_LABEL = 2

TARGET_SIZE = 512  # network input is a fixed 512x512 grid
TARGET_SPACING = (
    0.70  # mm; common in-plane spacing (dataset median) we resample every slice to
)


@dataclass(frozen=True)
class MamaSynthCase:
    case_id: str
    n_slices: int
    spacing_y: float
    spacing_x: float
    pre_path: Path
    post_path: Path
    mask_path: Path


#: Default split file and optional launcher-provided override.
SPLITS_FILE_NAME = "splits_final.json"
MAMASYNTH_SPLIT_FILE_ENV = "MAMASYNTH_SPLIT_FILE"


def load_cases(dataset_dir: Path) -> List[MamaSynthCase]:
    manifest = dataset_dir / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Missing manifest.csv in dataset directory: {manifest}"
        )
    cases: List[MamaSynthCase] = []
    with manifest.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"status", "post_status", "case_id", "n_slices", "spacing_3d"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"manifest.csv is missing required columns: {sorted(missing)}"
            )
        for row in reader:
            if row["status"] != "ok" or row["post_status"] != "ok":
                continue
            n_slices = int(row["n_slices"])
            if n_slices <= 0:
                continue
            # spacing_3d is [sy, sx, sz] (in-plane first two); we only need in-plane for 2D slices.
            spacing = ast.literal_eval(row["spacing_3d"])
            case_id = row["case_id"]
            cases.append(
                MamaSynthCase(
                    case_id=case_id,
                    n_slices=n_slices,
                    spacing_y=float(spacing[0]),
                    spacing_x=float(spacing[1]),
                    pre_path=dataset_dir / f"{case_id}_pre.b2nd",
                    post_path=dataset_dir / f"{case_id}_post.b2nd",
                    mask_path=dataset_dir / f"{case_id}_mask.b2nd",
                )
            )
    if not cases:
        raise ValueError(f"No usable cases found in {manifest}")
    return cases


def load_split(dataset_dir: Path, fold, split_file: Path | str | None = None) -> dict:
    """Read nnU-Net train/val case IDs from an explicit, environment, or dataset-local split file.

    Resolution order is ``split_file``, ``MAMASYNTH_SPLIT_FILE``, then
    ``<dataset_dir>/splits_final.json``. No split is computed in code.
    """
    configured_split = split_file or os.environ.get(MAMASYNTH_SPLIT_FILE_ENV)
    splits_file = (
        Path(configured_split).expanduser()
        if configured_split
        else Path(dataset_dir) / SPLITS_FILE_NAME
    )
    if not splits_file.is_file():
        raise FileNotFoundError(
            f"{splits_file} not found. Provide a {SPLITS_FILE_NAME} (list of per-fold "
            "{{'train': [...], 'val': [...]}}) in the dataset directory."
        )
    with splits_file.open("r", encoding="utf-8") as handle:
        folds = json.load(handle)
    if not isinstance(folds, list):
        raise ValueError(
            f"{splits_file} must be a list of per-fold {{'train','val'}} dicts."
        )
    fold_index = 0 if fold == "all" else int(fold)
    if fold_index >= len(folds):
        raise ValueError(
            f"{splits_file} defines {len(folds)} fold(s); fold {fold_index} was requested."
        )
    split = folds[fold_index]
    if "train" not in split or "val" not in split:
        raise ValueError(
            f"Fold {fold_index} in {splits_file} must contain 'train' and 'val' keys."
        )
    return {
        "train": [str(x) for x in split["train"]],
        "val": [str(x) for x in split["val"]],
    }


def open_b2nd(path: Path):
    dparams = {"nthreads": 1}
    mmap_kwargs = {} if os.name == "nt" else {"mmap_mode": "r"}
    return blosc2.open(str(path), mode="r", dparams=dparams, **mmap_kwargs)


@lru_cache(maxsize=8192)
def lesion_slice_indices(mask_path: str) -> tuple:
    """Slice indices (along axis 0) that contain the lesion label. Cached per case."""
    mask = np.asarray(open_b2nd(Path(mask_path))[:])
    present = np.any(mask == LESION_LABEL, axis=(1, 2))
    return tuple(int(i) for i in np.nonzero(present)[0])


def resample_slice(
    slice_2d: np.ndarray, spacing_y: float, spacing_x: float, is_mask: bool
) -> np.ndarray:
    """Resample a single (H, W) slice to TARGET_SPACING mm/px (nearest for masks, bilinear for images)."""
    h, w = slice_2d.shape
    new_h = max(1, round(h * spacing_y / TARGET_SPACING))
    new_w = max(1, round(w * spacing_x / TARGET_SPACING))
    if (new_h, new_w) == (h, w):
        return slice_2d
    tensor = torch.from_numpy(np.ascontiguousarray(slice_2d)).float()[None, None]
    if is_mask:
        resized = F.interpolate(tensor, size=(new_h, new_w), mode="nearest")
    else:
        resized = F.interpolate(
            tensor, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
    return resized[0, 0].numpy()


def zscore_per_slice(slice_2d: np.ndarray) -> np.ndarray:
    """Z-score a single slice individually (over all of its pixels)."""
    mean = float(slice_2d.mean())
    std = float(slice_2d.std())
    return (slice_2d - mean) / max(std, 1e-8)


def zscore_with_stats(slice_2d: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Z-score a slice with fixed dataset-level statistics."""
    return (slice_2d - float(mean)) / max(float(std), 1e-8)


def _random_placement(length: int, size: int) -> Tuple[int, int, int]:
    """Random src/dst offsets to fit a 1-D axis of ``length`` into ``size`` (crop if larger, pad if smaller)."""
    if length >= size:
        src = int(np.random.randint(0, length - size + 1))
        return src, 0, size
    dst = int(np.random.randint(0, size - length + 1))
    return 0, dst, length


def _place_into_canvas(arr: np.ndarray, placement_yx, fill) -> np.ndarray:
    (src_y, dst_y, len_y), (src_x, dst_x, len_x) = placement_yx
    canvas = np.full((TARGET_SIZE, TARGET_SIZE), fill, dtype=arr.dtype)
    canvas[dst_y : dst_y + len_y, dst_x : dst_x + len_x] = arr[
        src_y : src_y + len_y, src_x : src_x + len_x
    ]
    return canvas


def _centered_placement(length: int, size: int, center: float) -> Tuple[int, int, int]:
    if length >= size:
        src = int(np.clip(round(center - size / 2), 0, length - size))
        return src, 0, size
    return 0, (size - length) // 2, length


@lru_cache(maxsize=8192)
def peak_lesion_slice_and_center(
    mask_path: str, spacing_y: float, spacing_x: float
) -> tuple:
    """Return the maximum-lesion slice and its centroid after in-plane resampling."""
    mask_volume = np.asarray(open_b2nd(Path(mask_path))[:])
    lesion_counts = (mask_volume == LESION_LABEL).sum(axis=(1, 2))
    peak_slice = int(np.argmax(lesion_counts))
    if lesion_counts[peak_slice] == 0:
        raise ValueError(f"No lesion voxels in {mask_path}")
    peak_mask = resample_slice(
        (mask_volume[peak_slice] == LESION_LABEL).astype(np.uint8),
        spacing_y,
        spacing_x,
        is_mask=True,
    )
    lesion_y, lesion_x = np.nonzero(peak_mask)
    return peak_slice, (float(lesion_y.mean()), float(lesion_x.mean()))


def load_pre_and_mask_slice(case: MamaSynthCase, slice_idx: int) -> tuple:
    """Load one pre slice and its mask: resample to common spacing, z-score the image, then a shared
    random 512x512 crop (when larger) or pad (when smaller). Returns (512x512 float32, 512x512 int16).
    """
    pre = np.asarray(open_b2nd(case.pre_path)[slice_idx], dtype=np.float32)
    lesion_mask = np.asarray(open_b2nd(case.mask_path)[slice_idx], dtype=np.uint8)
    lesion_mask[lesion_mask != LESION_LABEL] = (
        BACKGROUND_LABEL  # lesion-only task: breast -> background
    )

    pre = zscore_per_slice(
        resample_slice(pre, case.spacing_y, case.spacing_x, is_mask=False)
    ).astype(np.float32)
    lesion_mask = resample_slice(
        lesion_mask, case.spacing_y, case.spacing_x, is_mask=True
    ).astype(np.int16)

    placement_yx = (
        _random_placement(pre.shape[0], TARGET_SIZE),
        _random_placement(pre.shape[1], TARGET_SIZE),
    )
    pre = _place_into_canvas(pre, placement_yx, fill=0.0)  # 0 == mean after z-score
    lesion_mask = _place_into_canvas(lesion_mask, placement_yx, fill=BACKGROUND_LABEL)
    return np.ascontiguousarray(pre), np.ascontiguousarray(lesion_mask)


def load_pre_post_delta_slice(
    case: MamaSynthCase,
    slice_idx: int,
    mean: float,
    std: float,
    crop_center_yx: Tuple[float, float] | None = None,
) -> tuple:
    """Load an aligned pre/post pair and explicit loss-region masks.

    ``acquired_image_mask`` is one for every pixel originating from the resampled scan and zero only
    for synthetic 512x512 canvas padding. It is independent of anatomical labels: label 0 may be air,
    thorax, abdomen, lung, or other non-breast anatomy and remains part of the acquired image.
    ``lesion_mask`` is one only for anatomical label 1; label 2 (breast) is not part of this ROI.
    """
    pre = np.asarray(open_b2nd(case.pre_path)[slice_idx], dtype=np.float32)
    post = np.asarray(open_b2nd(case.post_path)[slice_idx], dtype=np.float32)
    lesion_mask = np.asarray(open_b2nd(case.mask_path)[slice_idx], dtype=np.uint8)
    lesion_mask[lesion_mask != LESION_LABEL] = BACKGROUND_LABEL

    pre = zscore_with_stats(
        resample_slice(pre, case.spacing_y, case.spacing_x, is_mask=False), mean, std
    )
    post = zscore_with_stats(
        resample_slice(post, case.spacing_y, case.spacing_x, is_mask=False), mean, std
    )
    lesion_mask = resample_slice(
        lesion_mask, case.spacing_y, case.spacing_x, is_mask=True
    ).astype(np.int16)
    acquired_image_mask = np.ones(pre.shape, dtype=np.float32)

    if crop_center_yx is None:
        placement_yx = (
            _random_placement(pre.shape[0], TARGET_SIZE),
            _random_placement(pre.shape[1], TARGET_SIZE),
        )
    else:
        placement_yx = (
            _centered_placement(pre.shape[0], TARGET_SIZE, crop_center_yx[0]),
            _centered_placement(pre.shape[1], TARGET_SIZE, crop_center_yx[1]),
        )
    pre = _place_into_canvas(pre.astype(np.float32), placement_yx, fill=0.0)
    post = _place_into_canvas(post.astype(np.float32), placement_yx, fill=0.0)
    lesion_mask = _place_into_canvas(lesion_mask, placement_yx, fill=BACKGROUND_LABEL)
    acquired_image_mask = _place_into_canvas(
        acquired_image_mask, placement_yx, fill=0.0
    )
    delta = post - pre
    return (
        np.ascontiguousarray(pre),
        np.ascontiguousarray(delta.astype(np.float32)),
        np.ascontiguousarray(post),
        np.ascontiguousarray(acquired_image_mask.astype(np.float32)),
        np.ascontiguousarray(lesion_mask),
    )
