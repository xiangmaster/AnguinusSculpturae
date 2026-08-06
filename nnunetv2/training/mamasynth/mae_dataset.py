from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from nnunetv2.training.mamasynth.dataset import (
    TARGET_SIZE,
    _centered_placement,
    _place_into_canvas,
    _random_placement,
    open_b2nd,
    resample_slice,
    zscore_per_slice,
)


@dataclass(frozen=True)
class MamaSynthMAECase:
    """One unlabeled blosc2 volume from Dataset626_MAE (any modality: pre/post/t2/sub)."""

    volume_id: str
    n_slices: int
    spacing_y: float
    spacing_x: float
    image_path: Path


def load_mae_cases(dataset_dir: Path) -> List[MamaSynthMAECase]:
    """Read the MAE manifest. Unlike Dataset625, the schema keys volumes by ``volume_id`` (one
    modality per row), carries no masks and no ``post_status``. Each volume is ``<volume_id>.b2nd``
    shaped ``(n_slices, H, W)``; only in-plane spacing matters for 2D slices."""
    manifest = dataset_dir / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Missing manifest.csv in dataset directory: {manifest}"
        )
    cases: List[MamaSynthMAECase] = []
    with manifest.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"status", "volume_id", "n_slices", "spacing_3d"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"manifest.csv is missing required columns: {sorted(missing)}"
            )
        for row in reader:
            if row["status"] != "ok":
                continue
            n_slices = int(row["n_slices"])
            if n_slices <= 0:
                continue
            spacing = ast.literal_eval(row["spacing_3d"])  # [sy, sx, sz]
            volume_id = row["volume_id"]
            cases.append(
                MamaSynthMAECase(
                    volume_id=volume_id,
                    n_slices=n_slices,
                    spacing_y=float(spacing[0]),
                    spacing_x=float(spacing[1]),
                    image_path=dataset_dir / f"{volume_id}.b2nd",
                )
            )
    if not cases:
        raise ValueError(f"No usable cases found in {manifest}")
    return cases


def load_image_slice(
    case: MamaSynthMAECase,
    slice_idx: int,
    crop_center_yx: Tuple[float, float] | None = None,
    center_crop: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a single image slice: resample to TARGET_SPACING mm/px, z-score **per slice**, then a
    512x512 crop/pad. Placement is random by default, centered on the slice midpoint when
    ``center_crop`` is set, or centered on ``crop_center_yx`` when that is given.

    Per-slice (not global) normalization is essential here: the MAE pool mixes source datasets whose
    raw intensities span 0-255 (e.g. BreaDM) to 0-11000 (e.g. duke), so a single global mean/std maps
    them to wildly different ranges (z from -0.5 to ~50). Per-slice z-score puts every slice on a
    comparable unit-variance scale. The encoder's conv->InstanceNorm stem is invariant to the
    per-sample affine difference vs the global-stats fine-tune, so transfer is unaffected.

    Returns ``(image, acquired_image_mask)`` both ``float32`` and C-contiguous. ``acquired_image_mask``
    is 1 for pixels originating from the scan and 0 for synthetic canvas padding.
    """
    image = np.asarray(open_b2nd(case.image_path)[slice_idx], dtype=np.float32)
    image = zscore_per_slice(
        resample_slice(image, case.spacing_y, case.spacing_x, is_mask=False)
    )
    acquired_image_mask = np.ones(image.shape, dtype=np.float32)

    if center_crop and crop_center_yx is None:
        crop_center_yx = (image.shape[0] / 2.0, image.shape[1] / 2.0)
    if crop_center_yx is None:
        placement_yx = (
            _random_placement(image.shape[0], TARGET_SIZE),
            _random_placement(image.shape[1], TARGET_SIZE),
        )
    else:
        placement_yx = (
            _centered_placement(image.shape[0], TARGET_SIZE, crop_center_yx[0]),
            _centered_placement(image.shape[1], TARGET_SIZE, crop_center_yx[1]),
        )
    image = _place_into_canvas(image.astype(np.float32), placement_yx, fill=0.0)
    acquired_image_mask = _place_into_canvas(
        acquired_image_mask, placement_yx, fill=0.0
    )
    return np.ascontiguousarray(image), np.ascontiguousarray(
        acquired_image_mask.astype(np.float32)
    )
