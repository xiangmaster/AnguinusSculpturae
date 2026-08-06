"""Slice 3D pre-contrast NIfTI cases under raw/ into per-case blosc2 NDArrays.

Output layout
-------------
<output_root>/<dataset_name>/
    <case>_pre.b2nd        # float32, shape (N, H, W), chunks=(1, H, W)
    <case>_post.b2nd       # float32, optional (skipped if post NIfTI missing/mismatched)
    <case>_mask.b2nd       # uint8,   shape (N, H, W), chunks=(1, H, W)
    manifest.csv

The 2D slices stored are the high-resolution in-plane slices.
The slice axis is normally the coarse through-plane axis. For sandwich
volumes this is ``argmax(spacing)``. If two spacing axes tie and the third
differs, the tied axes are treated as the in-plane axes and the differing
axis is sliced. If all spacings tie, the shortest shape axis is used as the
frame/through-plane axis. ``chunks=(1, H, W)`` means one chunk per high-res 2D
slice, so ``blosc2.open(path)[i]`` decompresses exactly one slice (zstd,
~1-5 ms on this data).

Each .b2nd file carries vlmeta:
    case_id            : str
    through_axis_3d    : int
    spacing_3d         : tuple[float,float,float]   (original NIfTI zooms)
    slice_spacing_2d   : tuple[float,float]         (in-plane spacing, numpy order)
    retained_indices_3d: list[int]                  (slice indices in the original 3D volume)
    dataset_source     : str                        (case_id prefix grouping)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import blosc2
import nibabel as nib
import numpy as np
from tqdm import tqdm

from nnunetv2.preprocessing.mamasynth.nifti_io import load_nifti_float32

DEFAULT_DATASET_NAME = "Dataset625_Pre_Seg"

BREAST_LABEL = 2

MANIFEST_FIELDS = [
    "status",
    "reason",
    "case_id",
    "n_slices",
    "through_axis_3d",
    "image_shape_3d",
    "spacing_3d",
    "slice_spacing_2d",
    "post_status",
    "pre_path",
    "post_path",
    "mask_path",
]

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def load_mask_uint8(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Load a NIfTI mask as uint8 and return ``(data, spacing[:3])``."""
    img = nib.load(str(path))
    data = np.asarray(img.dataobj).astype(np.uint8)
    spacing = tuple(float(v) for v in img.header.get_zooms()[:3])
    return data, spacing  # type: ignore[return-value]


def read_nifti_header(path: Path) -> Tuple[Tuple[int, ...], Tuple[float, float, float]]:
    """Read NIfTI shape and 3D spacing without loading voxel data."""
    img = nib.load(str(path))
    shape = tuple(int(v) for v in img.shape)
    spacing = tuple(float(v) for v in img.header.get_zooms()[:3])
    return shape, spacing  # type: ignore[return-value]


def inplane_spacing_2d(
    spacing_3d: Tuple[float, float, float], through_axis: int
) -> Tuple[float, float]:
    """Return the two in-plane spacings in numpy axis order after ``np.take``."""
    remaining = [spacing_3d[i] for i in range(3) if i != through_axis]
    return float(remaining[0]), float(remaining[1])


def select_slice_axis(
    spacing: Tuple[float, float, float], shape: Tuple[int, ...]
) -> Tuple[int | None, str | None]:
    """Select the through-plane axis for high-resolution 2D slice extraction."""
    if len(spacing) < 3:
        return None, f"invalid spacing length: {tuple(spacing)}"
    if len(shape) < 3:
        return None, f"invalid shape length: {tuple(shape)}"

    spacing_arr = np.asarray(spacing[:3], dtype=np.float64)
    if not np.all(np.isfinite(spacing_arr)) or np.any(spacing_arr <= 0):
        return None, f"invalid spacing values: {tuple(float(v) for v in spacing_arr)}"

    max_spacing = float(np.max(spacing_arr))
    max_axes = np.flatnonzero(np.isclose(spacing_arr, max_spacing))
    if len(max_axes) == 1:
        return int(max_axes[0]), None

    if len(max_axes) == 2:
        axis = int(({0, 1, 2} - set(int(a) for a in max_axes)).pop())
        return axis, (
            "two axes tied for largest spacing "
            f"{tuple(float(v) for v in spacing_arr)}; treating tied axes as "
            f"in-plane and using differing axis {axis}"
        )

    shape_arr = np.asarray(shape[:3], dtype=np.int64)
    min_size = int(np.min(shape_arr))
    min_axes = np.flatnonzero(shape_arr == min_size)
    axis = int(min_axes[0])
    if len(min_axes) > 1:
        return axis, (
            "all axes tied for spacing "
            f"{tuple(float(v) for v in spacing_arr)} and multiple shortest "
            f"shape axes {tuple(int(a) for a in min_axes)} in shape "
            f"{tuple(int(v) for v in shape_arr)}; using axis {axis}"
        )
    return axis, (
        "all axes tied for spacing "
        f"{tuple(float(v) for v in spacing_arr)}; using shortest shape axis "
        f"{axis} from shape {tuple(int(v) for v in shape_arr)}"
    )


def stack_retained_slices(
    volume: np.ndarray, retained: List[int], axis: int
) -> np.ndarray:
    """Stack the retained 2D high-res slices along a new leading axis 0."""
    moved = np.moveaxis(volume, axis, 0)
    return np.ascontiguousarray(moved[retained])


def label_slice_indices(mask: np.ndarray, label: int, axis: int) -> List[int]:
    """Return slice indices where ``mask`` contains ``label`` on ``axis``."""
    other_axes = tuple(i for i in range(mask.ndim) if i != axis)
    has_label = np.any(mask == label, axis=other_axes)
    return [int(idx) for idx in np.flatnonzero(has_label)]


def discover_pre_cases(raw_root: Path) -> List[Tuple[str, Path, Path, Path]]:
    """Enumerate ``(case_id, pre_path, post_path, mask_path)`` from raw/."""
    images_dir = raw_root / "images"
    masks_dir = raw_root / "masks"
    out: List[Tuple[str, Path, Path, Path]] = []
    for pre_path in sorted(images_dir.glob("*_0000.nii.gz")):
        case_id = pre_path.name[: -len("_0000.nii.gz")]
        post_path = images_dir / f"{case_id}_0001.nii.gz"
        mask_path = masks_dir / f"{case_id}.nii.gz"
        out.append((case_id, pre_path, post_path, mask_path))
    return out


def dataset_source_from_case(case_id: str) -> str:
    """Coarse grouping label from the case_id (e.g. 'Breast_MRI', 'AMBL', 'ACRIN')."""
    head = case_id.split("_", 1)[0] if "_" in case_id else case_id
    return head


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #


def _write_b2nd(
    arr: np.ndarray,
    urlpath: Path,
    clevel: int,
    use_shuffle: bool,
    vlmeta: Dict[str, object],
) -> None:
    """Persist ``arr`` as a chunked blosc2 NDArray with one chunk per axis-0 slice."""
    if arr.ndim != 3:
        raise ValueError(f"expected (N, H, W) array, got shape {arr.shape}")
    chunks = (1, arr.shape[1], arr.shape[2])
    cparams = {
        "codec": blosc2.Codec.ZSTD,
        "clevel": int(clevel),
        "filters": [blosc2.Filter.SHUFFLE if use_shuffle else blosc2.Filter.NOFILTER],
    }
    urlpath.parent.mkdir(parents=True, exist_ok=True)
    tmp_urlpath = urlpath.with_name(f"{urlpath.name}.tmp.{os.getpid()}.b2nd")
    if tmp_urlpath.exists():
        tmp_urlpath.unlink()
    try:
        nd = blosc2.asarray(
            arr,
            chunks=chunks,
            blocks=chunks,
            urlpath=str(tmp_urlpath),
            mode="w",
            cparams=cparams,
        )
        for k, v in vlmeta.items():
            nd.vlmeta[k] = v
        del nd
        tmp_urlpath.replace(urlpath)
    finally:
        if tmp_urlpath.exists():
            tmp_urlpath.unlink()


# --------------------------------------------------------------------------- #
# Per-case worker
# --------------------------------------------------------------------------- #


def _skip_row(case_id: str, reason: str, **extra) -> Dict[str, object]:
    row: Dict[str, object] = {f: "" for f in MANIFEST_FIELDS}
    row.update(
        {"status": "skipped", "reason": reason, "case_id": case_id, "n_slices": 0}
    )
    row.update(extra)
    return row


def process_case(
    case_id: str,
    pre_path: Path,
    post_path: Path,
    mask_path: Path,
    output_dir: Path,
    clevel: int,
    overwrite: bool,
    dry_run: bool,
) -> Dict[str, object]:
    try:
        if not pre_path.is_file():
            return _skip_row(
                case_id, f"missing_pre: {pre_path}", pre_path=str(pre_path)
            )
        if not mask_path.is_file():
            return _skip_row(
                case_id, f"missing_mask: {mask_path}", mask_path=str(mask_path)
            )

        pre_b2nd = output_dir / f"{case_id}_pre.b2nd"
        post_b2nd = output_dir / f"{case_id}_post.b2nd"
        mask_b2nd = output_dir / f"{case_id}_mask.b2nd"

        # Idempotency: skip if pre+mask outputs exist and (post output exists or
        # post NIfTI doesn't exist). Sidecars are inside .b2nd vlmeta — no extra
        # files to track.
        post_input_present = post_path.is_file()
        outputs_ok = (
            pre_b2nd.exists()
            and mask_b2nd.exists()
            and (post_b2nd.exists() or not post_input_present)
        )
        if outputs_ok and not overwrite:
            return {
                "status": "skipped_existing",
                "reason": "",
                "case_id": case_id,
                "n_slices": 0,
                "through_axis_3d": "",
                "image_shape_3d": "",
                "spacing_3d": "",
                "slice_spacing_2d": "",
                "post_status": "existing" if post_b2nd.exists() else "missing",
                "pre_path": str(pre_path),
                "post_path": str(post_path) if post_input_present else "",
                "mask_path": str(mask_path),
            }

        pre_shape, spacing_3d = read_nifti_header(pre_path)
        mask_shape, _ = read_nifti_header(mask_path)
        if len(pre_shape) != 3:
            return _skip_row(
                case_id,
                f"invalid_pre_ndim: pre shape {pre_shape}",
                image_shape_3d=json.dumps(list(pre_shape)),
                pre_path=str(pre_path),
                mask_path=str(mask_path),
            )
        if mask_shape != pre_shape:
            return _skip_row(
                case_id,
                f"shape_mismatch: pre {pre_shape} vs mask {mask_shape}",
                image_shape_3d=json.dumps(list(pre_shape)),
                pre_path=str(pre_path),
                mask_path=str(mask_path),
            )

        axis, spacing_msg = select_slice_axis(spacing_3d, pre_shape)
        if axis is None:
            return _skip_row(
                case_id,
                spacing_msg or "invalid_spacing",
                image_shape_3d=json.dumps(list(pre_shape)),
                spacing_3d=json.dumps(list(spacing_3d)),
                pre_path=str(pre_path),
                mask_path=str(mask_path),
            )
        if spacing_msg:
            logger.warning("%s: %s", case_id, spacing_msg)

        mask, _ = load_mask_uint8(mask_path)
        retained = label_slice_indices(mask, BREAST_LABEL, axis)
        if not retained:
            return _skip_row(
                case_id,
                "no_breast_voxels",
                through_axis_3d=axis,
                image_shape_3d=json.dumps(list(pre_shape)),
                spacing_3d=json.dumps(list(spacing_3d)),
                pre_path=str(pre_path),
                mask_path=str(mask_path),
            )

        sl_spacing_2d = inplane_spacing_2d(spacing_3d, axis)

        # Post handling (optional)
        post_status = "ok" if post_input_present else "missing_post"
        post_can_write = False
        if post_input_present:
            post_shape, _ = read_nifti_header(post_path)
            if post_shape != pre_shape:
                post_status = (
                    f"shape_mismatch_post: post {post_shape} vs pre {pre_shape}"
                )
            else:
                post_can_write = True

        # Common vlmeta payload (msgpack-serialised by blosc2)
        common_meta: Dict[str, object] = {
            "case_id": case_id,
            "through_axis_3d": int(axis),
            "spacing_3d": list(spacing_3d),
            "slice_spacing_2d": list(sl_spacing_2d),
            "retained_indices_3d": [int(i) for i in retained],
            "dataset_source": dataset_source_from_case(case_id),
        }

        if dry_run:
            return {
                "status": "ok",
                "reason": spacing_msg or "",
                "case_id": case_id,
                "n_slices": len(retained),
                "through_axis_3d": axis,
                "image_shape_3d": json.dumps(list(pre_shape)),
                "spacing_3d": json.dumps(list(spacing_3d)),
                "slice_spacing_2d": json.dumps(list(sl_spacing_2d)),
                "post_status": post_status,
                "pre_path": str(pre_path),
                "post_path": str(post_path) if post_input_present else "",
                "mask_path": str(mask_path),
            }

        # Stack and write
        pre, _ = load_nifti_float32(pre_path)
        pre_stack = stack_retained_slices(pre, retained, axis)
        _write_b2nd(
            pre_stack,
            pre_b2nd,
            clevel=clevel,
            use_shuffle=True,
            vlmeta={**common_meta, "modality": "pre", "dtype": "float32"},
        )
        del pre, pre_stack

        mask_stack = stack_retained_slices(mask, retained, axis)
        _write_b2nd(
            mask_stack,
            mask_b2nd,
            clevel=clevel,
            use_shuffle=False,
            vlmeta={**common_meta, "modality": "mask", "dtype": "uint8"},
        )
        del mask, mask_stack

        if post_can_write:
            post, _ = load_nifti_float32(post_path)
            post_stack = stack_retained_slices(post, retained, axis)
            _write_b2nd(
                post_stack,
                post_b2nd,
                clevel=clevel,
                use_shuffle=True,
                vlmeta={**common_meta, "modality": "post", "dtype": "float32"},
            )
            del post, post_stack

        return {
            "status": "ok",
            "reason": spacing_msg or "",
            "case_id": case_id,
            "n_slices": len(retained),
            "through_axis_3d": axis,
            "image_shape_3d": json.dumps(list(pre_shape)),
            "spacing_3d": json.dumps(list(spacing_3d)),
            "slice_spacing_2d": json.dumps(list(sl_spacing_2d)),
            "post_status": post_status,
            "pre_path": str(pre_path),
            "post_path": str(post_path) if post_input_present else "",
            "mask_path": str(mask_path),
        }

    except Exception as exc:  # noqa: BLE001 - audit-safe outer guard
        return _skip_row(case_id, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# Run / output
# --------------------------------------------------------------------------- #


def write_manifest(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw_root",
        type=Path,
        required=True,
        help="Root holding the per-case 3D NIfTI inputs (<raw_root>/<case>/...).",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Where to create <output_root>/<dataset_name>/ with the .b2nd stacks.",
    )
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--clevel",
        type=int,
        default=3,
        help="zstd compression level (1-9). Use 1 for speed-first runs.",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Whitelist of case_ids (smoke test convenience).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N cases (after --cases filtering).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = args.output_root / args.dataset_name
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    cases = discover_pre_cases(args.raw_root)
    if args.cases:
        whitelist = set(args.cases)
        cases = [c for c in cases if c[0] in whitelist]
    if args.limit:
        cases = cases[: args.limit]
    logger.info("Found %d cases under %s", len(cases), args.raw_root)
    if not cases:
        logger.warning("No cases to process; exiting")
        return

    rows: List[Dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                process_case,
                case_id,
                pre_path,
                post_path,
                mask_path,
                output_dir,
                args.clevel,
                args.overwrite,
                args.dry_run,
            ): case_id
            for case_id, pre_path, post_path, mask_path in cases
        }
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=args.dataset_name,
            unit="case",
        ):
            rows.append(fut.result())

    rows.sort(key=lambda r: str(r["case_id"]))

    statuses: Dict[str, int] = {}
    post_statuses: Dict[str, int] = {}
    total_slices = 0
    for r in rows:
        statuses[str(r["status"])] = statuses.get(str(r["status"]), 0) + 1
        ps = str(r.get("post_status") or "")
        if ps:
            post_statuses[ps] = post_statuses.get(ps, 0) + 1
        total_slices += int(r.get("n_slices") or 0)
    logger.info("Per-case statuses: %s", statuses)
    logger.info("Post statuses: %s", post_statuses)
    logger.info("Total retained slices: %d", total_slices)

    if args.dry_run:
        logger.info("Dry run — manifest not written")
        return

    write_manifest(output_dir / "manifest.csv", rows)
    logger.info("Wrote manifest: %s", output_dir / "manifest.csv")


if __name__ == "__main__":
    main()
