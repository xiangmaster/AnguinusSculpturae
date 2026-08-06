"""Convert all image NIfTI volumes under a root into 2D Blosc2 slice stacks.

Each 3D NIfTI is handled independently. Files below directories named
``masks`` are excluded. The coarse through-plane axis is selected from the
voxel spacing, and all high-resolution in-plane slices are stored as a
float32 array with shape ``(N, H, W)`` and one Blosc2 chunk per slice.

Output layout
-------------
<output_dir>/
    <collision-safe-volume-id>.b2nd
    manifest.csv

The stored NDArray carries vlmeta describing the original source path,
volume ID, shape, spacing, selected axis, in-plane spacing, and original
slice indices.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple

import blosc2
import nibabel as nib
import numpy as np
from tqdm import tqdm

from nnunetv2.preprocessing.mamasynth.nifti_io import load_nifti_float32

MANIFEST_FIELDS = [
    "status",
    "reason",
    "volume_id",
    "source_path",
    "relative_path",
    "output_path",
    "n_slices",
    "through_axis_3d",
    "image_shape_3d",
    "spacing_3d",
    "slice_spacing_2d",
]

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def read_nifti_header(
    path: Path,
) -> Tuple[Tuple[int, ...], Tuple[float, float, float]]:
    """Read NIfTI shape and 3D spacing without loading voxel data."""
    img = nib.load(str(path))
    shape = tuple(int(v) for v in img.shape)
    zooms = img.header.get_zooms()
    spacing = tuple(float(v) for v in zooms[:3])
    return shape, spacing  # type: ignore[return-value]


def inplane_spacing_2d(
    spacing_3d: Tuple[float, float, float], through_axis: int
) -> Tuple[float, float]:
    """Return in-plane spacings in numpy order after moving the slice axis."""
    remaining = [spacing_3d[i] for i in range(3) if i != through_axis]
    return float(remaining[0]), float(remaining[1])


def select_slice_axis(
    spacing: Tuple[float, float, float], shape: Tuple[int, ...]
) -> Tuple[int | None, str | None]:
    """Select the through-plane axis for high-resolution 2D slices."""
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


def stack_all_slices(volume: np.ndarray, axis: int) -> np.ndarray:
    """Move the through-plane axis first and return a contiguous slice stack."""
    return np.ascontiguousarray(np.moveaxis(volume, axis, 0))


def is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def iter_niftis(source_root: Path) -> Iterator[Path]:
    """Yield image NIfTIs deterministically while pruning ``masks`` trees."""
    for current_root, dirnames, filenames in os.walk(source_root):
        dirnames[:] = sorted(
            dirname for dirname in dirnames if dirname.casefold() != "masks"
        )
        for filename in sorted(filenames):
            path = Path(current_root) / filename
            if is_nifti(path):
                yield path


def strip_nifti_suffix(path_text: str) -> str:
    lower = path_text.lower()
    if lower.endswith(".nii.gz"):
        return path_text[:-7]
    if lower.endswith(".nii"):
        return path_text[:-4]
    return path_text


def make_volume_id(relative_path: Path) -> str:
    """Create a readable, deterministic, collision-safe ID from a relative path."""
    relative_text = relative_path.as_posix()
    stem = strip_nifti_suffix(relative_text)
    readable = re.sub(r"[^A-Za-z0-9._-]+", "__", stem).strip("._-")
    digest = hashlib.sha1(relative_text.encode("utf-8")).hexdigest()[:12]
    return f"{readable}__{digest}" if readable else digest


def _write_b2nd(
    arr: np.ndarray,
    urlpath: Path,
    clevel: int,
    vlmeta: Dict[str, object],
) -> None:
    """Atomically write a float32 stack with one Blosc2 chunk per slice."""
    if arr.ndim != 3:
        raise ValueError(f"expected (N, H, W) array, got shape {arr.shape}")
    chunks = (1, arr.shape[1], arr.shape[2])
    cparams = {
        "codec": blosc2.Codec.ZSTD,
        "clevel": int(clevel),
        "filters": [blosc2.Filter.SHUFFLE],
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
        for key, value in vlmeta.items():
            nd.vlmeta[key] = value
        del nd
        tmp_urlpath.replace(urlpath)
    finally:
        if tmp_urlpath.exists():
            tmp_urlpath.unlink()


def manifest_row(**values: object) -> Dict[str, object]:
    row: Dict[str, object] = {field: "" for field in MANIFEST_FIELDS}
    row.update(values)
    return row


def process_volume(
    source_path: Path,
    source_root: Path,
    output_dir: Path,
    clevel: int,
    overwrite: bool,
    dry_run: bool,
) -> Dict[str, object]:
    relative_path = source_path.relative_to(source_root)
    relative_text = relative_path.as_posix()
    volume_id = make_volume_id(relative_path)
    output_path = output_dir / f"{volume_id}.b2nd"

    common_row: Dict[str, object] = {
        "volume_id": volume_id,
        "source_path": str(source_path),
        "relative_path": relative_text,
        "output_path": str(output_path),
    }

    try:
        shape, spacing_3d = read_nifti_header(source_path)
        if len(shape) != 3:
            return manifest_row(
                status="skipped",
                reason=f"invalid_ndim: shape {shape}",
                image_shape_3d=json.dumps(list(shape)),
                spacing_3d=json.dumps(list(spacing_3d)),
                n_slices=0,
                **common_row,
            )

        axis, spacing_msg = select_slice_axis(spacing_3d, shape)
        if axis is None:
            return manifest_row(
                status="skipped",
                reason=spacing_msg or "invalid_spacing",
                image_shape_3d=json.dumps(list(shape)),
                spacing_3d=json.dumps(list(spacing_3d)),
                n_slices=0,
                **common_row,
            )
        if spacing_msg:
            logger.debug("%s: %s", relative_text, spacing_msg)

        slice_spacing = inplane_spacing_2d(spacing_3d, axis)
        n_slices = int(shape[axis])

        if output_path.exists() and not overwrite:
            return manifest_row(
                status="skipped_existing",
                reason="",
                n_slices=n_slices,
                through_axis_3d=axis,
                image_shape_3d=json.dumps(list(shape)),
                spacing_3d=json.dumps(list(spacing_3d)),
                slice_spacing_2d=json.dumps(list(slice_spacing)),
                **common_row,
            )

        if not dry_run:
            volume, _ = load_nifti_float32(source_path)
            stack = stack_all_slices(volume, axis)
            _write_b2nd(
                stack,
                output_path,
                clevel=clevel,
                vlmeta={
                    "source_path": str(source_path),
                    "relative_path": relative_text,
                    "volume_id": volume_id,
                    "image_shape_3d": list(shape),
                    "spacing_3d": list(spacing_3d),
                    "through_axis_3d": int(axis),
                    "slice_spacing_2d": list(slice_spacing),
                    "retained_indices_3d": list(range(n_slices)),
                    "dtype": "float32",
                },
            )
            del volume, stack

        return manifest_row(
            status="ok",
            reason=spacing_msg or "",
            n_slices=n_slices,
            through_axis_3d=axis,
            image_shape_3d=json.dumps(list(shape)),
            spacing_3d=json.dumps(list(spacing_3d)),
            slice_spacing_2d=json.dumps(list(slice_spacing)),
            **common_row,
        )
    except Exception as exc:  # noqa: BLE001 - preserve an audit row per input
        return manifest_row(
            status="skipped",
            reason=f"{type(exc).__name__}: {exc}",
            n_slices=0,
            **common_row,
        )


def write_manifest(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in MANIFEST_FIELDS})


def collect_completed(
    pending: Set[Future[Dict[str, object]]],
    rows: List[Dict[str, object]],
    progress: tqdm,
    wait_for_all: bool,
) -> Set[Future[Dict[str, object]]]:
    """Collect completed worker results and return futures still in flight."""
    if not pending:
        return pending
    if wait_for_all:
        done, remaining = wait(pending)
    else:
        done, remaining = wait(pending, return_when=FIRST_COMPLETED)
    rows.extend(future.result() for future in done)
    progress.update(len(done))
    return set(remaining)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source_root",
        type=Path,
        required=True,
        help="Root searched recursively for image NIfTI volumes (dirs named 'masks' "
        "are excluded).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for the Dataset626_MAE .b2nd slice stacks.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--clevel",
        type=int,
        default=3,
        choices=range(1, 10),
        metavar="1-9",
        help="ZSTD compression level.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N volumes after deterministic path sorting.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source_root.is_dir():
        raise SystemExit(f"Source root does not exist: {args.source_root}")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    seen_volume_ids: Set[str] = set()
    discovered_count = 0
    max_pending = max(args.workers * 2, 1)
    progress = tqdm(
        total=args.limit,
        desc="Dataset626_MAE",
        unit="volume",
        dynamic_ncols=True,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending: Set[Future[Dict[str, object]]] = set()
        for path in iter_niftis(args.source_root):
            if args.limit is not None and discovered_count >= args.limit:
                break

            relative_path = path.relative_to(args.source_root)
            volume_id = make_volume_id(relative_path)
            if volume_id in seen_volume_ids:
                raise RuntimeError(
                    f"Internal error: duplicate volume ID for {relative_path}: "
                    f"{volume_id}"
                )
            seen_volume_ids.add(volume_id)
            discovered_count += 1

            pending.add(
                executor.submit(
                    process_volume,
                    path,
                    args.source_root,
                    args.output_dir,
                    args.clevel,
                    args.overwrite,
                    args.dry_run,
                )
            )
            progress.set_postfix(discovered=discovered_count, queued=len(pending))

            if len(pending) >= max_pending:
                pending = collect_completed(
                    pending, rows, progress, wait_for_all=False
                )

        pending = collect_completed(pending, rows, progress, wait_for_all=True)
    progress.close()

    logger.info(
        "Discovered and processed %d non-mask NIfTIs under %s",
        discovered_count,
        args.source_root,
    )
    if not rows:
        logger.warning("No input NIfTIs found; exiting")
        return

    rows.sort(key=lambda row: str(row["relative_path"]))
    status_counts: Dict[str, int] = {}
    total_slices = 0
    for row in rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        if status in {"ok", "skipped_existing"}:
            total_slices += int(row.get("n_slices") or 0)

    logger.info("Statuses: %s", status_counts)
    logger.info("Total usable slices: %d", total_slices)

    if args.dry_run:
        logger.info("Dry run: manifest and Blosc2 files were not written")
        return

    manifest_path = args.output_dir / "manifest.csv"
    write_manifest(manifest_path, rows)
    logger.info("Wrote manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
