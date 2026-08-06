"""Create Blosc2 foreground mask stacks aligned to Dataset625 pre images.

Input layout
------------
Dataset625 pre Blosc2 directory:
    <case>_pre.b2nd

Foreground mask root:
    <chunk>/<source>__<rest>.nii.gz

Output layout
-------------
<output_dir>/
    <case>_foreground_mask.b2nd   # uint8, shape (N, H, W), chunks=(1, H, W)
    manifest.csv

The foreground mask stacks reuse the slice axis and retained slice indices
stored in the Dataset625 pre .b2nd metadata. Cases without a matching
foreground mask, invalid metadata, or incompatible shapes are recorded in the
manifest and skipped.

Mask filenames relate to Dataset625 case IDs differently per cohort — most keep only
the part after ``__``, fastMRI keeps the source prefix, and the ODELIA family keeps the
cohort tag without ``odelia_``. ``case_ids_from_foreground_mask`` covers all of them, so
one pass resolves every case. See that function for the per-cohort forms.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Sequence, Set, Tuple

import blosc2
import nibabel as nib
import numpy as np
from tqdm import tqdm

PRE_SUFFIX = "_pre.b2nd"
FOREGROUND_SUFFIX = "_foreground_mask.b2nd"

MANIFEST_FIELDS = [
    "status",
    "reason",
    "case_id",
    "n_slices",
    "through_axis_3d",
    "pre_shape",
    "foreground_shape_3d",
    "output_shape",
    "spacing_3d",
    "slice_spacing_2d",
    "source_pre_b2nd",
    "source_foreground_mask",
    "output_path",
]

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def strip_nifti_suffix(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        return name[:-7]
    if lower.endswith(".nii"):
        return name[:-4]
    return name


def case_ids_from_foreground_mask(path: Path) -> List[str]:
    """Return the Dataset625 case IDs a foreground mask filename can denote.

    Mask files are named ``<source>__<rest>.nii.gz``, but how much of that name the
    Dataset625 case ID keeps differs per cohort, so every form is offered in priority
    order and the caller keeps whichever one exists in the dataset:

    ``<rest>``
        Most cohorts: ``MAMA-MIA__Breast_MRI_001`` -> ``Breast_MRI_001``.
    ``<source>_<rest>``
        fastMRI, which keeps the source in the case ID:
        ``fastMRI_breast__102`` -> ``fastMRI_breast_102``.
    ``<tag>_<rest>``
        The ODELIA family, which keeps the cohort tag but drops the ``odelia_`` prefix:
        ``odelia_CAM__ODELIA_BRAID1_0158_1`` -> ``CAM_ODELIA_BRAID1_0158_1``.
    ``<stem>``
        Filenames carrying no ``__`` separator.
    """
    stem = strip_nifti_suffix(path.name)
    if "__" not in stem:
        return [stem]
    source, rest = stem.split("__", 1)
    candidates = [rest, f"{source}_{rest}"]
    if source.startswith("odelia_"):
        candidates.append(f"{source[len('odelia_'):]}_{rest}")
    candidates.append(stem)
    return candidates


def discover_pre_b2nds(pre_blosc2_dir: Path) -> Iterator[Tuple[str, Path]]:
    """Yield ``(case_id, pre_b2nd_path)`` deterministically."""
    for path in sorted(pre_blosc2_dir.glob(f"*{PRE_SUFFIX}")):
        case_id = path.name[: -len(PRE_SUFFIX)]
        yield case_id, path


def index_foreground_masks(mask_root: Path, target_case_ids: Set[str]) -> Dict[str, Path]:
    """Index foreground NIfTIs by Dataset625 case ID.

    Raises if two NIfTIs map to the same target case ID. The foreground tree
    can contain unrelated duplicate bare IDs from source-specific datasets, so
    duplicates outside Dataset625's case set are ignored.
    """
    out: Dict[str, Path] = {}
    duplicates: Dict[str, List[Path]] = {}
    for current_root, dirnames, filenames in os.walk(mask_root):
        dirnames[:] = sorted(dirnames)
        for filename in sorted(filenames):
            path = Path(current_root) / filename
            if not is_nifti(path):
                continue
            for case_id in case_ids_from_foreground_mask(path):
                if case_id not in target_case_ids:
                    continue
                previous = out.get(case_id)
                if previous is not None:
                    duplicates.setdefault(case_id, [previous]).append(path)
                else:
                    out[case_id] = path
                break

    if duplicates:
        preview = "; ".join(
            f"{case}: {[str(p) for p in paths[:3]]}"
            for case, paths in sorted(duplicates.items())[:5]
        )
        raise RuntimeError(
            f"Duplicate foreground masks for {len(duplicates)} case IDs: {preview}"
        )
    return out


def load_mask_uint8(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Load a NIfTI mask as uint8 and return ``(data, spacing[:3])``."""
    img = nib.load(str(path))
    data = np.asarray(img.dataobj).astype(np.uint8)
    spacing = tuple(float(v) for v in img.header.get_zooms()[:3])
    return data, spacing  # type: ignore[return-value]


def stack_retained_slices(
    volume: np.ndarray, retained: Sequence[int], axis: int
) -> np.ndarray:
    """Move the through-plane axis first and stack retained 2D slices."""
    moved = np.moveaxis(volume, axis, 0)
    return np.ascontiguousarray(moved[list(retained)])


def validate_metadata(
    pre_nd: object,
) -> Tuple[int, List[int], List[float], List[float], str | None]:
    """Extract and validate Dataset625 metadata required for alignment."""
    vlmeta = pre_nd.vlmeta  # type: ignore[attr-defined]
    missing = [
        key
        for key in (
            "through_axis_3d",
            "retained_indices_3d",
            "spacing_3d",
            "slice_spacing_2d",
        )
        if key not in vlmeta
    ]
    if missing:
        raise ValueError(f"missing vlmeta keys: {missing}")

    axis = int(vlmeta["through_axis_3d"])
    retained = [int(v) for v in vlmeta["retained_indices_3d"]]
    spacing_3d = [float(v) for v in vlmeta["spacing_3d"]]
    slice_spacing_2d = [float(v) for v in vlmeta["slice_spacing_2d"]]
    dataset_source = vlmeta.get("dataset_source")

    if axis not in (0, 1, 2):
        raise ValueError(f"invalid through_axis_3d: {axis}")
    if not retained:
        raise ValueError("empty retained_indices_3d")
    if any(idx < 0 for idx in retained):
        raise ValueError("negative retained_indices_3d value")
    if len(spacing_3d) != 3:
        raise ValueError(f"invalid spacing_3d length: {len(spacing_3d)}")
    if len(slice_spacing_2d) != 2:
        raise ValueError(
            f"invalid slice_spacing_2d length: {len(slice_spacing_2d)}"
        )

    return axis, retained, spacing_3d, slice_spacing_2d, dataset_source


def _write_b2nd(
    arr: np.ndarray,
    urlpath: Path,
    clevel: int,
    vlmeta: Mapping[str, object],
) -> None:
    """Persist ``arr`` as a chunked Blosc2 NDArray with one chunk per slice."""
    if arr.ndim != 3:
        raise ValueError(f"expected (N, H, W) array, got shape {arr.shape}")
    chunks = (1, arr.shape[1], arr.shape[2])
    cparams = {
        "codec": blosc2.Codec.ZSTD,
        "clevel": int(clevel),
        "filters": [blosc2.Filter.NOFILTER],
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


def process_case(
    case_id: str,
    pre_b2nd_path: Path,
    foreground_mask_by_case: Mapping[str, Path],
    output_dir: Path,
    clevel: int,
    overwrite: bool,
    dry_run: bool,
) -> Dict[str, object]:
    output_path = output_dir / f"{case_id}{FOREGROUND_SUFFIX}"
    foreground_path = foreground_mask_by_case.get(case_id)

    common_row: Dict[str, object] = {
        "case_id": case_id,
        "source_pre_b2nd": str(pre_b2nd_path),
        "source_foreground_mask": str(foreground_path) if foreground_path else "",
        "output_path": str(output_path),
    }

    try:
        if foreground_path is None:
            return manifest_row(
                status="missing_foreground_mask",
                reason="no foreground mask matched by case ID",
                n_slices=0,
                **common_row,
            )

        pre_nd = blosc2.open(str(pre_b2nd_path))
        pre_shape = tuple(int(v) for v in pre_nd.shape)
        if len(pre_shape) != 3:
            return manifest_row(
                status="invalid_metadata",
                reason=f"invalid pre b2nd ndim: shape {pre_shape}",
                pre_shape=json.dumps(list(pre_shape)),
                n_slices=0,
                **common_row,
            )

        try:
            axis, retained, spacing_3d, slice_spacing_2d, dataset_source = (
                validate_metadata(pre_nd)
            )
        except Exception as exc:  # noqa: BLE001 - record the case and continue
            return manifest_row(
                status="invalid_metadata",
                reason=f"{type(exc).__name__}: {exc}",
                pre_shape=json.dumps(list(pre_shape)),
                n_slices=0,
                **common_row,
            )

        if output_path.exists() and not overwrite:
            return manifest_row(
                status="skipped_existing",
                reason="",
                n_slices=pre_shape[0],
                through_axis_3d=axis,
                pre_shape=json.dumps(list(pre_shape)),
                output_shape=json.dumps(list(pre_shape)),
                spacing_3d=json.dumps(spacing_3d),
                slice_spacing_2d=json.dumps(slice_spacing_2d),
                **common_row,
            )

        mask, foreground_spacing = load_mask_uint8(foreground_path)
        foreground_shape = tuple(int(v) for v in mask.shape)
        if len(foreground_shape) != 3:
            return manifest_row(
                status="shape_mismatch",
                reason=f"foreground mask is not 3D: shape {foreground_shape}",
                n_slices=0,
                through_axis_3d=axis,
                pre_shape=json.dumps(list(pre_shape)),
                foreground_shape_3d=json.dumps(list(foreground_shape)),
                spacing_3d=json.dumps(spacing_3d),
                slice_spacing_2d=json.dumps(slice_spacing_2d),
                **common_row,
            )

        if max(retained) >= foreground_shape[axis]:
            return manifest_row(
                status="shape_mismatch",
                reason=(
                    f"retained max index {max(retained)} exceeds foreground "
                    f"axis {axis} size {foreground_shape[axis]}"
                ),
                n_slices=0,
                through_axis_3d=axis,
                pre_shape=json.dumps(list(pre_shape)),
                foreground_shape_3d=json.dumps(list(foreground_shape)),
                spacing_3d=json.dumps(spacing_3d),
                slice_spacing_2d=json.dumps(slice_spacing_2d),
                **common_row,
            )

        stack_shape = (
            len(retained),
            *[foreground_shape[i] for i in range(3) if i != axis],
        )
        if stack_shape != pre_shape:
            return manifest_row(
                status="shape_mismatch",
                reason=f"foreground stack {stack_shape} vs pre {pre_shape}",
                n_slices=0,
                through_axis_3d=axis,
                pre_shape=json.dumps(list(pre_shape)),
                foreground_shape_3d=json.dumps(list(foreground_shape)),
                output_shape=json.dumps(list(stack_shape)),
                spacing_3d=json.dumps(spacing_3d),
                slice_spacing_2d=json.dumps(slice_spacing_2d),
                **common_row,
            )

        if not dry_run:
            stack = stack_retained_slices(mask, retained, axis)
            _write_b2nd(
                stack,
                output_path,
                clevel=clevel,
                vlmeta={
                    "case_id": case_id,
                    "source_pre_b2nd": str(pre_b2nd_path),
                    "source_foreground_mask": str(foreground_path),
                    "through_axis_3d": int(axis),
                    "spacing_3d": spacing_3d,
                    "slice_spacing_2d": slice_spacing_2d,
                    "retained_indices_3d": [int(i) for i in retained],
                    "modality": "foreground_mask",
                    "dtype": "uint8",
                    "foreground_spacing_3d": [float(v) for v in foreground_spacing],
                    **(
                        {"dataset_source": dataset_source}
                        if dataset_source is not None
                        else {}
                    ),
                },
            )
            del stack
        del mask

        return manifest_row(
            status="ok",
            reason="",
            n_slices=pre_shape[0],
            through_axis_3d=axis,
            pre_shape=json.dumps(list(pre_shape)),
            foreground_shape_3d=json.dumps(list(foreground_shape)),
            output_shape=json.dumps(list(pre_shape)),
            spacing_3d=json.dumps(spacing_3d),
            slice_spacing_2d=json.dumps(slice_spacing_2d),
            **common_row,
        )
    except Exception as exc:  # noqa: BLE001 - preserve an audit row per input
        return manifest_row(
            status="error",
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
        "--pre_blosc2_dir",
        type=Path,
        required=True,
        help="Dataset625_Pre_Seg directory holding the <case>_pre.b2nd stacks whose "
        "slice axis and retained indices are reused here.",
    )
    parser.add_argument(
        "--foreground_mask_root",
        type=Path,
        required=True,
        help="Root of the 3D foreground-mask NIfTIs, laid out as "
        "<chunk>/<source-prefix>__<case>.nii.gz.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for the Dataset627_foreground .b2nd stacks.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--clevel",
        type=int,
        default=3,
        choices=range(1, 10),
        metavar="1-9",
        help="ZSTD compression level.",
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
        help="Process at most N cases after deterministic sorting/filtering.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.pre_blosc2_dir.is_dir():
        raise SystemExit(f"Pre Blosc2 directory does not exist: {args.pre_blosc2_dir}")
    if not args.foreground_mask_root.is_dir():
        raise SystemExit(
            f"Foreground mask root does not exist: {args.foreground_mask_root}"
        )
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    cases = list(discover_pre_b2nds(args.pre_blosc2_dir))
    if args.cases:
        whitelist = set(args.cases)
        cases = [case for case in cases if case[0] in whitelist]
    if args.limit is not None:
        cases = cases[: args.limit]
    logger.info("Found %d Dataset625 pre cases", len(cases))
    if not cases:
        logger.warning("No cases to process; exiting")
        return

    target_case_ids = {case_id for case_id, _ in cases}
    logger.info("Indexing foreground masks under %s", args.foreground_mask_root)
    foreground_mask_by_case = index_foreground_masks(
        args.foreground_mask_root, target_case_ids
    )
    logger.info("Indexed %d foreground masks for target cases", len(foreground_mask_by_case))

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    max_pending = max(args.workers * 2, 1)
    progress = tqdm(
        total=len(cases),
        desc="Dataset627_foreground",
        unit="case",
        dynamic_ncols=True,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending: Set[Future[Dict[str, object]]] = set()
        for case_id, pre_b2nd_path in cases:
            pending.add(
                executor.submit(
                    process_case,
                    case_id,
                    pre_b2nd_path,
                    foreground_mask_by_case,
                    args.output_dir,
                    args.clevel,
                    args.overwrite,
                    args.dry_run,
                )
            )
            progress.set_postfix(queued=len(pending))
            if len(pending) >= max_pending:
                pending = collect_completed(
                    pending, rows, progress, wait_for_all=False
                )
        pending = collect_completed(pending, rows, progress, wait_for_all=True)
    progress.close()

    rows.sort(key=lambda row: str(row["case_id"]))
    status_counts: Dict[str, int] = {}
    total_slices = 0
    for row in rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        if status in {"ok", "skipped_existing"}:
            total_slices += int(row.get("n_slices") or 0)

    logger.info("Statuses: %s", status_counts)
    logger.info("Total usable foreground slices: %d", total_slices)

    if args.dry_run:
        logger.info("Dry run: manifest and Blosc2 files were not written")
        return

    manifest_path = args.output_dir / "manifest.csv"
    write_manifest(manifest_path, rows)
    logger.info("Wrote manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
