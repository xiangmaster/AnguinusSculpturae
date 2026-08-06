"""Precompute lesion masks for a folder of slices.

Runs a lesion segmentation checkpoint over every slice in a folder and writes one
binary mask per case, following exactly the path the compositing predictors use:
resize, per-slice z-score, optional flip averaging, argmax, then resize the binary
mask back to the native size with nearest-neighbour interpolation.

Computing these once lets downstream work that needs the lesion region iterate
without re-running the segmenter each time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from nnunetv2.inference.predict_mamasynth_translate import read_image, write_image
from nnunetv2.inference.predict_mamasynth_translate_stitched import (
    load_seg_network,
    predict_breast_mask,
)


def _device_from_arg(device_arg: str) -> torch.device:
    """Resolve the device name into a torch device, setting thread or backend options."""
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def _predict_mask(
    network: torch.nn.Module,
    array: np.ndarray,
    device: torch.device,
    lesion_label: int,
    tta: bool,
) -> np.ndarray:
    """Binary lesion mask at native resolution for one slice."""
    if array.ndim == 2:
        return predict_breast_mask(
            network, array, device, breast_label=lesion_label, tta=tta
        ).astype(np.float32)
    if array.ndim == 3:
        slices = [
            predict_breast_mask(
                network, slice_2d, device, breast_label=lesion_label, tta=tta
            )
            for slice_2d in array
        ]
        return np.stack(slices, axis=0).astype(np.float32)
    raise ValueError(f"Expected a 2D or 3D array, got shape {array.shape}.")


def main():
    """Parse arguments, load the segmenter, and write a mask for every input file."""
    parser = argparse.ArgumentParser(
        description="Predict and save lesion-ROI masks for a folder of MamaSynth slices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("-o", "--output", required=True, type=str)
    parser.add_argument(
        "--weights",
        required=True,
        type=str,
        help="Lesion segmentation checkpoint (.pth) or fold folder (with --checkpoint-name).",
    )
    parser.add_argument(
        "--checkpoint-name",
        default=None,
        type=str,
        help="Checkpoint filename when --weights is a folder.",
    )
    parser.add_argument("--lesion-label", type=int, default=1)
    parser.add_argument("--plans", type=str, default=None)
    parser.add_argument("--dataset-json", type=str, default=None)
    parser.add_argument("-device", type=str, default="cuda", help="cuda | cpu | mps.")
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()

    weights = Path(args.weights).expanduser()
    if weights.is_dir():
        if args.checkpoint_name is None:
            parser.error("--checkpoint-name is required when --weights is a folder.")
        checkpoint = weights / args.checkpoint_name
    else:
        checkpoint = weights
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")

    input_dir = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser()
    if not input_dir.is_dir():
        parser.error(f"Input folder does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _device_from_arg(args.device)
    network, num_classes = load_seg_network(
        str(checkpoint), device, args.plans, args.dataset_json
    )
    tta = not args.no_tta

    suffixes = (".mha", ".tif", ".tiff")
    paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in suffixes)
    if not paths:
        parser.error(f"No .mha/.tif/.tiff files found in {input_dir}")

    print(
        f"Lesion seg: {checkpoint.name} | classes {num_classes} | label {args.lesion_label} "
        f"| TTA {tta} | {len(paths)} cases -> {output_dir}"
    )
    for index, path in enumerate(paths, start=1):
        array, kind, like_itk = read_image(path)
        mask = _predict_mask(network, array, device, args.lesion_label, tta)
        write_image(mask, output_dir / path.name, kind, like_itk)
        if index % 25 == 0 or index == len(paths):
            print(f"  {index}/{len(paths)}")
    print("Done.")


if __name__ == "__main__":
    main()
