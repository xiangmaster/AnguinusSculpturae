"""Synthesis composited onto the real pre-contrast background outside the tissue.

The foreground-loss trainers only optimise the synthetic image inside the tissue
support; everything outside is meant to remain the untouched pre-contrast image.
This script reproduces that arrangement:

    foreground_mask = foreground_segmenter(pre)
    foreground_soft = feathered version of that mask
    synth           = translation_net(pre)
    final           = foreground_soft * synth + (1 - foreground_soft) * pre

which is the same composite those trainers were scored against. It is the
foreground-stitched pipeline without the inside/outside-breast stitch, so a single
translation model is enough.

The translation model follows the normalization options of the single-model
script. The foreground segmenter runs like the other segmentation models here:
resize, per-slice z-score, optional flip averaging, argmax, resize back.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

from nnunetv2.inference.predict_mamasynth_translate import (
    _gather_inputs,
    _make_mean_std_fn,
    load_translation_network,
    predict_slice,
    read_image,
    write_image,
)
from nnunetv2.inference.predict_mamasynth_translate_foreground_stitched import (
    _device_from_arg,
    _write_extras,
    resolve_checkpoint,
)
from nnunetv2.inference.predict_mamasynth_translate_stitched import (
    _build_blend_weight,
    load_seg_network,
    predict_breast_mask,
)


def composite_foreground_slice(
    pre: np.ndarray,
    synth: np.ndarray,
    foreground_mask: np.ndarray,
    feather: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Feather the synthetic foreground onto the pre-contrast background.

    Returns the composite together with the soft weight used to build it.
    """
    foreground_soft = _build_blend_weight(foreground_mask, dilate=0, feather=feather)
    final = (
        foreground_soft * synth.astype(np.float32)
        + (1.0 - foreground_soft) * pre.astype(np.float32)
    ).astype(np.float32)
    return final, foreground_soft.astype(np.float32)


def predict_foreground_array(
    translation_net: torch.nn.Module,
    foreground_net: torch.nn.Module,
    array: np.ndarray,
    mean_std_fn,
    divisor: int,
    device: torch.device,
    tta: bool,
    foreground_label: int,
    feather: float,
    return_masks: bool = False,
    return_intermediates: bool = False,
):
    """Composite a 2D or 3D array slice by slice onto its pre-contrast background."""

    def _one(slice_2d: np.ndarray):
        """Synthesise and composite a single slice."""
        mean, std = mean_std_fn(slice_2d)
        synth = predict_slice(
            translation_net, slice_2d, mean, std, divisor, device, tta
        )
        foreground_mask = predict_breast_mask(
            foreground_net, slice_2d, device, breast_label=foreground_label, tta=tta
        )
        final, foreground_soft = composite_foreground_slice(
            slice_2d, synth, foreground_mask, feather
        )
        extras = {}
        if return_masks:
            extras["foreground_mask"] = foreground_mask.astype(np.uint8)
            extras["foreground_soft"] = foreground_soft.astype(np.float32)
        if return_intermediates:
            extras["synth"] = synth.astype(np.float32)
        return final, extras

    want_extras = return_masks or return_intermediates
    if array.ndim == 2:
        final, extras = _one(array)
        return (final, extras) if want_extras else final
    if array.ndim == 3:
        final = np.empty_like(array, dtype=np.float32)
        extras_out = {} if want_extras else None
        for index in range(array.shape[0]):
            final[index], extras = _one(array[index])
            if want_extras:
                for key, value in extras.items():
                    if key not in extras_out:
                        extras_out[key] = np.empty(array.shape, dtype=value.dtype)
                    extras_out[key][index] = value
        return (final, extras_out) if want_extras else final
    raise ValueError(
        f"Unsupported array shape {array.shape}; expected 2D (H,W) or 3D (n,H,W)."
    )


def main():
    """Parse arguments, load both networks, and composite every file in the input folder."""
    parser = argparse.ArgumentParser(
        description="Run a MamaSynth ForegroundLoss translation checkpoint, then restore the "
        "pre-contrast background outside a foreground/support mask (feathered composite).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("-o", "--output", required=True, type=str)
    parser.add_argument(
        "--weights",
        required=True,
        type=str,
        help="Translation checkpoint (.pth) or fold folder for the ForegroundLoss model.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default=None,
        type=str,
        help="Checkpoint filename when --weights is a folder (e.g. checkpoint_best.pth).",
    )
    parser.add_argument(
        "--foreground-weights",
        required=True,
        type=str,
        help="Foreground segmentation checkpoint (.pth) or fold folder.",
    )
    parser.add_argument(
        "--foreground-checkpoint-name",
        default=None,
        type=str,
        help="Checkpoint filename when --foreground-weights is a folder.",
    )
    parser.add_argument("--foreground-label", type=int, default=1)
    parser.add_argument(
        "--feather",
        type=float,
        default=4.0,
        help="Soft-blend the foreground/pre background transition over ~N pixels "
        "(matches the trainer's composite_feather4 LPIPS composite).",
    )
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--save-intermediates", action="store_true")
    parser.add_argument(
        "--pre-stats",
        type=str,
        default=None,
        help="JSON with global {'mean','std'} for synthesis input z-scoring.",
    )
    parser.add_argument(
        "--dont-norm",
        action="store_true",
        help="Feed raw synthesis input values as-is (mean=0, std=1).",
    )
    parser.add_argument("--plans", type=str, default=None)
    parser.add_argument("--plans-foreground", type=str, default=None)
    parser.add_argument("--dataset-json", type=str, default=None)
    parser.add_argument("--dataset-json-foreground", type=str, default=None)
    parser.add_argument("-device", type=str, default="cuda", help="cuda | cpu | mps.")
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()

    if args.dont_norm and args.pre_stats is not None:
        parser.error("--pre-stats and --dont-norm are mutually exclusive.")

    input_dir = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser()
    if not input_dir.is_dir():
        parser.error(f"Input folder does not exist: {input_dir}")

    translation_checkpoint = resolve_checkpoint(
        args.weights, args.checkpoint_name, "--weights"
    )
    foreground_checkpoint = resolve_checkpoint(
        args.foreground_weights, args.foreground_checkpoint_name, "--foreground-weights"
    )

    device = _device_from_arg(args.device)
    translation_net, divisor = load_translation_network(
        str(translation_checkpoint), device, args.plans, args.dataset_json
    )
    foreground_net, foreground_classes = load_seg_network(
        str(foreground_checkpoint),
        device,
        args.plans_foreground,
        args.dataset_json_foreground,
    )
    if args.foreground_label >= foreground_classes:
        parser.error(
            f"--foreground-label {args.foreground_label} is out of range for a "
            f"{foreground_classes}-class foreground net."
        )

    mean_std_fn = _make_mean_std_fn(args.pre_stats, args.dont_norm)
    tta = not args.no_tta
    files = _gather_inputs(input_dir)
    if not files:
        parser.error(f"No .mha/.tif/.tiff files found in {input_dir}")

    maybe_mkdir_p(str(output_dir))
    norm_mode = (
        "raw (no norm)"
        if args.dont_norm
        else ("global pre-stats" if args.pre_stats else "per-slice z-score")
    )
    print(
        f"translation : {translation_net.__class__.__name__} ({translation_checkpoint.name})"
    )
    print(
        f"foreground  : {foreground_net.__class__.__name__} "
        f"({foreground_classes} classes, label {args.foreground_label})"
    )
    print(
        f"divisor {divisor} | TTA {tta} | synth norm: {norm_mode} | feather {args.feather}"
    )
    print(f"Predicting {len(files)} file(s) from {input_dir} -> {output_dir}")

    for index, path in enumerate(files, 1):
        array, kind, like_itk = read_image(path)
        result = predict_foreground_array(
            translation_net,
            foreground_net,
            array,
            mean_std_fn,
            divisor,
            device,
            tta,
            args.foreground_label,
            args.feather,
            return_masks=args.save_masks,
            return_intermediates=args.save_intermediates,
        )
        if args.save_masks or args.save_intermediates:
            prediction, extras = result
        else:
            prediction, extras = result, {}
        write_image(prediction, output_dir / path.name, kind, like_itk)
        _write_extras(extras, output_dir, path, kind, like_itk)
        print(f"  [{index}/{len(files)}] {path.name}  shape={tuple(array.shape)}")

    print("Done.")


if __name__ == "__main__":
    main()
