"""Two-model stitched synthesis composited onto the pre-contrast background.

Runs two translation checkpoints plus breast and foreground segmentations, then:

    stitched = breast_soft * lesion_synth + (1 - breast_soft) * image_synth
    final    = foreground_soft * stitched + (1 - foreground_soft) * pre

so one model covers the breast, the other the remaining tissue, and everything
outside the tissue support comes from the real input.

The translation models follow the normalization options of the single-model
script. The segmentation models run as elsewhere here: resize, per-slice z-score,
optional flip averaging, argmax, resize back.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

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
from nnunetv2.inference.predict_mamasynth_translate_stitched import (
    _build_blend_weight,
    load_seg_network,
    predict_breast_mask,
)


def resolve_checkpoint(
    path_or_folder: str | Path,
    checkpoint_name: str | None,
    argument_name: str = "weights",
) -> Path:
    """Accept either a direct checkpoint path or a folder plus a checkpoint name."""
    path = Path(path_or_folder).expanduser()
    if path.is_file():
        if path.suffix != ".pth":
            raise ValueError(f"Expected a .pth checkpoint file, got: {path}")
        return path.resolve()
    if path.is_dir():
        if checkpoint_name is None:
            raise ValueError(
                f"{argument_name} was given as a folder, so the matching "
                "--*-checkpoint-name argument is required."
            )
        checkpoint = path / checkpoint_name
        if checkpoint.is_file():
            return checkpoint.resolve()
        raise FileNotFoundError(
            f"Could not find checkpoint '{checkpoint_name}' in folder: {path}"
        )
    raise FileNotFoundError(f"Checkpoint path or folder does not exist: {path}")


def composite_foreground_stitched_slice(
    pre: np.ndarray,
    lesion_synth: np.ndarray,
    image_synth: np.ndarray,
    breast_mask: np.ndarray,
    foreground_mask: np.ndarray,
    feather: float,
    air_from: str = "pre",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the final image for one native-resolution slice.

    ``air_from`` chooses what fills the region outside the tissue support: the raw
    pre-contrast image, or the synthesis from the model that covers non-breast tissue.
    The region between the tissue boundary and the breast comes from that same model
    either way, so this only affects the empty area around the body.

    Returns the final image, the stitched image before compositing, and both soft
    weights.
    """
    breast_soft = _build_blend_weight(breast_mask, dilate=0, feather=feather)
    foreground_soft = _build_blend_weight(foreground_mask, dilate=0, feather=feather)

    stitched = (
        breast_soft * lesion_synth.astype(np.float32)
        + (1.0 - breast_soft) * image_synth.astype(np.float32)
    ).astype(np.float32)
    if air_from == "image":
        background = image_synth.astype(np.float32)
    elif air_from == "pre":
        background = pre.astype(np.float32)
    else:
        raise ValueError(f"air_from must be 'pre' or 'image', got {air_from!r}")
    final = (foreground_soft * stitched + (1.0 - foreground_soft) * background).astype(
        np.float32
    )
    return (
        final,
        stitched,
        breast_soft.astype(np.float32),
        foreground_soft.astype(np.float32),
    )


def predict_foreground_stitched_array(
    lesion_net: torch.nn.Module,
    image_net: torch.nn.Module,
    breast_net: torch.nn.Module,
    foreground_net: torch.nn.Module,
    array: np.ndarray,
    mean_std_fn,
    divisor: int,
    device: torch.device,
    tta: bool,
    breast_label: int,
    foreground_label: int,
    feather: float,
    return_masks: bool = False,
    return_intermediates: bool = False,
):
    """Composite a 2D or 3D array slice by slice.

    Returns the final image, or a pair of the image and a dictionary of intermediate
    masks and syntheses when debug output is requested.
    """

    def _one(slice_2d: np.ndarray):
        """Synthesise, stitch and composite a single slice."""
        mean, std = mean_std_fn(slice_2d)
        lesion_synth = predict_slice(
            lesion_net, slice_2d, mean, std, divisor, device, tta
        )
        image_synth = predict_slice(
            image_net, slice_2d, mean, std, divisor, device, tta
        )
        breast_mask = predict_breast_mask(
            breast_net, slice_2d, device, breast_label=breast_label, tta=tta
        )
        foreground_mask = predict_breast_mask(
            foreground_net,
            slice_2d,
            device,
            breast_label=foreground_label,
            tta=tta,
        )
        final, stitched, breast_soft, foreground_soft = (
            composite_foreground_stitched_slice(
                slice_2d,
                lesion_synth,
                image_synth,
                breast_mask,
                foreground_mask,
                feather,
            )
        )
        extras = {}
        if return_masks:
            extras.update(
                {
                    "breast_mask": breast_mask.astype(np.uint8),
                    "foreground_mask": foreground_mask.astype(np.uint8),
                    "breast_soft": breast_soft.astype(np.float32),
                    "foreground_soft": foreground_soft.astype(np.float32),
                }
            )
        if return_intermediates:
            extras.update(
                {
                    "lesion_synth": lesion_synth.astype(np.float32),
                    "image_synth": image_synth.astype(np.float32),
                    "stitched": stitched.astype(np.float32),
                }
            )
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


def _device_from_arg(device_arg: str) -> torch.device:
    """Resolve the device name into a torch device, setting thread or backend options."""
    if device_arg == "cuda":
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        return device
    if device_arg == "mps":
        return torch.device("mps")
    torch.set_num_threads(os.cpu_count() or 1)
    return torch.device("cpu")


def _write_extras(
    extras: dict[str, np.ndarray],
    output_dir: Path,
    input_path: Path,
    kind: str,
    like_itk,
) -> None:
    """Write the requested intermediate masks and images alongside the output."""
    for key, value in extras.items():
        out_name = f"{input_path.stem}_{key}{input_path.suffix}"
        write_image(value.astype(np.float32), output_dir / out_name, kind, like_itk)


def main():
    """Parse arguments, load every network, and composite each file in the input folder."""
    parser = argparse.ArgumentParser(
        description="Run two MamaSynth translation checkpoints, stitch them with a breast mask, "
        "then restore pre-contrast background outside a foreground/support mask.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("-o", "--output", required=True, type=str)
    parser.add_argument(
        "--image-weights",
        required=True,
        type=str,
        help="Translation checkpoint or fold folder for the non-breast foreground/image region.",
    )
    parser.add_argument(
        "--lesion-image-weights",
        required=True,
        type=str,
        help="Translation checkpoint or fold folder for the breast/lesion image region.",
    )
    parser.add_argument(
        "--foreground-weights",
        required=True,
        type=str,
        help="Foreground segmentation checkpoint or fold folder.",
    )
    parser.add_argument(
        "--breast-weights",
        required=True,
        type=str,
        help="Breast segmentation checkpoint or fold folder.",
    )
    parser.add_argument(
        "--image-checkpoint-name",
        default=None,
        type=str,
        help="Checkpoint filename when --image-weights is a folder.",
    )
    parser.add_argument(
        "--lesion-image-checkpoint-name",
        default=None,
        type=str,
        help="Checkpoint filename when --lesion-image-weights is a folder.",
    )
    parser.add_argument(
        "--foreground-checkpoint-name",
        default=None,
        type=str,
        help="Checkpoint filename when --foreground-weights is a folder.",
    )
    parser.add_argument(
        "--breast-checkpoint-name",
        default=None,
        type=str,
        help="Checkpoint filename when --breast-weights is a folder.",
    )
    parser.add_argument("--breast-label", type=int, default=1)
    parser.add_argument("--foreground-label", type=int, default=1)
    parser.add_argument(
        "--feather",
        type=float,
        default=4.0,
        help="Soft-blend both breast stitch and foreground/pre background transition over ~N pixels.",
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
        help="Feed raw synthesis input values as-is.",
    )
    parser.add_argument("--plans-image", type=str, default=None)
    parser.add_argument("--plans-lesion-image", type=str, default=None)
    parser.add_argument("--plans-foreground", type=str, default=None)
    parser.add_argument("--plans-breast", type=str, default=None)
    parser.add_argument("--dataset-json-image", type=str, default=None)
    parser.add_argument("--dataset-json-lesion-image", type=str, default=None)
    parser.add_argument("--dataset-json-foreground", type=str, default=None)
    parser.add_argument("--dataset-json-breast", type=str, default=None)
    parser.add_argument("-device", type=str, default="cuda", help="cuda | cpu | mps.")
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()

    if args.dont_norm and args.pre_stats is not None:
        parser.error("--pre-stats and --dont-norm are mutually exclusive.")

    input_dir = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser()
    if not input_dir.is_dir():
        parser.error(f"Input folder does not exist: {input_dir}")

    image_checkpoint = resolve_checkpoint(
        args.image_weights, args.image_checkpoint_name, "--image-weights"
    )
    lesion_checkpoint = resolve_checkpoint(
        args.lesion_image_weights,
        args.lesion_image_checkpoint_name,
        "--lesion-image-weights",
    )
    foreground_checkpoint = resolve_checkpoint(
        args.foreground_weights,
        args.foreground_checkpoint_name,
        "--foreground-weights",
    )
    breast_checkpoint = resolve_checkpoint(
        args.breast_weights, args.breast_checkpoint_name, "--breast-weights"
    )

    device = _device_from_arg(args.device)
    image_net, divisor_image = load_translation_network(
        str(image_checkpoint), device, args.plans_image, args.dataset_json_image
    )
    lesion_net, divisor_lesion = load_translation_network(
        str(lesion_checkpoint),
        device,
        args.plans_lesion_image,
        args.dataset_json_lesion_image,
    )
    foreground_net, foreground_classes = load_seg_network(
        str(foreground_checkpoint),
        device,
        args.plans_foreground,
        args.dataset_json_foreground,
    )
    breast_net, breast_classes = load_seg_network(
        str(breast_checkpoint), device, args.plans_breast, args.dataset_json_breast
    )
    if args.foreground_label >= foreground_classes:
        parser.error(
            f"--foreground-label {args.foreground_label} is out of range for a "
            f"{foreground_classes}-class foreground net."
        )
    if args.breast_label >= breast_classes:
        parser.error(
            f"--breast-label {args.breast_label} is out of range for a "
            f"{breast_classes}-class breast net."
        )

    divisor = max(divisor_image, divisor_lesion)
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
    print(f"image        : {image_net.__class__.__name__} ({image_checkpoint.name})")
    print(f"lesion image : {lesion_net.__class__.__name__} ({lesion_checkpoint.name})")
    print(
        f"foreground   : {foreground_net.__class__.__name__} "
        f"({foreground_classes} classes, label {args.foreground_label})"
    )
    print(
        f"breast       : {breast_net.__class__.__name__} "
        f"({breast_classes} classes, label {args.breast_label})"
    )
    print(
        f"divisor {divisor} | TTA {tta} | synth norm: {norm_mode} | feather {args.feather}"
    )
    print(f"Predicting {len(files)} file(s) from {input_dir} -> {output_dir}")

    for index, path in enumerate(files, 1):
        array, kind, like_itk = read_image(path)
        result = predict_foreground_stitched_array(
            lesion_net,
            image_net,
            breast_net,
            foreground_net,
            array,
            mean_std_fn,
            divisor,
            device,
            tta,
            args.breast_label,
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
