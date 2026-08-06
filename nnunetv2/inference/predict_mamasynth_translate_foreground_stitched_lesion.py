"""Foreground-stitched synthesis with a final lesion step.

This is the foreground-stitched pipeline plus a lesion enhancement. Per slice,
every network optionally averaged over flips:

    stitched = breast_soft * lesion_synth + (1 - breast_soft) * image_synth
    final    = foreground_soft * stitched + (1 - foreground_soft) * pre

after which a lesion segmentation is run and the in-breast lesion region of the
result is scaled up by ``--lesion-gain``. Multiplying the intensity already there
keeps the lesion's internal texture, which setting the region to a single value
would flatten.

The two translation models and the breast and foreground segmenters are exactly as
in the foreground-stitched script. The lesion network is an ordinary segmentation
run, executed like the breast network and returning a lesion probability map.

The lesion threshold is lowered in steps until at least one in-breast voxel
passes, falling back to the single highest-probability in-breast voxel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
    composite_foreground_stitched_slice,
    resolve_checkpoint,
)
from nnunetv2.inference.predict_mamasynth_translate_stitched import (
    load_seg_network,
    predict_breast_mask,
)
from nnunetv2.inference.predict_mamasynth_translate_stitched_lesion import (
    enhance_lesion_slice,
    predict_lesion_prob,
)


def predict_foreground_stitched_lesion_array(
    lesion_synth_net: torch.nn.Module,
    image_net: torch.nn.Module,
    breast_net: torch.nn.Module,
    foreground_net: torch.nn.Module,
    lesion_seg_net: torch.nn.Module,
    array: np.ndarray,
    mean_std_fn,
    divisor: int,
    device: torch.device,
    tta: bool,
    breast_label: int,
    foreground_label: int,
    lesion_label: int,
    feather: float,
    lesion_threshold: float,
    lesion_step: float,
    lesion_gain: float,
    lesion_feather: float,
    return_masks: bool = False,
    return_intermediates: bool = False,
    return_lesion_mask: bool = False,
):
    """Composite and enhance a 2D or 3D array slice by slice.

    Returns the final image, or a pair of the image and a dictionary of intermediate
    masks, syntheses and the lesion mask when debug output is requested.
    """

    def _one(slice_2d: np.ndarray):
        """Synthesise, stitch, composite and enhance a single slice."""
        mean, std = mean_std_fn(slice_2d)
        lesion_synth = predict_slice(
            lesion_synth_net, slice_2d, mean, std, divisor, device, tta
        )
        image_synth = predict_slice(
            image_net, slice_2d, mean, std, divisor, device, tta
        )
        breast_mask = predict_breast_mask(
            breast_net, slice_2d, device, breast_label=breast_label, tta=tta
        )
        foreground_mask = predict_breast_mask(
            foreground_net, slice_2d, device, breast_label=foreground_label, tta=tta
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
        lesion_prob = predict_lesion_prob(
            lesion_seg_net, slice_2d, device, lesion_label, tta
        )
        final, lesion_mask = enhance_lesion_slice(
            final,
            breast_mask,
            lesion_prob,
            lesion_threshold,
            lesion_step,
            lesion_gain,
            lesion_feather,
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
                    "lesion_prob": lesion_prob.astype(np.float32),
                }
            )
        if return_lesion_mask:
            extras["lesion_mask"] = lesion_mask.astype(np.uint8)
        return final, extras

    want_extras = return_masks or return_intermediates or return_lesion_mask
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
    """Parse arguments, load every network, and process each file in the input folder."""
    parser = argparse.ArgumentParser(
        description="Foreground-aware stitched MamaSynth synthesis (restores pre-contrast background "
        "outside the foreground mask) with a final lesion step: the in-breast lesion region from a "
        "lesion segmentation net is scaled by --lesion-gain, which preserves its texture.",
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
        help="Translation checkpoint or fold folder for the breast/lesion image region (inside breast).",
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
        "--lesion-seg-weights",
        required=True,
        type=str,
        help="Lesion segmentation checkpoint or fold folder (e.g. "
        "nnUNetTrainerMamaSynthLesionTverskyA02B08_BS_64_epoch_1000__mamaSynthPlans__2d).",
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
    parser.add_argument(
        "--lesion-seg-checkpoint-name",
        default=None,
        type=str,
        help="Checkpoint filename when --lesion-seg-weights is a folder.",
    )
    parser.add_argument("--breast-label", type=int, default=1)
    parser.add_argument("--foreground-label", type=int, default=1)
    parser.add_argument(
        "--lesion-label",
        type=int,
        default=1,
        help="Softmax channel index that denotes 'lesion' in the lesion segmentation net.",
    )
    parser.add_argument(
        "--lesion-threshold",
        type=float,
        default=0.5,
        help="Default lesion probability cutoff (lowered per slice until any in-breast voxel passes).",
    )
    parser.add_argument(
        "--lesion-step",
        type=float,
        default=0.05,
        help="Decrement used to lower the lesion threshold.",
    )
    parser.add_argument(
        "--lesion-gain",
        type=float,
        default=1.25,
        help="Multiplicative intensity boost of the lesion region (out = in * (1 + (gain-1)*soft)); "
        "scaling the existing intensity preserves the lesion's texture. <=1 disables.",
    )
    parser.add_argument(
        "--lesion-feather",
        type=float,
        default=2.0,
        help="Gaussian feather (sigma, px) applied to the lesion soft mask edge (0 = hard mask edge).",
    )
    parser.add_argument(
        "--feather",
        type=float,
        default=4.0,
        help="Soft-blend both breast stitch and foreground/pre background transition over ~N pixels.",
    )
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--save-intermediates", action="store_true")
    parser.add_argument(
        "--save-lesion-mask",
        action="store_true",
        help="Also write the lesion mask next to each output as '<name>_lesion_mask.<ext>'.",
    )
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
    parser.add_argument("--plans-lesion-seg", type=str, default=None)
    parser.add_argument("--dataset-json-image", type=str, default=None)
    parser.add_argument("--dataset-json-lesion-image", type=str, default=None)
    parser.add_argument("--dataset-json-foreground", type=str, default=None)
    parser.add_argument("--dataset-json-breast", type=str, default=None)
    parser.add_argument("--dataset-json-lesion-seg", type=str, default=None)
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
    lesion_image_checkpoint = resolve_checkpoint(
        args.lesion_image_weights,
        args.lesion_image_checkpoint_name,
        "--lesion-image-weights",
    )
    foreground_checkpoint = resolve_checkpoint(
        args.foreground_weights, args.foreground_checkpoint_name, "--foreground-weights"
    )
    breast_checkpoint = resolve_checkpoint(
        args.breast_weights, args.breast_checkpoint_name, "--breast-weights"
    )
    lesion_seg_checkpoint = resolve_checkpoint(
        args.lesion_seg_weights, args.lesion_seg_checkpoint_name, "--lesion-seg-weights"
    )

    device = _device_from_arg(args.device)
    image_net, divisor_image = load_translation_network(
        str(image_checkpoint), device, args.plans_image, args.dataset_json_image
    )
    lesion_synth_net, divisor_lesion = load_translation_network(
        str(lesion_image_checkpoint),
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
    lesion_seg_net, lesion_classes = load_seg_network(
        str(lesion_seg_checkpoint),
        device,
        args.plans_lesion_seg,
        args.dataset_json_lesion_seg,
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
    if args.lesion_label >= lesion_classes:
        parser.error(
            f"--lesion-label {args.lesion_label} is out of range for a "
            f"{lesion_classes}-class lesion net."
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
    print(
        f"lesion image : {lesion_synth_net.__class__.__name__} ({lesion_image_checkpoint.name})"
    )
    print(
        f"foreground   : {foreground_net.__class__.__name__} "
        f"({foreground_classes} classes, label {args.foreground_label})"
    )
    print(
        f"breast       : {breast_net.__class__.__name__} "
        f"({breast_classes} classes, label {args.breast_label})"
    )
    print(
        f"lesion seg   : {lesion_seg_net.__class__.__name__} "
        f"({lesion_classes} classes, label {args.lesion_label})"
    )
    print(
        f"divisor {divisor} | TTA {tta} | synth norm: {norm_mode} | feather {args.feather} | "
        f"lesion gain {args.lesion_gain} thr {args.lesion_threshold} step {args.lesion_step} feather {args.lesion_feather}"
    )
    print(f"Predicting {len(files)} file(s) from {input_dir} -> {output_dir}")

    for index, path in enumerate(files, 1):
        array, kind, like_itk = read_image(path)
        result = predict_foreground_stitched_lesion_array(
            lesion_synth_net,
            image_net,
            breast_net,
            foreground_net,
            lesion_seg_net,
            array,
            mean_std_fn,
            divisor,
            device,
            tta,
            args.breast_label,
            args.foreground_label,
            args.lesion_label,
            args.feather,
            args.lesion_threshold,
            args.lesion_step,
            args.lesion_gain,
            args.lesion_feather,
            return_masks=args.save_masks,
            return_intermediates=args.save_intermediates,
            return_lesion_mask=args.save_lesion_mask,
        )
        if args.save_masks or args.save_intermediates or args.save_lesion_mask:
            prediction, extras = result
        else:
            prediction, extras = result, {}
        write_image(prediction, output_dir / path.name, kind, like_itk)
        _write_extras(extras, output_dir, path, kind, like_itk)
        print(f"  [{index}/{len(files)}] {path.name}  shape={tuple(array.shape)}")

    print("Done.")


if __name__ == "__main__":
    main()
