"""Foreground-stitched synthesis with the translation models ensembled over folds too.

Where the fold-ensembled variant averages only the lesion segmentation, here both
translation models are also averaged over folds, their per-slice outputs combined
before compositing:

  - the model covering tissue outside the breast: run directory, folds, best checkpoint
  - the model covering the breast: run directory, folds, best-by-segmentation checkpoint
  - the lesion segmenter: run directory, folds, best checkpoint

The breast and foreground segmentations stay single checkpoints. Compositing and
the lesion step are unchanged:

    image_synth  = mean over folds of the outside model
    lesion_synth = mean over folds of the breast model
    stitched     = breast_soft * lesion_synth + (1 - breast_soft) * image_synth
    final        = foreground_soft * stitched + (1 - foreground_soft) * pre
    final        = lesion enhancement of that

Averaging the synthetic slices is equivalent to averaging the predicted deltas,
since all folds share the same input normalization for a given slice.
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


def _predict_synth_ensemble(nets, slice_2d, mean, std, divisor, device, tta):
    """Mean of the synthetic slices produced by each fold."""
    acc = None
    for net in nets:
        out = predict_slice(net, slice_2d, mean, std, divisor, device, tta)
        acc = out if acc is None else acc + out
    return (acc / float(len(nets))).astype(np.float32)


def predict_array(
    lesion_synth_nets,
    image_nets,
    breast_net,
    foreground_net,
    lesion_seg_nets,
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
    air_from: str = "pre",
    return_masks: bool = False,
    return_intermediates: bool = False,
    return_lesion_mask: bool = False,
):
    """Composite a 2D or 3D array slice by slice using the ensembled translations."""

    def _one(slice_2d: np.ndarray):
        """Synthesise with every fold, stitch, composite and enhance a single slice."""
        mean, std = mean_std_fn(slice_2d)
        lesion_synth = _predict_synth_ensemble(
            lesion_synth_nets, slice_2d, mean, std, divisor, device, tta
        )
        image_synth = _predict_synth_ensemble(
            image_nets, slice_2d, mean, std, divisor, device, tta
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
                air_from,
            )
        )
        lesion_prob = predict_lesion_prob(
            lesion_seg_nets, slice_2d, device, lesion_label, tta
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


def _resolve_fold_ckpts(run_dir, folds, ckpt_name, parser, what):
    """Expand a run directory and a list of folds into concrete checkpoint paths."""
    run_dir = Path(run_dir).expanduser()
    out = []
    for fold in folds:
        ckpt = run_dir / f"fold_{fold}" / ckpt_name
        if not ckpt.is_file():
            parser.error(f"{what} fold checkpoint not found: {ckpt}")
        out.append(ckpt)
    return out


def main():
    """Parse arguments, load all fold ensembles, and process each file in the input folder."""
    parser = argparse.ArgumentParser(
        description="Foreground-stitched synthesis with lesion gain where BOTH translation models "
        "(outside-breast image net, inside-breast lesion-image net) AND the lesion segmentation are "
        "fold-ensembled (averaged outputs). Same compositing/gain as the _lesion_folds variant.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("-o", "--output", required=True, type=str)
    parser.add_argument(
        "--image-dir",
        required=True,
        type=str,
        help="OUTSIDE-breast translation RUN directory (contains fold_X/ + plans.json).",
    )
    parser.add_argument("--image-folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--image-checkpoint-name", default="checkpoint_best.pth", type=str
    )
    parser.add_argument(
        "--lesion-image-dir",
        required=True,
        type=str,
        help="INSIDE-breast translation RUN directory (contains fold_X/ + plans.json).",
    )
    parser.add_argument(
        "--lesion-image-folds", type=int, nargs="+", default=[0, 1, 2, 3, 4]
    )
    parser.add_argument(
        "--lesion-image-checkpoint-name", default="checkpoint_best_dice.pth", type=str
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
    parser.add_argument("--foreground-checkpoint-name", default=None, type=str)
    parser.add_argument("--breast-checkpoint-name", default=None, type=str)
    parser.add_argument(
        "--lesion-seg-dir",
        required=True,
        type=str,
        help="Lesion segmentation RUN directory (contains fold_X/ + plans.json).",
    )
    parser.add_argument("--lesion-seg-folds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument(
        "--lesion-seg-checkpoint-name", default="checkpoint_best.pth", type=str
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
    parser.add_argument("--breast-label", type=int, default=1)
    parser.add_argument("--foreground-label", type=int, default=1)
    parser.add_argument("--lesion-label", type=int, default=1)
    parser.add_argument("--lesion-threshold", type=float, default=0.5)
    parser.add_argument("--lesion-step", type=float, default=0.05)
    parser.add_argument(
        "--lesion-gain",
        type=float,
        default=1.25,
        help="Multiplicative intensity boost of the lesion region (<=1 disables).",
    )
    parser.add_argument(
        "--lesion-feather",
        type=float,
        default=2.0,
        help="Gaussian feather (sigma, px) on the lesion soft-mask edge.",
    )
    parser.add_argument("--feather", type=float, default=4.0)
    parser.add_argument(
        "--air-from",
        choices=["pre", "image"],
        default="pre",
        help="Out-of-foreground (air) fill: 'pre' (raw pre image, default) or 'image' "
        "(ensembled image-net synthesis).",
    )
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--save-intermediates", action="store_true")
    parser.add_argument("--save-lesion-mask", action="store_true")
    parser.add_argument("--pre-stats", type=str, default=None)
    parser.add_argument("--dont-norm", action="store_true")
    parser.add_argument("-device", type=str, default="cuda", help="cuda | cpu | mps.")
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()

    if args.dont_norm and args.pre_stats is not None:
        parser.error("--pre-stats and --dont-norm are mutually exclusive.")

    input_dir = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser()
    if not input_dir.is_dir():
        parser.error(f"Input folder does not exist: {input_dir}")

    image_ckpts = _resolve_fold_ckpts(
        args.image_dir,
        args.image_folds,
        args.image_checkpoint_name,
        parser,
        "--image-dir",
    )
    lesion_image_ckpts = _resolve_fold_ckpts(
        args.lesion_image_dir,
        args.lesion_image_folds,
        args.lesion_image_checkpoint_name,
        parser,
        "--lesion-image-dir",
    )
    lesion_seg_ckpts = _resolve_fold_ckpts(
        args.lesion_seg_dir,
        args.lesion_seg_folds,
        args.lesion_seg_checkpoint_name,
        parser,
        "--lesion-seg-dir",
    )
    foreground_checkpoint = resolve_checkpoint(
        args.foreground_weights, args.foreground_checkpoint_name, "--foreground-weights"
    )
    breast_checkpoint = resolve_checkpoint(
        args.breast_weights, args.breast_checkpoint_name, "--breast-weights"
    )

    device = _device_from_arg(args.device)

    image_nets = []
    divisor_image = None
    for ckpt in image_ckpts:
        net, divisor_image = load_translation_network(
            str(ckpt), device, args.plans_image, args.dataset_json_image
        )
        image_nets.append(net)
    lesion_synth_nets = []
    divisor_lesion = None
    for ckpt in lesion_image_ckpts:
        net, divisor_lesion = load_translation_network(
            str(ckpt), device, args.plans_lesion_image, args.dataset_json_lesion_image
        )
        lesion_synth_nets.append(net)

    foreground_net, foreground_classes = load_seg_network(
        str(foreground_checkpoint),
        device,
        args.plans_foreground,
        args.dataset_json_foreground,
    )
    breast_net, breast_classes = load_seg_network(
        str(breast_checkpoint), device, args.plans_breast, args.dataset_json_breast
    )
    lesion_seg_nets = []
    lesion_classes = None
    for ckpt in lesion_seg_ckpts:
        net, lesion_classes = load_seg_network(
            str(ckpt), device, args.plans_lesion_seg, args.dataset_json_lesion_seg
        )
        lesion_seg_nets.append(net)

    if args.foreground_label >= foreground_classes:
        parser.error(
            f"--foreground-label {args.foreground_label} out of range for {foreground_classes}-class foreground net."
        )
    if args.breast_label >= breast_classes:
        parser.error(
            f"--breast-label {args.breast_label} out of range for {breast_classes}-class breast net."
        )
    if args.lesion_label >= lesion_classes:
        parser.error(
            f"--lesion-label {args.lesion_label} out of range for {lesion_classes}-class lesion net."
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
    print(
        f"image (outside): {image_nets[0].__class__.__name__}  ENSEMBLE folds {args.image_folds} "
        f"({args.image_checkpoint_name})"
    )
    print(
        f"lesion (inside): {lesion_synth_nets[0].__class__.__name__}  ENSEMBLE folds {args.lesion_image_folds} "
        f"({args.lesion_image_checkpoint_name})"
    )
    print(
        f"foreground     : {foreground_net.__class__.__name__} ({foreground_classes} cls, label {args.foreground_label})"
    )
    print(
        f"breast         : {breast_net.__class__.__name__} ({breast_classes} cls, label {args.breast_label})"
    )
    print(
        f"lesion seg     : {lesion_seg_nets[0].__class__.__name__}  ENSEMBLE folds {args.lesion_seg_folds} "
        f"({args.lesion_seg_checkpoint_name}, {lesion_classes} cls, label {args.lesion_label})"
    )
    print(
        f"divisor {divisor} | TTA {tta} | synth norm: {norm_mode} | feather {args.feather} | "
        f"lesion gain {args.lesion_gain} thr {args.lesion_threshold} step {args.lesion_step} "
        f"lesion_feather {args.lesion_feather} | air from: {args.air_from}"
    )
    print(f"Predicting {len(files)} file(s) from {input_dir} -> {output_dir}")

    for index, path in enumerate(files, 1):
        array, kind, like_itk = read_image(path)
        result = predict_array(
            lesion_synth_nets,
            image_nets,
            breast_net,
            foreground_net,
            lesion_seg_nets,
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
            args.air_from,
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
