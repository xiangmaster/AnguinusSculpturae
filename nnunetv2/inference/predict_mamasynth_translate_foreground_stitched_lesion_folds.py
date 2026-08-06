"""Foreground-stitched synthesis with the lesion segmentation ensembled over folds.

Identical to the foreground-stitched lesion script except that the lesion
segmentation driving the in-breast mask comes from several folds of one run: each
fold's probability map is averaged before thresholding, which localises the lesion
more reliably than a single fold. The rest of the pipeline is reused unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

from nnunetv2.inference.predict_mamasynth_translate import (
    _gather_inputs,
    _make_mean_std_fn,
    load_translation_network,
    read_image,
    write_image,
)
from nnunetv2.inference.predict_mamasynth_translate_foreground_stitched import (
    _device_from_arg,
    _write_extras,
    resolve_checkpoint,
)
from nnunetv2.inference.predict_mamasynth_translate_foreground_stitched_lesion import (
    predict_foreground_stitched_lesion_array,
)
from nnunetv2.inference.predict_mamasynth_translate_stitched import load_seg_network


def main():
    """Parse arguments, load the fold ensemble alongside the other networks, and run."""
    parser = argparse.ArgumentParser(
        description="Foreground-stitched synthesis with lesion gain, where the lesion segmentation is "
        "ensembled over several folds of one run (averaged softmax). Same compositing/gain as "
        "predict_mamasynth_translate_foreground_stitched_lesion.",
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
    parser.add_argument("--image-checkpoint-name", default=None, type=str)
    parser.add_argument("--lesion-image-checkpoint-name", default=None, type=str)
    parser.add_argument("--foreground-checkpoint-name", default=None, type=str)
    parser.add_argument("--breast-checkpoint-name", default=None, type=str)
    parser.add_argument(
        "--lesion-seg-dir",
        required=True,
        type=str,
        help="Lesion segmentation RUN directory (contains fold_X/ subfolders + plans.json).",
    )
    parser.add_argument(
        "--lesion-seg-folds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="Fold indices to ensemble for the lesion segmentation.",
    )
    parser.add_argument(
        "--lesion-seg-checkpoint-name",
        default="checkpoint_best.pth",
        type=str,
        help="Checkpoint filename within each fold_X/ directory.",
    )
    parser.add_argument(
        "--plans-lesion-seg",
        type=str,
        default=None,
        help="Override plans.json for the lesion seg run (shared by all folds).",
    )
    parser.add_argument(
        "--dataset-json-lesion-seg",
        type=str,
        default=None,
        help="Override dataset.json for the lesion seg run (shared by all folds).",
    )
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
        default=0.0,
        help="Gaussian feather (sigma, px) on the lesion soft-mask edge.",
    )
    parser.add_argument("--feather", type=float, default=4.0)
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--save-intermediates", action="store_true")
    parser.add_argument("--save-lesion-mask", action="store_true")
    parser.add_argument("--pre-stats", type=str, default=None)
    parser.add_argument("--dont-norm", action="store_true")
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

    lesion_seg_dir = Path(args.lesion_seg_dir).expanduser()
    lesion_seg_ckpts = []
    for fold in args.lesion_seg_folds:
        ckpt = lesion_seg_dir / f"fold_{fold}" / args.lesion_seg_checkpoint_name
        if not ckpt.is_file():
            parser.error(f"Lesion seg fold checkpoint not found: {ckpt}")
        lesion_seg_ckpts.append(ckpt)

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

    lesion_seg_nets = []
    lesion_classes = None
    for ckpt in lesion_seg_ckpts:
        net, ncls = load_seg_network(
            str(ckpt), device, args.plans_lesion_seg, args.dataset_json_lesion_seg
        )
        lesion_seg_nets.append(net)
        lesion_classes = ncls

    if args.foreground_label >= foreground_classes:
        parser.error(
            f"--foreground-label {args.foreground_label} is out of range for a "
            f"{foreground_classes}-class foreground net."
        )
    if args.breast_label >= breast_classes:
        parser.error(
            f"--breast-label {args.breast_label} is out of range for a {breast_classes}-class breast net."
        )
    if args.lesion_label >= lesion_classes:
        parser.error(
            f"--lesion-label {args.lesion_label} is out of range for a {lesion_classes}-class lesion net."
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
        f"foreground   : {foreground_net.__class__.__name__} ({foreground_classes} cls, label {args.foreground_label})"
    )
    print(
        f"breast       : {breast_net.__class__.__name__} ({breast_classes} cls, label {args.breast_label})"
    )
    print(
        f"lesion seg   : {lesion_seg_nets[0].__class__.__name__}  ENSEMBLE of folds {args.lesion_seg_folds} "
        f"({args.lesion_seg_checkpoint_name}, {lesion_classes} cls, label {args.lesion_label})"
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
