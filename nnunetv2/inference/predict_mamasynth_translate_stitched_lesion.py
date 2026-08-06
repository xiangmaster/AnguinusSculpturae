"""Stitched synthesis with a final lesion-enhancement step.

The inside/outside breast composite is built as in the plain stitched script, then
a lesion segmentation is run on the same slice and the predicted lesion region,
kept inside the breast, is scaled up by ``--lesion-gain``. Multiplying the
intensity already there keeps the lesion's internal texture, which setting the
region to a single value would flatten.

Per slice, with flip averaging by default:
  1. synthesise the image for inside the breast
  2. synthesise the image for outside it
  3. segment the breast
  4. blend the two by the breast mask, with optional dilation and feathering
  5. segment the lesion, giving a probability map
  6. scale the in-breast lesion region of the composite

The lesion threshold is lowered in steps until at least one in-breast voxel
passes, falling back to the single highest-probability in-breast voxel.

The lesion network is loaded and run exactly like the breast network.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

from nnunetv2.inference.predict_mamasynth_translate import (
    _gather_inputs,
    _make_mean_std_fn,
    _nullcontext,
    load_translation_network,
    predict_slice,
    read_image,
    write_image,
)
from nnunetv2.inference.predict_mamasynth_translate_stitched import (
    SEG_TARGET_SIZE,
    _forward_logits,
    composite_slice,
    load_seg_network,
    predict_breast_mask,
)


@torch.no_grad()
def _predict_lesion_prob_single(
    network: torch.nn.Module,
    x: np.ndarray,
    device: torch.device,
    lesion_label: int = 1,
    tta: bool = True,
) -> np.ndarray:
    """Native-resolution lesion probability map for one slice from one network.

    Follows the same path as the breast mask, but keeps the continuous probability and
    resizes it back bilinearly rather than argmaxing first.
    """
    height, width = x.shape
    slice_t = torch.from_numpy(np.ascontiguousarray(x.astype(np.float32)))[
        None, None
    ].to(device)
    resized = F.interpolate(
        slice_t,
        size=(SEG_TARGET_SIZE, SEG_TARGET_SIZE),
        mode="bilinear",
        align_corners=False,
    )

    mean = resized.mean()
    std = resized.std().clamp_min(1e-8)
    resized = (resized - mean) / std

    use_autocast = device.type == "cuda"
    with (
        torch.autocast(device.type, enabled=use_autocast)
        if use_autocast
        else _nullcontext()
    ):
        if tta:
            probs = None
            for flip_dims in ((), (2,), (3,), (2, 3)):
                flipped = torch.flip(resized, dims=flip_dims) if flip_dims else resized
                logits = _forward_logits(network, flipped)
                p = torch.softmax(logits.float(), dim=1)
                p = torch.flip(p, dims=flip_dims) if flip_dims else p
                probs = p if probs is None else probs + p
            probs = probs / 4.0
        else:
            probs = torch.softmax(_forward_logits(network, resized).float(), dim=1)

    lesion512 = probs[:, lesion_label : lesion_label + 1]
    lesion_native = F.interpolate(
        lesion512, size=(height, width), mode="bilinear", align_corners=False
    )
    return lesion_native[0, 0].cpu().numpy().astype(np.float32)


def predict_lesion_prob(
    network,
    x: np.ndarray,
    device: torch.device,
    lesion_label: int = 1,
    tta: bool = True,
) -> np.ndarray:
    """Native-resolution lesion probability for one slice, from one or several networks.

    Given several networks, their probability maps are averaged before being returned,
    which is the usual way to combine folds of the same run.
    """
    if isinstance(network, (list, tuple)):
        if len(network) == 0:
            raise ValueError("predict_lesion_prob received an empty network list.")
        acc = None
        for net in network:
            p = _predict_lesion_prob_single(net, x, device, lesion_label, tta)
            acc = p if acc is None else acc + p
        return (acc / float(len(network))).astype(np.float32)
    return _predict_lesion_prob_single(network, x, device, lesion_label, tta)


def in_breast_lesion_candidate(
    breast: np.ndarray,
    lesion_prob: np.ndarray,
    threshold: float,
    step: float,
) -> np.ndarray:
    """Boolean in-breast lesion mask, lowering the threshold until it is non-empty.

    The threshold drops in fixed steps until at least one in-breast voxel passes; if
    nothing passes even near zero, the single highest-probability in-breast voxel is
    taken. Assumes the breast mask has at least one voxel.
    """
    breast = breast.astype(bool)
    prob_in_breast = np.where(breast, lesion_prob, -np.inf)
    t = float(threshold)
    cand = breast & (lesion_prob >= t)
    while not cand.any() and t > 0.0:
        t = max(0.0, t - float(step))
        cand = breast & (lesion_prob >= t)
    if not cand.any():
        cand = (prob_in_breast == prob_in_breast.max()) & breast
    return cand


def lesion_soft_weight(
    cand: np.ndarray, breast: np.ndarray, feather: float
) -> np.ndarray:
    """Turn a binary lesion mask into a weight in [0, 1], feathered and clipped.

    Clipping to the breast keeps the weight from bleeding into surrounding tissue.
    """
    soft = cand.astype(np.float32)
    if feather and feather > 0.0:
        from scipy.ndimage import gaussian_filter

        soft = gaussian_filter(soft, sigma=float(feather))
        peak = soft.max()
        if peak > 0:
            soft = soft / peak
    return np.where(breast.astype(bool), soft, 0.0).astype(np.float32)


def enhance_lesion_and_dim_healthy(
    composite: np.ndarray,
    breast_mask: np.ndarray,
    lesion_prob: np.ndarray,
    threshold: float,
    step: float,
    lesion_gain: float,
    healthy_gain: float,
    feather: float = 0.0,
    healthy_erode: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scale the in-breast lesion up and the remaining breast tissue down.

    The lesion weight is feathered and clipped to the breast. Healthy tissue is the
    breast minus a dilated lesion, so the lesion's own bright rim is not included in
    what gets scaled down. Everything outside the breast is left alone. Modifies the
    composite in place and returns it with the lesion mask; a no-op when both factors
    are one.
    """
    breast = breast_mask.astype(bool)
    lesion_mask = np.zeros_like(breast, dtype=np.uint8)
    if not breast.any():
        return composite, lesion_mask
    cand = in_breast_lesion_candidate(breast, lesion_prob, threshold, step)
    lesion_mask[cand] = 1

    gain_field = np.ones(composite.shape, dtype=np.float32)
    if lesion_gain is not None and float(lesion_gain) != 1.0:
        lesion_soft = lesion_soft_weight(cand, breast, feather)
        gain_field = gain_field + (float(lesion_gain) - 1.0) * lesion_soft
    if healthy_gain is not None and float(healthy_gain) != 1.0:
        if healthy_erode and healthy_erode > 0:
            from scipy.ndimage import binary_dilation

            exclude = binary_dilation(cand, iterations=int(healthy_erode))
        else:
            exclude = cand
        healthy = breast & ~exclude
        gain_field = np.where(healthy, gain_field * float(healthy_gain), gain_field)

    composite[:] = (composite * gain_field).astype(composite.dtype)
    return composite, lesion_mask


def enhance_lesion_slice(
    composite: np.ndarray,
    breast_mask: np.ndarray,
    lesion_prob: np.ndarray,
    threshold: float,
    step: float,
    gain: float,
    feather: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scale the in-breast lesion region of the composite by a multiplicative factor.

    Multiplying the intensity that is already there preserves the lesion's internal
    texture, unlike setting the whole region to one value. The weight is feathered and
    clipped to the breast, so the change never spreads beyond it.

    Candidate voxels are those inside the breast whose lesion probability clears the
    threshold, which is lowered in steps until the set is non-empty.
    """
    breast = breast_mask.astype(bool)
    lesion_mask = np.zeros_like(breast, dtype=np.uint8)
    if not breast.any() or gain is None or float(gain) <= 1.0:
        return composite, lesion_mask

    cand = in_breast_lesion_candidate(breast, lesion_prob, threshold, step)
    lesion_mask[cand] = 1

    soft = lesion_soft_weight(cand, breast, feather)
    composite[:] = (composite * (1.0 + (float(gain) - 1.0) * soft)).astype(
        composite.dtype
    )
    return composite, lesion_mask


def predict_stitched_lesion_array(
    net_in: torch.nn.Module,
    net_out: torch.nn.Module,
    seg_net: torch.nn.Module,
    lesion_net: torch.nn.Module,
    array: np.ndarray,
    mean_std_fn,
    divisor: int,
    device: torch.device,
    tta: bool,
    breast_label: int,
    lesion_label: int,
    dilate: int,
    feather: float,
    lesion_threshold: float,
    lesion_step: float,
    lesion_gain: float,
    lesion_feather: float,
    return_mask: bool = False,
    return_lesion_mask: bool = False,
):
    """Stitch and enhance a 2D or 3D array, optionally returning the masks as well."""

    def _one(slice_2d: np.ndarray):
        """Synthesise, stitch and enhance a single slice."""
        mean, std = mean_std_fn(slice_2d)
        synth_in = predict_slice(net_in, slice_2d, mean, std, divisor, device, tta)
        synth_out = predict_slice(net_out, slice_2d, mean, std, divisor, device, tta)
        breast = predict_breast_mask(seg_net, slice_2d, device, breast_label, tta)
        composite = composite_slice(synth_in, synth_out, breast, dilate, feather)
        lesion_prob = predict_lesion_prob(
            lesion_net, slice_2d, device, lesion_label, tta
        )
        composite, lesion = enhance_lesion_slice(
            composite,
            breast,
            lesion_prob,
            lesion_threshold,
            lesion_step,
            lesion_gain,
            lesion_feather,
        )
        return composite, breast, lesion

    def _pack(out, breast, lesion):
        """Assemble the requested return values in a fixed order."""
        result = [out]
        if return_mask:
            result.append(breast)
        if return_lesion_mask:
            result.append(lesion)
        return tuple(result) if len(result) > 1 else result[0]

    if array.ndim == 2:
        return _pack(*_one(array))
    if array.ndim == 3:
        out = np.empty_like(array, dtype=np.float32)
        masks = np.empty(array.shape, dtype=np.uint8) if return_mask else None
        lesions = np.empty(array.shape, dtype=np.uint8) if return_lesion_mask else None
        for i in range(array.shape[0]):
            o, b, l = _one(array[i])
            out[i] = o
            if return_mask:
                masks[i] = b
            if return_lesion_mask:
                lesions[i] = l
        return _pack(out, masks, lesions)
    raise ValueError(
        f"Unsupported array shape {array.shape}; expected 2D (H,W) or 3D (n,H,W)."
    )


def main():
    """Parse arguments, load every network, and process each file in the input folder."""
    parser = argparse.ArgumentParser(
        description="Stitched MamaSynth synthesis with lesion enhancement: composite the inside/outside "
        "breast translation models, then scale the in-breast lesion region (from a lesion seg model) "
        "by --lesion-gain, which preserves its texture. All nets run with flip TTA by default.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Input folder of pre-contrast .mha/.tif/.tiff slices.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="Output folder for stitched synthetic post-contrast files (same filenames).",
    )
    parser.add_argument(
        "-chk-in",
        "--checkpoint-in",
        required=True,
        type=str,
        help="Translation .pth used INSIDE the predicted breast (e.g. <run>/fold_0/checkpoint_best.pth).",
    )
    parser.add_argument(
        "-chk-out",
        "--checkpoint-out",
        required=True,
        type=str,
        help="Translation .pth used OUTSIDE the predicted breast.",
    )
    parser.add_argument(
        "-chk-breast",
        "--checkpoint-breast",
        required=True,
        type=str,
        help="Breast segmentation .pth (e.g. Dataset625_Pre_Seg/nnUNetTrainerMamaSynthBreast_BS_32__mamaSynthPlans__2d/fold_0/checkpoint_final.pth).",
    )
    parser.add_argument(
        "-chk-lesion",
        "--checkpoint-lesion",
        required=True,
        type=str,
        help="Lesion segmentation .pth (e.g. Dataset625_Pre_Seg/nnUNetTrainerMamaSynthLesionTverskyA02B08_BS_64_epoch_1000__mamaSynthPlans__2d/fold_0/checkpoint_final.pth).",
    )
    parser.add_argument(
        "--breast-label",
        type=int,
        default=1,
        help="Label index that denotes 'breast' in the breast segmentation net.",
    )
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
        "--dilate",
        type=int,
        default=0,
        help="Grow the breast region by N pixels before compositing.",
    )
    parser.add_argument(
        "--feather",
        type=float,
        default=0.0,
        help="Soft-blend the inside/outside seam over ~N pixels (0 = hard edge).",
    )
    parser.add_argument(
        "--save-mask",
        action="store_true",
        help="Also write the breast mask next to each output as '<name>_breastmask.<ext>'.",
    )
    parser.add_argument(
        "--save-lesion-mask",
        action="store_true",
        help="Also write the lesion mask next to each output as '<name>_lesionmask.<ext>'.",
    )
    parser.add_argument(
        "--pre-stats",
        type=str,
        default=None,
        help="JSON with global {'mean','std'} for synthesis input z-scoring (mutually exclusive with --dont-norm).",
    )
    parser.add_argument(
        "--dont-norm",
        action="store_true",
        help="Feed raw synthesis input values as-is (no normalization).",
    )
    parser.add_argument(
        "--plans-in",
        type=str,
        default=None,
        help="Override plans.json for the inside model (default: derived from checkpoint).",
    )
    parser.add_argument(
        "--plans-out",
        type=str,
        default=None,
        help="Override plans.json for the outside model.",
    )
    parser.add_argument(
        "--plans-breast",
        type=str,
        default=None,
        help="Override plans.json for the breast model.",
    )
    parser.add_argument(
        "--plans-lesion",
        type=str,
        default=None,
        help="Override plans.json for the lesion model.",
    )
    parser.add_argument(
        "--dataset-json-in",
        type=str,
        default=None,
        help="Override dataset.json for the inside model.",
    )
    parser.add_argument(
        "--dataset-json-out",
        type=str,
        default=None,
        help="Override dataset.json for the outside model.",
    )
    parser.add_argument(
        "--dataset-json-breast",
        type=str,
        default=None,
        help="Override dataset.json for the breast model.",
    )
    parser.add_argument(
        "--dataset-json-lesion",
        type=str,
        default=None,
        help="Override dataset.json for the lesion model.",
    )
    parser.add_argument("-device", type=str, default="cuda", help="cuda | cpu | mps.")
    parser.add_argument(
        "--no-tta", action="store_true", help="Disable flip TTA for all networks."
    )
    args = parser.parse_args()

    if args.dont_norm and args.pre_stats is not None:
        parser.error("--pre-stats and --dont-norm are mutually exclusive.")

    input_dir = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser()
    if not input_dir.is_dir():
        parser.error(f"Input folder does not exist: {input_dir}")

    if args.device == "cuda":
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    elif args.device == "mps":
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        torch.set_num_threads(os.cpu_count() or 1)

    net_in, divisor_in = load_translation_network(
        args.checkpoint_in, device, args.plans_in, args.dataset_json_in
    )
    net_out, divisor_out = load_translation_network(
        args.checkpoint_out, device, args.plans_out, args.dataset_json_out
    )
    seg_net, num_classes = load_seg_network(
        args.checkpoint_breast, device, args.plans_breast, args.dataset_json_breast
    )
    lesion_net, num_lesion_classes = load_seg_network(
        args.checkpoint_lesion, device, args.plans_lesion, args.dataset_json_lesion
    )
    if args.breast_label >= num_classes:
        parser.error(
            f"--breast-label {args.breast_label} is out of range for a {num_classes}-class segmentation net."
        )
    if args.lesion_label >= num_lesion_classes:
        parser.error(
            f"--lesion-label {args.lesion_label} is out of range for a {num_lesion_classes}-class lesion net."
        )
    divisor = max(divisor_in, divisor_out)

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
    print(f"inside  : {net_in.__class__.__name__}  ({Path(args.checkpoint_in).name})")
    print(f"outside : {net_out.__class__.__name__}  ({Path(args.checkpoint_out).name})")
    print(
        f"breast  : {seg_net.__class__.__name__}  ({num_classes} classes, breast label {args.breast_label})"
    )
    print(
        f"lesion  : {lesion_net.__class__.__name__}  ({num_lesion_classes} classes, lesion label {args.lesion_label})"
    )
    print(
        f"divisor {divisor} | TTA {tta} | synth norm: {norm_mode} | dilate {args.dilate} | feather {args.feather} | "
        f"lesion gain {args.lesion_gain} thr {args.lesion_threshold} step {args.lesion_step} feather {args.lesion_feather}"
    )
    print(f"Stitching {len(files)} file(s) from {input_dir} -> {output_dir}")

    for idx, path in enumerate(files, 1):
        array, kind, like_itk = read_image(path)
        result = predict_stitched_lesion_array(
            net_in,
            net_out,
            seg_net,
            lesion_net,
            array,
            mean_std_fn,
            divisor,
            device,
            tta,
            args.breast_label,
            args.lesion_label,
            args.dilate,
            args.feather,
            args.lesion_threshold,
            args.lesion_step,
            args.lesion_gain,
            args.lesion_feather,
            return_mask=args.save_mask,
            return_lesion_mask=args.save_lesion_mask,
        )
        result = result if isinstance(result, tuple) else (result,)
        prediction = result[0]
        extras = list(result[1:])
        write_image(prediction, output_dir / path.name, kind, like_itk)
        if args.save_mask:
            mask = extras.pop(0)
            mask_name = f"{path.stem}_breastmask{path.suffix}"
            write_image(mask.astype(np.float32), output_dir / mask_name, kind, like_itk)
        if args.save_lesion_mask:
            lesion = extras.pop(0)
            lesion_name = f"{path.stem}_lesionmask{path.suffix}"
            write_image(
                lesion.astype(np.float32), output_dir / lesion_name, kind, like_itk
            )
        print(f"  [{idx}/{len(files)}] {path.name}  shape={tuple(array.shape)}")

    print("Done.")


if __name__ == "__main__":
    main()
