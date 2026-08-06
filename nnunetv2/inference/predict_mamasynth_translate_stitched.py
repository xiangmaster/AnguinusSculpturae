"""Synthesis from two translation models, joined along the breast boundary.

Per slice: one model synthesises the image used inside the predicted breast,
another the image used outside it, and a breast segmentation decides where each
applies.

The translation networks are loaded exactly as in the single-model script. The
breast network is an ordinary segmentation run; because it was trained on
fixed-size, per-slice z-scored inputs, each slice is resized to that size,
z-scored, run through the network with optional flip averaging, argmaxed, and the
resulting binary mask resized back to the native size with nearest-neighbour
interpolation. Both synthetic images are produced at native resolution and joined
there.

``--dilate`` grows the breast region and ``--feather`` blends the seam over a soft
margin, so the transition is a ramp rather than a hard step.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from batchgenerators.utilities.file_and_folder_operations import (
    load_json,
    maybe_mkdir_p,
)

from nnunetv2.inference.predict_mamasynth_translate import (
    _gather_inputs,
    _input_divisor,
    _make_mean_std_fn,
    _nullcontext,
    load_translation_network,
    predict_slice,
    read_image,
    write_image,
)
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
from nnunetv2.utilities.label_handling.label_handling import (
    determine_num_input_channels,
)
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

SEG_TARGET_SIZE = 512


def load_seg_network(
    checkpoint_path: str,
    device: torch.device,
    plans_path: Optional[str] = None,
    dataset_json_path: Optional[str] = None,
) -> Tuple[torch.nn.Module, int]:
    """Rebuild a segmentation network from its run's plans and load its weights.

    Returns the network together with its number of output classes.
    """
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Breast checkpoint not found: {checkpoint_path}")
    run_dir = checkpoint_path.parent.parent

    plans = load_json(plans_path or str(run_dir / "plans.json"))
    dataset_json = load_json(dataset_json_path or str(run_dir / "dataset.json"))

    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False
    )
    trainer_name = checkpoint["trainer_name"]
    configuration_name = checkpoint.get("init_args", {}).get("configuration", "2d")

    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(configuration_name)
    num_input_channels = determine_num_input_channels(
        plans_manager, configuration_manager, dataset_json
    )
    label_manager = plans_manager.get_label_manager(dataset_json)
    num_classes = label_manager.num_segmentation_heads

    trainer_class = recursive_find_trainer_class_by_name(trainer_name)
    if trainer_class is None:
        raise RuntimeError(
            f"Could not find trainer class '{trainer_name}' referenced by the breast checkpoint."
        )

    network = trainer_class.build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels,
        num_classes,
        enable_deep_supervision=False,
    )
    network.load_state_dict(checkpoint["network_weights"])
    network = network.to(device)
    network.eval()
    return network, num_classes


def _forward_logits(network: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run the network and take the full-resolution output if several are returned."""
    out = network(x)
    if isinstance(out, (list, tuple)):
        out = out[0]
    return out


@torch.no_grad()
def predict_breast_mask(
    network: torch.nn.Module,
    x: np.ndarray,
    device: torch.device,
    breast_label: int = 1,
    tta: bool = True,
) -> np.ndarray:
    """Binary breast mask at native resolution for one slice.

    The slice is resized to the size the network was trained on and z-scored, run with
    optional flip averaging, and argmaxed; the binary mask is then resized back with
    nearest-neighbour interpolation so no interpolated in-between values appear.
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

    mask512 = (probs.argmax(dim=1, keepdim=True) == breast_label).float()
    mask_native = F.interpolate(mask512, size=(height, width), mode="nearest")
    return mask_native[0, 0].cpu().numpy().astype(np.uint8)


def _build_blend_weight(mask: np.ndarray, dilate: int, feather: float) -> np.ndarray:
    """Turn a binary breast mask into a blend weight in [0, 1] for the inside image.

    ``dilate`` grows the region first. ``feather`` softens the seam over roughly that
    many pixels using a signed-distance ramp: one well inside, zero well outside, and
    linear across the boundary.
    """
    from scipy import ndimage

    binary = mask.astype(bool)
    if dilate > 0:
        binary = ndimage.binary_dilation(binary, iterations=int(dilate))

    if feather <= 0:
        return binary.astype(np.float32)

    dist_in = ndimage.distance_transform_edt(binary)
    dist_out = ndimage.distance_transform_edt(~binary)
    signed = dist_in - dist_out
    weight = np.clip(0.5 + signed / (2.0 * float(feather)), 0.0, 1.0)
    return weight.astype(np.float32)


def composite_slice(
    synth_in: np.ndarray,
    synth_out: np.ndarray,
    mask: np.ndarray,
    dilate: int,
    feather: float,
) -> np.ndarray:
    """Blend the inside and outside images using the breast weight."""
    if dilate <= 0 and feather <= 0:
        return np.where(mask.astype(bool), synth_in, synth_out).astype(np.float32)
    weight = _build_blend_weight(mask, dilate, feather)
    return (weight * synth_in + (1.0 - weight) * synth_out).astype(np.float32)


def predict_stitched_array(
    net_in: torch.nn.Module,
    net_out: torch.nn.Module,
    seg_net: torch.nn.Module,
    array: np.ndarray,
    mean_std_fn,
    divisor: int,
    device: torch.device,
    tta: bool,
    breast_label: int,
    dilate: int,
    feather: float,
    return_mask: bool = False,
):
    """Stitch a 2D or 3D array slice by slice, optionally returning the masks too."""

    def _one(slice_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Synthesise and stitch a single slice."""
        mean, std = mean_std_fn(slice_2d)
        synth_in = predict_slice(net_in, slice_2d, mean, std, divisor, device, tta)
        synth_out = predict_slice(net_out, slice_2d, mean, std, divisor, device, tta)
        mask = predict_breast_mask(seg_net, slice_2d, device, breast_label, tta)
        return composite_slice(synth_in, synth_out, mask, dilate, feather), mask

    if array.ndim == 2:
        out, mask = _one(array)
        return (out, mask) if return_mask else out
    if array.ndim == 3:
        out = np.empty_like(array, dtype=np.float32)
        masks = np.empty(array.shape, dtype=np.uint8) if return_mask else None
        for i in range(array.shape[0]):
            out[i], m = _one(array[i])
            if return_mask:
                masks[i] = m
        return (out, masks) if return_mask else out
    raise ValueError(
        f"Unsupported array shape {array.shape}; expected 2D (H,W) or 3D (n,H,W)."
    )


def main():
    """Parse arguments, load all three networks, and stitch every file in the input folder."""
    parser = argparse.ArgumentParser(
        description="Composite two MamaSynth synthetic post-contrast images using a breast segmentation "
        "mask: inside-breast pixels come from the '--checkpoint-in' translation model, outside-breast "
        "pixels from '--checkpoint-out'. All three nets run with flip TTA by default.",
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
        "--breast-label",
        type=int,
        default=1,
        help="Label index that denotes 'breast' in the segmentation net.",
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
    parser.add_argument("-device", type=str, default="cuda", help="cuda | cpu | mps.")
    parser.add_argument(
        "--no-tta", action="store_true", help="Disable flip TTA for all three networks."
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
    if args.breast_label >= num_classes:
        parser.error(
            f"--breast-label {args.breast_label} is out of range for a {num_classes}-class segmentation net."
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
        f"divisor {divisor} | TTA {tta} | synth norm: {norm_mode} | dilate {args.dilate} | feather {args.feather}"
    )
    print(f"Stitching {len(files)} file(s) from {input_dir} -> {output_dir}")

    for idx, path in enumerate(files, 1):
        array, kind, like_itk = read_image(path)
        result = predict_stitched_array(
            net_in,
            net_out,
            seg_net,
            array,
            mean_std_fn,
            divisor,
            device,
            tta,
            args.breast_label,
            args.dilate,
            args.feather,
            return_mask=args.save_mask,
        )
        prediction, mask = result if args.save_mask else (result, None)
        write_image(prediction, output_dir / path.name, kind, like_itk)
        if args.save_mask:
            mask_name = f"{path.stem}_breastmask{path.suffix}"
            write_image(mask.astype(np.float32), output_dir / mask_name, kind, like_itk)
        print(f"  [{idx}/{len(files)}] {path.name}  shape={tuple(array.shape)}")

    print("Done.")


if __name__ == "__main__":
    main()
