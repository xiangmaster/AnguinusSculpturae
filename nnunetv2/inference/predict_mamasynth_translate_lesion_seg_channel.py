"""Synthesis for the two-channel generators that take a lesion mask as second input.

Those trainers receive the pre-contrast image and a binary lesion channel. During
training that channel is the augmented reference mask; at inference there is no
reference, so it is produced by a lesion segmentation model.

Per slice:
  1. segment the breast
  2. segment the lesion, giving a probability map
  3. build the lesion channel as the in-breast voxels above a threshold, lowered in
     steps until at least one passes, falling back to the single highest-probability
     in-breast voxel; predictions outside the breast are discarded
  4. run the translation network on the two channels together
  5. add the predicted delta to the input and map back to the input's scale

Both segmentation networks run like the other segmenters here. The synthesis input
follows the normalization options of the single-model script.
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
    _forward_delta,
    _gather_inputs,
    _make_mean_std_fn,
    _nullcontext,
    load_translation_network,
    read_image,
    write_image,
)
from nnunetv2.inference.predict_mamasynth_translate_stitched import (
    load_seg_network,
    predict_breast_mask,
)
from nnunetv2.inference.predict_mamasynth_translate_stitched_lesion import (
    predict_lesion_prob,
)


def lesion_channel_in_breast(
    breast_mask: np.ndarray,
    lesion_prob: np.ndarray,
    threshold: float,
    step: float,
) -> np.ndarray:
    """Binary in-breast lesion mask to use as the network's second input channel.

    Candidates are in-breast voxels whose probability clears the threshold, which is
    lowered in steps until the set is non-empty; if nothing passes even near zero, the
    single highest-probability in-breast voxel is used. Predictions outside the breast
    are dropped.
    """
    breast = breast_mask.astype(bool)
    mask = np.zeros_like(breast, dtype=np.uint8)
    if not breast.any():
        return mask

    prob_in_breast = np.where(breast, lesion_prob, -np.inf)
    t = float(threshold)
    cand = breast & (lesion_prob >= t)
    while not cand.any() and t > 0.0:
        t = max(0.0, t - float(step))
        cand = breast & (lesion_prob >= t)
    if not cand.any():
        cand = prob_in_breast == prob_in_breast.max()
        cand &= breast
    mask[cand] = 1
    return mask


@torch.no_grad()
def predict_slice_with_lesion_channel(
    network: torch.nn.Module,
    x: np.ndarray,
    lesion_mask: np.ndarray,
    mean: float,
    std: float,
    divisor: int,
    device: torch.device,
    tta: bool = False,
) -> np.ndarray:
    """Synthesise one slice from the pre-contrast image and its lesion channel.

    Mirrors the single-channel prediction path, but feeds both channels. The lesion
    channel is binary and zero-padded the same way the image is.
    """
    height, width = x.shape
    pre_norm = (x.astype(np.float32) - float(mean)) / max(float(std), 1e-8)
    pre_t = torch.from_numpy(np.ascontiguousarray(pre_norm))[None, None]
    mask_t = torch.from_numpy(np.ascontiguousarray(lesion_mask.astype(np.float32)))[
        None, None
    ]
    tensor = torch.cat((pre_t, mask_t), dim=1).to(device)

    pad_h = (-height) % divisor
    pad_w = (-width) % divisor
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

    use_autocast = device.type == "cuda"
    with (
        torch.autocast(device.type, enabled=use_autocast)
        if use_autocast
        else _nullcontext()
    ):
        if tta:
            deltas = []
            for flip_dims in ((), (2,), (3,), (2, 3)):
                flipped = torch.flip(tensor, dims=flip_dims) if flip_dims else tensor
                d = _forward_delta(network, flipped)
                deltas.append(torch.flip(d, dims=flip_dims) if flip_dims else d)
            delta = torch.stack(deltas, 0).mean(0)
        else:
            delta = _forward_delta(network, tensor)

    delta = delta[..., :height, :width].float()
    delta_np = delta[0, 0].cpu().numpy()
    post_norm = pre_norm + delta_np
    post_out = post_norm * max(float(std), 1e-8) + float(mean)
    return post_out.astype(np.float32)


def predict_lesion_seg_channel_array(
    net: torch.nn.Module,
    breast_net: torch.nn.Module,
    lesion_net: torch.nn.Module,
    array: np.ndarray,
    mean_std_fn,
    divisor: int,
    device: torch.device,
    tta: bool,
    breast_label: int,
    lesion_label: int,
    lesion_threshold: float,
    lesion_step: float,
    return_lesion_mask: bool = False,
):
    """Translate a 2D or 3D array, supplying the in-breast lesion mask as a second channel."""

    def _one(slice_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Segment, build the lesion channel and translate a single slice."""
        mean, std = mean_std_fn(slice_2d)
        breast = predict_breast_mask(breast_net, slice_2d, device, breast_label, tta)
        lesion_prob = predict_lesion_prob(
            lesion_net, slice_2d, device, lesion_label, tta
        )
        chan = lesion_channel_in_breast(
            breast, lesion_prob, lesion_threshold, lesion_step
        )
        post = predict_slice_with_lesion_channel(
            net, slice_2d, chan, mean, std, divisor, device, tta
        )
        return post, chan

    if array.ndim == 2:
        out, chan = _one(array)
        return (out, chan) if return_lesion_mask else out
    if array.ndim == 3:
        out = np.empty_like(array, dtype=np.float32)
        chans = np.empty(array.shape, dtype=np.uint8) if return_lesion_mask else None
        for i in range(array.shape[0]):
            out[i], c = _one(array[i])
            if return_lesion_mask:
                chans[i] = c
        return (out, chans) if return_lesion_mask else out
    raise ValueError(
        f"Unsupported array shape {array.shape}; expected 2D (H,W) or 3D (n,H,W)."
    )


def main():
    """Parse arguments, load all three networks, and translate each file in the input folder."""
    parser = argparse.ArgumentParser(
        description="MamaSynth translation inference for *_LesionSegChannel 2-channel generators: "
        "predict breast + lesion masks, keep the in-breast lesion (lowering the threshold until "
        "non-empty), and feed [pre, lesion_mask] to the translation net.",
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
        help="Output folder for synthetic post-contrast files (same filenames).",
    )
    parser.add_argument(
        "-chk",
        "--checkpoint",
        required=True,
        type=str,
        help="2-channel translation .pth (a *_LesionSegChannel run).",
    )
    parser.add_argument(
        "-chk-breast",
        "--breast-weights",
        required=True,
        type=str,
        help="Breast segmentation .pth.",
    )
    parser.add_argument(
        "-chk-lesion",
        "--lesion-seg-weights",
        required=True,
        type=str,
        help="Lesion segmentation .pth (produces the second input channel).",
    )
    parser.add_argument(
        "--breast-label",
        type=int,
        default=1,
        help="Label index that denotes 'breast' in the breast seg net.",
    )
    parser.add_argument(
        "--lesion-label",
        type=int,
        default=1,
        help="Softmax channel index that denotes 'lesion' in the lesion seg net.",
    )
    parser.add_argument(
        "--lesion-threshold",
        type=float,
        default=0.5,
        help="Default lesion probability cutoff (lowered until any in-breast voxel passes).",
    )
    parser.add_argument(
        "--lesion-step",
        type=float,
        default=0.05,
        help="Decrement used to lower the lesion threshold.",
    )
    parser.add_argument(
        "--save-lesion-mask",
        action="store_true",
        help="Also write the lesion channel next to each output as '<name>_lesionmask.<ext>'.",
    )
    parser.add_argument(
        "--pre-stats",
        type=str,
        default=None,
        help="JSON with global {'mean','std'} for input z-scoring (mutually exclusive with --dont-norm).",
    )
    parser.add_argument(
        "--dont-norm",
        action="store_true",
        help="Feed raw synthesis input values as-is (no normalization).",
    )
    parser.add_argument(
        "--plans",
        type=str,
        default=None,
        help="Override plans.json for the translation model.",
    )
    parser.add_argument(
        "--dataset-json",
        type=str,
        default=None,
        help="Override dataset.json for the translation model.",
    )
    parser.add_argument("--plans-breast", type=str, default=None)
    parser.add_argument("--dataset-json-breast", type=str, default=None)
    parser.add_argument("--plans-lesion", type=str, default=None)
    parser.add_argument("--dataset-json-lesion", type=str, default=None)
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

    net, divisor = load_translation_network(
        args.checkpoint, device, args.plans, args.dataset_json
    )
    breast_net, num_breast_classes = load_seg_network(
        args.breast_weights, device, args.plans_breast, args.dataset_json_breast
    )
    lesion_net, num_lesion_classes = load_seg_network(
        args.lesion_seg_weights, device, args.plans_lesion, args.dataset_json_lesion
    )
    if args.breast_label >= num_breast_classes:
        parser.error(
            f"--breast-label {args.breast_label} is out of range for a {num_breast_classes}-class breast net."
        )
    if args.lesion_label >= num_lesion_classes:
        parser.error(
            f"--lesion-label {args.lesion_label} is out of range for a {num_lesion_classes}-class lesion net."
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
    print(f"translate (2ch): {net.__class__.__name__}  ({Path(args.checkpoint).name})")
    print(
        f"breast : {breast_net.__class__.__name__}  ({num_breast_classes} classes, label {args.breast_label})"
    )
    print(
        f"lesion : {lesion_net.__class__.__name__}  ({num_lesion_classes} classes, label {args.lesion_label})"
    )
    print(
        f"divisor {divisor} | TTA {tta} | synth norm: {norm_mode} | "
        f"lesion thr {args.lesion_threshold} step {args.lesion_step}"
    )
    print(f"Predicting {len(files)} file(s) from {input_dir} -> {output_dir}")

    for idx, path in enumerate(files, 1):
        array, kind, like_itk = read_image(path)
        result = predict_lesion_seg_channel_array(
            net,
            breast_net,
            lesion_net,
            array,
            mean_std_fn,
            divisor,
            device,
            tta,
            args.breast_label,
            args.lesion_label,
            args.lesion_threshold,
            args.lesion_step,
            return_lesion_mask=args.save_lesion_mask,
        )
        prediction, lesion = result if args.save_lesion_mask else (result, None)
        write_image(prediction, output_dir / path.name, kind, like_itk)
        if args.save_lesion_mask:
            mask_name = f"{path.stem}_lesionmask{path.suffix}"
            write_image(
                lesion.astype(np.float32), output_dir / mask_name, kind, like_itk
            )
        print(f"  [{idx}/{len(files)}] {path.name}  shape={tuple(array.shape)}")

    print("Done.")


if __name__ == "__main__":
    main()
