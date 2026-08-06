"""Command-line synthesis of post-contrast slices from pre-contrast ones.

The translation networks take a normalized pre-contrast slice and predict a
normalized delta; the synthetic post-contrast image is the input plus that delta.
This script reads a folder of pre-contrast slices (``.mha``, ``.tif`` or ``.tiff``)
and writes a matching folder of synthetic post-contrast slices.

Only the generator's forward pass is needed here; none of the training-time loss
machinery is touched. The architecture is rebuilt from the run's ``plans.json``
using the trainer class named inside the checkpoint, so decoder variants are
handled automatically.

Normalization is per slice and chosen by flag:
  * default          -> z-score using the slice's own mean and standard deviation
  * ``--pre-stats``  -> z-score using the values stored in the given JSON file
  * ``--dont-norm``  -> feed raw values unchanged

The output is always mapped back to the input file's scale, so it comes out in the
same units it went in. For inputs that are already z-scored, ``--dont-norm`` keeps
the output in that same space.
"""

from __future__ import annotations

import argparse
import json
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

from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
from nnunetv2.utilities.label_handling.label_handling import (
    determine_num_input_channels,
)
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

MHA_SUFFIXES = (".mha",)
TIFF_SUFFIXES = (".tif", ".tiff")


def _input_divisor(configuration_manager) -> int:
    """The factor the input height and width must be divisible by.

    Equal to the product of the per-axis pooling strides, since the decoder can only
    restore sizes that survived every downsampling step. Falls back to a safe value if
    the plans cannot be read.
    """
    try:
        strides = configuration_manager.network_arch_init_kwargs["strides"]
        divisor = 1
        for stride in strides:
            divisor *= int(stride[0] if isinstance(stride, (list, tuple)) else stride)
        return max(divisor, 1)
    except Exception:
        return 64


def load_translation_network(
    checkpoint_path: str,
    device: torch.device,
    plans_path: Optional[str] = None,
    dataset_json_path: Optional[str] = None,
) -> Tuple[torch.nn.Module, int]:
    """Rebuild the generator from a run's plans and load one checkpoint into it.

    The run directory holding the plans and dataset files is taken to be the
    checkpoint's grandparent unless given explicitly. Returns the network together
    with its input-size divisor.
    """
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
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

    trainer_class = recursive_find_trainer_class_by_name(trainer_name)
    if trainer_class is None:
        raise RuntimeError(
            f"Could not find trainer class '{trainer_name}' referenced by the checkpoint."
        )

    # The trainer's builder is what forces a single output channel with deep
    # supervision off, and is also where decoder variants swap themselves in.
    network = trainer_class.build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels,
        1,
        enable_deep_supervision=False,
    )
    network.load_state_dict(checkpoint["network_weights"])
    network = network.to(device)
    network.eval()
    return network, _input_divisor(configuration_manager)


def _forward_delta(network: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run the network and take the full-resolution output if several are returned."""
    out = network(x)
    if isinstance(out, (list, tuple)):
        out = out[0]
    return out


@torch.no_grad()
def predict_slice(
    network: torch.nn.Module,
    x: np.ndarray,
    mean: float,
    std: float,
    divisor: int,
    device: torch.device,
    tta: bool = False,
) -> np.ndarray:
    """Synthesise the post-contrast version of one 2D slice.

    The slice is normalized, padded up to a valid size, passed through the generator,
    cropped back and mapped to the input's scale. With ``tta`` the prediction is
    averaged over the four flips of the input, each unflipped again before averaging.
    """
    height, width = x.shape
    pre_norm = (x.astype(np.float32) - float(mean)) / max(float(std), 1e-8)

    tensor = torch.from_numpy(np.ascontiguousarray(pre_norm))[None, None].to(device)

    pad_h = (-height) % divisor
    pad_w = (-width) % divisor
    if pad_h or pad_w:
        # Pad with zeros on the bottom and right, matching the canvas padding used during
        # training.
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


class _nullcontext:
    """Do-nothing context manager, for the branch where autocast is not used."""

    def __enter__(self):
        """Enter and yield nothing."""
        return None

    def __exit__(self, *exc):
        """Exit without suppressing exceptions."""
        return False


def predict_array(
    network: torch.nn.Module,
    array: np.ndarray,
    mean_std_fn,
    divisor: int,
    device: torch.device,
    tta: bool,
) -> np.ndarray:
    """Predict a 2D or 3D array slice by slice.

    ``mean_std_fn`` supplies the normalization statistics for each slice.
    """
    if array.ndim == 2:
        mean, std = mean_std_fn(array)
        return predict_slice(network, array, mean, std, divisor, device, tta)
    if array.ndim == 3:
        out = np.empty_like(array, dtype=np.float32)
        for i in range(array.shape[0]):
            mean, std = mean_std_fn(array[i])
            out[i] = predict_slice(network, array[i], mean, std, divisor, device, tta)
        return out
    raise ValueError(
        f"Unsupported array shape {array.shape}; expected 2D (H,W) or 3D (n,H,W)."
    )


def read_image(path: Path):
    """Read one image file, returning its array, its kind, and any header object.

    The header is kept for volume formats so the output can inherit its geometry.
    """
    suffix = path.suffix.lower()
    if suffix in MHA_SUFFIXES:
        import SimpleITK as sitk

        itk_image = sitk.ReadImage(str(path))
        array = sitk.GetArrayFromImage(itk_image).astype(np.float32)
        return array, "mha", itk_image
    if suffix in TIFF_SUFFIXES:
        import tifffile

        array = np.asarray(tifffile.imread(str(path))).astype(np.float32)
        return array, "tiff", None
    raise ValueError(f"Unsupported file type: {path}")


def write_image(array: np.ndarray, out_path: Path, kind: str, like_itk_image) -> None:
    """Write an array back out in the given format, copying geometry where available."""
    if kind == "mha":
        import SimpleITK as sitk

        out_itk = sitk.GetImageFromArray(array.astype(np.float32))
        if like_itk_image is not None:
            out_itk.CopyInformation(like_itk_image)
        sitk.WriteImage(out_itk, str(out_path), True)
    elif kind == "tiff":
        import tifffile

        tifffile.imwrite(str(out_path), array.astype(np.float32))
    else:
        raise ValueError(f"Unsupported output kind: {kind}")


def _gather_inputs(input_dir: Path):
    """All supported image files in a folder, sorted by name."""
    files = []
    for suffix in MHA_SUFFIXES + TIFF_SUFFIXES:
        files.extend(input_dir.glob(f"*{suffix}"))
    return sorted(set(files), key=lambda p: p.name)


def _make_mean_std_fn(pre_stats_path: Optional[str], dont_norm: bool):
    """Build the per-slice normalization function the flags ask for.

    Either fixed statistics from a file, no normalization at all, or statistics
    computed from each slice.
    """
    if dont_norm:
        return lambda _slice: (0.0, 1.0)
    if pre_stats_path is not None:
        with open(pre_stats_path, "r", encoding="utf-8") as handle:
            stats = json.load(handle)
        if "mean" not in stats or "std" not in stats:
            raise ValueError(f"{pre_stats_path} must contain 'mean' and 'std'.")
        mean = float(stats["mean"])
        std = float(stats["std"])
        return lambda _slice: (mean, std)
    return lambda slice_2d: (
        float(np.mean(slice_2d)),
        max(float(np.std(slice_2d)), 1e-8),
    )


def main():
    """Parse arguments, load the generator, and translate every file in the input folder."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic post-contrast slices from pre-contrast .mha/.tiff files using "
        "a MamaSynth translation checkpoint (predicts a normalized delta; synthetic post = pre + delta).",
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
        help="Full path to the specific .pth checkpoint to use (e.g. <run>/fold_0/checkpoint_best.pth). "
        "Its parent.parent must contain plans.json and dataset.json unless --plans/--dataset-json are given.",
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
        help="Feed raw input values as-is (no normalization).",
    )
    parser.add_argument(
        "--plans",
        type=str,
        default=None,
        help="Override path to plans.json (default: derived from checkpoint).",
    )
    parser.add_argument(
        "--dataset-json",
        type=str,
        default=None,
        help="Override path to dataset.json (default: derived from checkpoint).",
    )
    parser.add_argument("-device", type=str, default="cuda", help="cuda | cpu | mps.")
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Test-time augmentation: average over horizontal/vertical flips.",
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

    network, divisor = load_translation_network(
        args.checkpoint, device, args.plans, args.dataset_json
    )
    mean_std_fn = _make_mean_std_fn(args.pre_stats, args.dont_norm)

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
        f"Loaded {network.__class__.__name__} | input divisor {divisor} | normalization: {norm_mode}"
    )
    print(f"Predicting {len(files)} file(s) from {input_dir} -> {output_dir}")

    for idx, path in enumerate(files, 1):
        array, kind, like_itk = read_image(path)
        prediction = predict_array(
            network, array, mean_std_fn, divisor, device, not args.no_tta
        )
        write_image(prediction, output_dir / path.name, kind, like_itk)
        print(f"  [{idx}/{len(files)}] {path.name}  shape={tuple(array.shape)}")

    print("Done.")


if __name__ == "__main__":
    main()
