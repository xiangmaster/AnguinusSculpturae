from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from nnunetv2.architectures.curia.common import CURIA_MODEL_ID
from nnunetv2.training.mamasynth.dataset import (
    BACKGROUND_LABEL,
    BREAST_LABEL,
    TARGET_SIZE,
    lesion_slice_indices,
    load_cases,
    load_split,
    open_b2nd,
    resample_slice,
    zscore_per_slice,
)


def _crop_or_pad(
    array: np.ndarray, center_yx: tuple[float, float], fill: float
) -> np.ndarray:
    height, width = array.shape
    src_y = int(
        np.clip(round(center_yx[0] - TARGET_SIZE / 2), 0, max(0, height - TARGET_SIZE))
    )
    src_x = int(
        np.clip(round(center_yx[1] - TARGET_SIZE / 2), 0, max(0, width - TARGET_SIZE))
    )
    dst_y = max(0, (TARGET_SIZE - height) // 2)
    dst_x = max(0, (TARGET_SIZE - width) // 2)
    copy_h = min(height, TARGET_SIZE)
    copy_w = min(width, TARGET_SIZE)
    canvas = np.full((TARGET_SIZE, TARGET_SIZE), fill, dtype=array.dtype)
    canvas[dst_y : dst_y + copy_h, dst_x : dst_x + copy_w] = array[
        src_y : src_y + copy_h, src_x : src_x + copy_w
    ]
    return canvas


def _load_slice(
    case, slice_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pre = np.asarray(open_b2nd(case.pre_path)[slice_index], dtype=np.float32)
    post = np.asarray(open_b2nd(case.post_path)[slice_index], dtype=np.float32)
    tumor = np.asarray(open_b2nd(case.mask_path)[slice_index], dtype=np.uint8)
    tumor[tumor == BREAST_LABEL] = BACKGROUND_LABEL

    pre = resample_slice(pre, case.spacing_y, case.spacing_x, is_mask=False)
    post = resample_slice(post, case.spacing_y, case.spacing_x, is_mask=False)
    tumor = resample_slice(tumor, case.spacing_y, case.spacing_x, is_mask=True)
    valid = np.ones(tumor.shape, dtype=np.uint8)

    tumor = (tumor > 0).astype(np.uint8)
    lesion_y, lesion_x = np.nonzero(tumor)
    center_yx = (
        float(lesion_y.mean()) if lesion_y.size else tumor.shape[0] / 2,
        float(lesion_x.mean()) if lesion_x.size else tumor.shape[1] / 2,
    )
    pre = _crop_or_pad(zscore_per_slice(pre).astype(np.float32), center_yx, fill=0.0)
    post = _crop_or_pad(zscore_per_slice(post).astype(np.float32), center_yx, fill=0.0)
    tumor = _crop_or_pad(tumor, center_yx, fill=0)
    valid = _crop_or_pad(valid, center_yx, fill=0)
    return pre, post, tumor, valid


def _draw_samples(cases: list, num_samples: int, seed: int) -> list[tuple]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cases))
    samples = []
    cursor = 0
    attempts = 0
    max_attempts = max(num_samples * 20, len(cases) * 2)
    while len(samples) < num_samples and attempts < max_attempts:
        if cursor == len(order):
            order = rng.permutation(len(cases))
            cursor = 0
        case = cases[int(order[cursor])]
        cursor += 1
        attempts += 1
        indices = lesion_slice_indices(str(case.mask_path))
        if indices:
            samples.append((case, int(rng.choice(indices))))
    if len(samples) < num_samples:
        raise RuntimeError(
            f"Found only {len(samples)} lesion slices after {attempts} attempts."
        )
    return samples


def _extract_patch_tokens(model, images: torch.Tensor) -> torch.Tensor:
    outputs = model(pixel_values=images, output_hidden_states=True)
    hidden = outputs.hidden_states[-1]
    patch_size = int(model.config.patch_size)
    num_patches = (images.shape[-2] // patch_size) * (images.shape[-1] // patch_size)
    return hidden[:, -num_patches:].float()


def _token_masks(
    tumor: torch.Tensor, valid: torch.Tensor, patch_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tumor_tokens = (
        F.max_pool2d(tumor.float(), kernel_size=patch_size, stride=patch_size).flatten(
            1
        )
        > 0
    )
    valid_fraction = F.avg_pool2d(
        valid.float(), kernel_size=patch_size, stride=patch_size
    ).flatten(1)
    fully_valid_tokens = valid_fraction > 0.999
    return tumor_tokens, fully_valid_tokens & ~tumor_tokens, fully_valid_tokens


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def _distance_rows(
    pre_tokens: torch.Tensor,
    post_tokens: torch.Tensor,
    tumor: torch.Tensor,
    valid: torch.Tensor,
    patch_size: int,
) -> dict[str, torch.Tensor]:
    tumor_tokens, other_tokens, valid_tokens = _token_masks(tumor, valid, patch_size)
    pre_unit = F.normalize(pre_tokens, dim=-1)
    post_unit = F.normalize(post_tokens, dim=-1)
    cosine_distance = 1.0 - (pre_unit * post_unit).sum(dim=-1)
    normalized_mse = (pre_unit - post_unit).square().mean(dim=-1)
    raw_mse = (pre_tokens - post_tokens).square().mean(dim=-1)
    shuffled_post = post_unit.roll(shifts=1, dims=0)
    shuffled_cosine_distance = 1.0 - (pre_unit * shuffled_post).sum(dim=-1)
    tumor_cosine = _masked_mean(cosine_distance, tumor_tokens)
    non_tumor_cosine = _masked_mean(cosine_distance, other_tokens)
    whole_cosine = _masked_mean(cosine_distance, valid_tokens)
    tumor_shuffled_cosine = _masked_mean(shuffled_cosine_distance, tumor_tokens)
    whole_shuffled_cosine = _masked_mean(shuffled_cosine_distance, valid_tokens)

    return {
        "tumor_token_count": tumor_tokens.sum(dim=1),
        "non_tumor_token_count": other_tokens.sum(dim=1),
        "whole_valid_token_count": valid_tokens.sum(dim=1),
        "tumor_cosine_distance": tumor_cosine,
        "tumor_normalized_mse": _masked_mean(normalized_mse, tumor_tokens),
        "tumor_raw_mse": _masked_mean(raw_mse, tumor_tokens),
        "whole_cosine_distance": whole_cosine,
        "whole_normalized_mse": _masked_mean(normalized_mse, valid_tokens),
        "whole_raw_mse": _masked_mean(raw_mse, valid_tokens),
        "non_tumor_cosine_distance": non_tumor_cosine,
        "tumor_minus_whole_cosine": tumor_cosine - whole_cosine,
        "tumor_over_whole_cosine": tumor_cosine / whole_cosine.clamp_min(1e-8),
        "tumor_relative_to_whole_cosine_increase": (
            (tumor_cosine - whole_cosine) / whole_cosine.clamp_min(1e-8)
        ),
        "tumor_minus_non_tumor_cosine": tumor_cosine - non_tumor_cosine,
        "tumor_over_non_tumor_cosine": tumor_cosine / non_tumor_cosine.clamp_min(1e-8),
        "tumor_relative_cosine_increase": (
            (tumor_cosine - non_tumor_cosine) / non_tumor_cosine.clamp_min(1e-8)
        ),
        "shuffled_post_tumor_cosine_distance": tumor_shuffled_cosine,
        "shuffled_post_whole_cosine_distance": whole_shuffled_cosine,
        "tumor_matched_over_shuffled_cosine": tumor_cosine
        / tumor_shuffled_cosine.clamp_min(1e-8),
        "whole_matched_over_shuffled_cosine": whole_cosine
        / whole_shuffled_cosine.clamp_min(1e-8),
    }


def _summary(rows: list[dict], field: str) -> dict[str, float]:
    values = np.asarray([row[field] for row in rows], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _paired_signal_summary(rows: list[dict], seed: int) -> dict[str, float]:
    from scipy.stats import wilcoxon

    differences = np.asarray(
        [row["tumor_minus_non_tumor_cosine"] for row in rows], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(10_000, dtype=np.float64)
    for index in range(bootstrap_means.size):
        bootstrap_means[index] = rng.choice(
            differences, size=differences.size, replace=True
        ).mean()
    std = differences.std(ddof=1)
    statistic, p_value = wilcoxon(differences, alternative="greater")
    return {
        "mean_paired_difference": float(differences.mean()),
        "mean_difference_ci95_low": float(np.quantile(bootstrap_means, 0.025)),
        "mean_difference_ci95_high": float(np.quantile(bootstrap_means, 0.975)),
        "fraction_tumor_greater": float((differences > 0).mean()),
        "paired_effect_size_cohens_dz": float(differences.mean() / max(std, 1e-12)),
        "wilcoxon_greater_statistic": float(statistic),
        "wilcoxon_greater_p_value": float(p_value),
    }


def assess(args: argparse.Namespace) -> None:
    from transformers import AutoModel

    dataset_dir = Path(args.dataset_dir)
    split = load_split(dataset_dir, args.fold)
    selected_ids = set(split[args.split])
    cases = [case for case in load_cases(dataset_dir) if case.case_id in selected_ids]
    if not cases:
        raise ValueError(f"No usable cases from fold {args.fold} split '{args.split}'.")

    samples = _draw_samples(cases, args.num_samples, args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    model = AutoModel.from_pretrained(args.model).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    patch_size = int(model.config.patch_size)
    if TARGET_SIZE % patch_size:
        raise ValueError(
            f"Input size {TARGET_SIZE} is not divisible by Curia patch size {patch_size}."
        )

    rows: list[dict] = []
    start = 0
    progress = tqdm(total=len(samples), desc="Curia-2")
    while start < len(samples):
        end = min(start + args.batch_size, len(samples))
        if len(samples) - end == 1:
            end += 1
        batch = samples[start:end]
        loaded = [_load_slice(case, index) for case, index in batch]
        pre = torch.from_numpy(np.stack([item[0] for item in loaded]))[:, None].to(
            device
        )
        post = torch.from_numpy(np.stack([item[1] for item in loaded]))[:, None].to(
            device
        )
        tumor = torch.from_numpy(np.stack([item[2] for item in loaded]))[:, None].to(
            device
        )
        valid = torch.from_numpy(np.stack([item[3] for item in loaded]))[:, None].to(
            device
        )

        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type, dtype=dtype, enabled=device.type == "cuda"
            ),
        ):
            pre_tokens = _extract_patch_tokens(model, pre)
            post_tokens = _extract_patch_tokens(model, post)
        metrics = _distance_rows(pre_tokens, post_tokens, tumor, valid, patch_size)

        for batch_index, (case, slice_index) in enumerate(batch):
            row = {"case_id": case.case_id, "slice_index": slice_index}
            row.update(
                {
                    name: float(values[batch_index].cpu())
                    for name, values in metrics.items()
                }
            )
            rows.append(row)
        progress.update(len(batch))
        start = end
    progress.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    count_fields = {
        "tumor_token_count",
        "non_tumor_token_count",
        "whole_valid_token_count",
    }
    metric_fields = [
        field
        for field in rows[0]
        if field not in {"case_id", "slice_index", *count_fields}
    ]
    report = {
        "model": args.model,
        "dataset_dir": str(dataset_dir),
        "fold": str(args.fold),
        "split": args.split,
        "num_samples": len(rows),
        "input_normalization": "independent per-slice z-score",
        "input_size": TARGET_SIZE,
        "patch_size": patch_size,
        "sampling": "approximately uniform over cases with one random lesion slice per draw",
        "metrics": {field: _summary(rows, field) for field in metric_fields},
        "paired_tumor_vs_non_tumor_signal": _paired_signal_summary(rows, args.seed),
    }
    report_path = output.with_suffix(".summary.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report["metrics"], indent=2))
    print("\nPaired tumor vs non-tumor signal:")
    print(json.dumps(report["paired_tumor_vs_non_tumor_signal"], indent=2))
    print(f"\nPer-slice results: {output}")
    print(f"Summary: {report_path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure Curia-2 patch-token differences between real MamaSynth pre/post lesion slices."
    )
    parser.add_argument(
        "dataset_dir",
        help="MamaSynth blosc2 dataset containing manifest.csv and splits_final.json",
    )
    parser.add_argument("--fold", default="0")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--model", default=CURIA_MODEL_ID)
    parser.add_argument("--output", default="curia2_real_pre_post_distances.csv")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.num_samples < 2:
        raise ValueError(
            "--num-samples must be at least 2 for the shuffled-post reference."
        )
    if args.batch_size < 2:
        raise ValueError(
            "--batch-size must be at least 2 for the shuffled-post reference."
        )
    assess(args)


if __name__ == "__main__":
    main()
