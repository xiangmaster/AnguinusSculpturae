from __future__ import annotations

import argparse
import os
from pathlib import Path

# Both fall back to the environment variables this launcher itself exports to the
# trainers, so a shell that already has them set needs no flags. Unset means the
# matching flag is required.
DEFAULT_PRE_STATS = os.environ.get("MAMASYNTH_PRE_STATS_PATH")
DEFAULT_SEG_MODEL_DIR = os.environ.get("MAMASYNTH_SEGMENTATION_MODEL_DIR")


def _run_training_mamasynth(
    default_trainer: str, description: str, translation: bool = False
):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Path to the blosc2 dataset directory (e.g. .../Dataset625_Pre_Seg).",
    )
    parser.add_argument(
        "fold", type=str, help='Fold of the cross-validation, or "all".'
    )
    parser.add_argument(
        "--split-file",
        type=str,
        default=None,
        help="Path to an nnU-Net train/val split JSON. Default: <dataset_dir>/splits_final.json.",
    )
    parser.add_argument(
        "-tr",
        type=str,
        default=default_trainer,
        help=f"Trainer class. Default: {default_trainer}.",
    )
    parser.add_argument(
        "-num_epochs",
        type=int,
        default=None,
        help="[OPTIONAL] Override number of epochs.",
    )
    parser.add_argument(
        "-device", type=str, default="cuda", help="cuda | cpu | mps. Default: cuda."
    )
    parser.add_argument(
        "--c", action="store_true", help="Continue training from the latest checkpoint."
    )
    parser.add_argument(
        "-pretrained_weights",
        type=str,
        default=None,
        help="[OPTIONAL] Path to a checkpoint to initialize the network from.",
    )
    if translation:
        parser.add_argument(
            "--pre-stats",
            type=str,
            default=DEFAULT_PRE_STATS,
            required=DEFAULT_PRE_STATS is None,
            help="Global pre-contrast stats JSON used to normalize pre and post. "
            "Defaults to $MAMASYNTH_PRE_STATS_PATH when that is set.",
        )
        parser.add_argument(
            "--seg-model-dir",
            type=str,
            default=DEFAULT_SEG_MODEL_DIR,
            required=DEFAULT_SEG_MODEL_DIR is None,
            help="Folder containing the chained segmentation nnU-Net. You can pass either "
            "the models root or the segmentation subfolder. Defaults to "
            "$MAMASYNTH_SEGMENTATION_MODEL_DIR when that is set.",
        )
        parser.add_argument(
            "--pre-seg-model-dir",
            type=str,
            default=None,
            help="Folder with the frozen PRE-contrast segmentation nnU-Net (plans.json/dataset.json/"
            "fold_0) whose lesion probability is fed as an extra input channel by the *_SegChannel "
            "trainers. Models root or the segmentation subfolder. Default: reuse --seg-model-dir.",
        )
        parser.add_argument(
            "--pre-seg-checkpoint-name",
            type=str,
            default=None,
            help="Checkpoint filename inside <pre-seg-model-dir>/fold_0 for the *_SegChannel trainers. "
            "Default: checkpoint_final.pth.",
        )
        parser.add_argument(
            "--foreground-dir",
            type=str,
            default=None,
            help="Folder with the per-case foreground masks (e.g. .../Dataset627_foreground, holding "
            "<case_id>_foreground_mask.b2nd) consumed by the *_ForegroundLoss trainers, which restrict "
            "the loss to the tissue foreground. The pre/post images still come from the positional "
            "dataset_dir. Default: unset (required only by foreground-loss trainers).",
        )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset directory does not exist: {dataset_dir}")

    if args.split_file:
        split_file = Path(args.split_file).resolve()
    elif (dataset_dir / "splits_final.json").is_file():
        split_file = dataset_dir / "splits_final.json"
    else:
        split_file = dataset_dir.parent / "splits_final.json"
    if not split_file.is_file():
        raise FileNotFoundError(f"Split JSON does not exist: {split_file}")

    # Must be set before importing anything that reads nnunetv2.paths or the MamaSynth split.
    os.environ["MAMASYNTH_DATASET_DIR"] = str(dataset_dir)
    os.environ["MAMASYNTH_SPLIT_FILE"] = str(split_file)
    if translation:
        pre_stats = Path(args.pre_stats).resolve()
        if not pre_stats.is_file():
            raise FileNotFoundError(f"Pre stats JSON does not exist: {pre_stats}")
        os.environ["MAMASYNTH_PRE_STATS_PATH"] = str(pre_stats)
        seg_model_dir = Path(args.seg_model_dir).resolve()
        if (seg_model_dir / "segmentation").is_dir() and not (
            seg_model_dir / "plans.json"
        ).is_file():
            seg_model_dir = seg_model_dir / "segmentation"
        if (
            not (seg_model_dir / "plans.json").is_file()
            or not (seg_model_dir / "dataset.json").is_file()
        ):
            raise FileNotFoundError(
                f"Segmentation model folder must contain plans.json and dataset.json: {seg_model_dir}"
            )
        os.environ["MAMASYNTH_SEGMENTATION_MODEL_DIR"] = str(seg_model_dir)
        # Optional dedicated PRE-contrast segmenter for *_SegChannel trainers (extra input channel).
        if args.pre_seg_model_dir is not None:
            pre_seg_model_dir = Path(args.pre_seg_model_dir).resolve()
            if (pre_seg_model_dir / "segmentation").is_dir() and not (
                pre_seg_model_dir / "plans.json"
            ).is_file():
                pre_seg_model_dir = pre_seg_model_dir / "segmentation"
            if (
                not (pre_seg_model_dir / "plans.json").is_file()
                or not (pre_seg_model_dir / "dataset.json").is_file()
            ):
                raise FileNotFoundError(
                    f"Pre-segmentation model folder must contain plans.json and dataset.json: "
                    f"{pre_seg_model_dir}"
                )
            os.environ["MAMASYNTH_PRE_SEG_MODEL_DIR"] = str(pre_seg_model_dir)
        if args.pre_seg_checkpoint_name is not None:
            os.environ["MAMASYNTH_PRE_SEG_CHECKPOINT"] = args.pre_seg_checkpoint_name
        # Optional foreground-mask dataset for the *_ForegroundLoss trainers (loss restricted to tissue).
        if args.foreground_dir is not None:
            foreground_dir = Path(args.foreground_dir).resolve()
            if not foreground_dir.is_dir():
                raise NotADirectoryError(
                    f"Foreground dataset directory does not exist: {foreground_dir}"
                )
            os.environ["MAMASYNTH_FOREGROUND_DIR"] = str(foreground_dir)
    if "nnUNet_results" not in os.environ:
        default_results = str(dataset_dir.parent / "nnUNet_results")
        os.environ["nnUNet_results"] = default_results
        print(f"INFO: nnUNet_results was not set; defaulting to {default_results}")

    import torch

    from nnunetv2.run.run_training import maybe_load_checkpoint
    from nnunetv2.training.mamasynth.dataset import load_split
    from nnunetv2.training.mamasynth.plans import (
        build_plans_and_dataset_json,
        dataset_name_from_dir,
    )
    from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name

    fold = args.fold if args.fold == "all" else int(args.fold)
    device = torch.device(args.device)

    num_training = len(load_split(dataset_dir, fold)["train"])
    plans, dataset_json = build_plans_and_dataset_json(
        dataset_name_from_dir(dataset_dir), num_training
    )
    plans["continue_training"] = args.c

    trainer_class = recursive_find_trainer_class_by_name(args.tr)
    if trainer_class is None:
        raise RuntimeError(f"Could not find trainer class {args.tr}.")
    trainer = trainer_class(plans, "2d", fold, dataset_json, device=device)
    if args.num_epochs is not None:
        trainer.num_epochs = args.num_epochs

    maybe_load_checkpoint(trainer, args.c, False, args.pretrained_weights)
    trainer.run_training()
    print(f"Training finished. Results in {trainer.output_folder}")


def run_training_mamasynth_entry():
    _run_training_mamasynth(
        "nnUNetTrainerMamaSynth",
        "Train an nnU-Net / Curia model directly on the unified (unpreprocessed) MamaSynth blosc2 dataset. "
        "Slices are resized to 512x512 and z-scored individually on the fly; results are stored under "
        "$nnUNet_results/<dataset folder name>/ as usual.",
        translation=False,
    )


def run_training_mamasynth_translate_entry():
    _run_training_mamasynth(
        "nnUNetTrainerMamaSynthTranslation",
        "Train a MamaSynth pre-to-post image translation model directly on the unified blosc2 dataset. "
        "Pre and post slices use shared global pre-contrast normalization stats; the network predicts post-pre delta.",
        translation=True,
    )


def run_training_mamasynth_diffusion_entry():
    _run_training_mamasynth(
        "nnUNetTrainerMamaSynthDiffusion",
        "Train a MamaSynth conditional residual-diffusion (DDPM) model on the unified blosc2 dataset. "
        "The network denoises the (normalized) post-pre residual conditioned on the pre-contrast slice "
        "(epsilon-prediction, cosine schedule, FiLM time conditioning); samples are drawn with DDIM. "
        "Uses the same translation plumbing (pre stats, fixed validation, checkpointing) as "
        "nnUNetv2_train_mamasynth_translate; pass any diffusion trainer via -tr.",
        translation=True,
    )


def run_training_mamasynth_mae_entry():
    _run_training_mamasynth(
        "nnUNetTrainerMamaSynthMAE",
        "MAE-pretrain the MamaSynth ResEnc U-Net on the unlabeled blosc2 dataset (Dataset626_MAE). "
        "Image slices are resized to 512x512 and z-scored per slice; the network reconstructs "
        "block-masked inputs. Use the resulting checkpoint as -pretrained_weights for "
        "nnUNetv2_train_mamasynth_translate.",
    )


def run_training_mamasynth_foreground_entry():
    _run_training_mamasynth(
        "nnUNetTrainerMamaSynthForeground",
        "Train a MamaSynth 2D nnU-Net foreground-support segmenter from pre-contrast slices "
        "and Dataset627_foreground binary foreground masks.",
    )


if __name__ == "__main__":
    run_training_mamasynth_entry()
