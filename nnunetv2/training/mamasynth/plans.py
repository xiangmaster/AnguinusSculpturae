from __future__ import annotations

from pathlib import Path
from typing import Tuple

# A self-contained 2D plan for the MamaSynth dataset. We bypass nnU-Net's planning/preprocessing
# entirely (slices are loaded and normalized on the fly), so this hand-written plan only needs to
# carry what the trainer/transforms actually read: patch size, batch size, normalization flags and a
# valid network architecture (a standard nnU-Net ResEnc-M UNet for 512x512, used by the baseline; the
# Curia trainers ignore the architecture and build their own).

PATCH_SIZE = [512, 512]
BATCH_SIZE = 4

_RESENC_M_2D = {
    "network_class_name": "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet",
    "arch_kwargs": {
        "n_stages": 7,
        "features_per_stage": [32, 64, 128, 256, 512, 512, 512],
        "conv_op": "torch.nn.modules.conv.Conv2d",
        "kernel_sizes": [[3, 3]] * 7,
        "strides": [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
        "n_blocks_per_stage": [1, 3, 4, 6, 6, 6, 6],
        "n_conv_per_stage_decoder": [1, 1, 1, 1, 1, 1],
        "conv_bias": True,
        "norm_op": "torch.nn.modules.instancenorm.InstanceNorm2d",
        "norm_op_kwargs": {"eps": 1e-05, "affine": True},
        "dropout_op": None,
        "dropout_op_kwargs": None,
        "nonlin": "torch.nn.LeakyReLU",
        "nonlin_kwargs": {"inplace": True},
    },
    "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"],
}


def build_plans_and_dataset_json(
    dataset_name: str, num_training: int
) -> Tuple[dict, dict]:
    plans = {
        "dataset_name": dataset_name,
        "plans_name": "mamaSynthPlans",
        "original_median_spacing_after_transp": [1.0, 1.0, 1.0],
        "original_median_shape_after_transp": [512, 512, 512],
        "image_reader_writer": "SimpleITKIO",
        "transpose_forward": [0, 1, 2],
        "transpose_backward": [0, 1, 2],
        "experiment_planner_used": "ExperimentPlanner",
        "label_manager": "LabelManager",
        "foreground_intensity_properties_per_channel": {
            "0": {
                "mean": 0.0,
                "std": 1.0,
                "median": 0.0,
                "min": 0.0,
                "max": 1.0,
                "percentile_00_5": 0.0,
                "percentile_99_5": 1.0,
            }
        },
        "configurations": {
            "2d": {
                "data_identifier": "mamaSynthPlans_2d",
                "preprocessor_name": "DefaultPreprocessor",
                "batch_size": BATCH_SIZE,
                "patch_size": PATCH_SIZE,
                "median_image_size_in_voxels": PATCH_SIZE,
                "spacing": [1.0, 1.0],
                "normalization_schemes": ["ZScoreNormalization"],
                "use_mask_for_norm": [False],
                "batch_dice": True,
                "architecture": _RESENC_M_2D,
            }
        },
    }
    dataset_json = {
        "channel_names": {"0": "PRE"},
        # Lesion-only segmentation: breast (raw mask label 2) is folded into background in the dataloader.
        "labels": {"background": 0, "lesion": 1},
        "numTraining": int(num_training),
        "file_ending": ".b2nd",
    }
    return plans, dataset_json


def dataset_name_from_dir(dataset_dir: Path) -> str:
    return Path(dataset_dir).name
