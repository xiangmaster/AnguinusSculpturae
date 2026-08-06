import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from nnunetv2.training.mamasynth.dataset import MamaSynthCase
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerMamaSynthForeground import (
    MamaSynthForegroundSliceDataLoader,
    foreground_slice_indices,
    load_foreground_cases,
    load_pre_and_foreground_mask_slice,
    nnUNetTrainerMamaSynthForeground,
)
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name


def _case(case_id: str = "case") -> MamaSynthCase:
    return MamaSynthCase(
        case_id=case_id,
        n_slices=3,
        spacing_y=0.70,
        spacing_x=0.70,
        pre_path=Path(f"{case_id}_pre.b2nd"),
        post_path=Path(f"{case_id}_pre.b2nd"),
        mask_path=Path(f"{case_id}_foreground_mask.b2nd"),
    )


class TestMamaSynthForegroundData(unittest.TestCase):
    def tearDown(self):
        foreground_slice_indices.cache_clear()

    def test_load_foreground_cases_accepts_resumed_manifest_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "case_pre.b2nd").touch()
            (root / "case_foreground_mask.b2nd").touch()
            (root / "manifest.csv").write_text(
                "status,case_id,n_slices,spacing_3d,source_pre_b2nd,output_path\n"
                'skipped_existing,case,5,"[0.7, 0.8, 2.0]",,\n'
                'failed,bad,5,"[0.7, 0.8, 2.0]",,\n',
                encoding="utf-8",
            )

            cases = load_foreground_cases(root)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].case_id, "case")
        self.assertEqual(cases[0].n_slices, 5)
        self.assertEqual(cases[0].spacing_y, 0.7)
        self.assertEqual(cases[0].spacing_x, 0.8)

    def test_foreground_slice_indices_use_any_positive_mask_value(self):
        mask = np.zeros((3, 4, 4), dtype=np.uint8)
        mask[0, 1, 1] = 1
        mask[2, 2, 2] = 255

        with patch(
            "nnunetv2.training.nnUNetTrainer.variants.network_architecture."
            "nnUNetTrainerMamaSynthForeground.open_b2nd",
            return_value=mask,
        ):
            self.assertEqual(foreground_slice_indices("mask.b2nd"), (0, 2))

    def test_slice_loader_maps_positive_values_to_binary_foreground(self):
        pre = np.arange(16, dtype=np.float32).reshape(4, 4)
        mask = np.array(
            [
                [0, 1, 2, 0],
                [0, 0, 0, 0],
                [255, 1, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )

        def open_volume(path):
            return np.stack([pre] * 3) if "pre" in str(path) else np.stack([mask] * 3)

        with (
            patch(
                "nnunetv2.training.nnUNetTrainer.variants.network_architecture."
                "nnUNetTrainerMamaSynthForeground.open_b2nd",
                side_effect=open_volume,
            ),
            patch(
                "nnunetv2.training.nnUNetTrainer.variants.network_architecture."
                "nnUNetTrainerMamaSynthForeground.TARGET_SIZE",
                4,
            ),
            patch("nnunetv2.training.mamasynth.dataset.TARGET_SIZE", 4),
        ):
            image, target = load_pre_and_foreground_mask_slice(_case(), 0)

        self.assertEqual(image.shape, (4, 4))
        np.testing.assert_array_equal(target, (mask > 0).astype(np.int16))
        self.assertEqual(set(np.unique(target)), {0, 1})

    def test_dataloader_returns_only_positive_binary_targets(self):
        case = _case()
        positive_mask = np.zeros((512, 512), dtype=np.int16)
        positive_mask[100:200, 100:200] = 1

        with (
            patch(
                "nnunetv2.training.nnUNetTrainer.variants.network_architecture."
                "nnUNetTrainerMamaSynthForeground.foreground_slice_indices",
                return_value=(0, 2),
            ),
            patch(
                "nnunetv2.training.nnUNetTrainer.variants.network_architecture."
                "nnUNetTrainerMamaSynthForeground.load_pre_and_foreground_mask_slice",
                return_value=(np.zeros((512, 512), dtype=np.float32), positive_mask),
            ),
        ):
            loader = MamaSynthForegroundSliceDataLoader([case], batch_size=2)
            batch = loader.generate_train_batch()

        self.assertEqual(batch["data"].shape, (2, 1, 512, 512))
        self.assertEqual(batch["target"].shape, (2, 1, 512, 512))
        self.assertTrue(torch.all(batch["target"].sum(dim=(1, 2, 3)) > 0))
        self.assertEqual(set(torch.unique(batch["target"]).tolist()), {0, 1})
        self.assertTrue(all(key in {"case_s0", "case_s2"} for key in batch["keys"]))


class TestMamaSynthForegroundTrainer(unittest.TestCase):
    def test_trainer_is_discoverable(self):
        trainer_class = recursive_find_trainer_class_by_name(
            "nnUNetTrainerMamaSynthForeground"
        )
        self.assertIs(trainer_class, nnUNetTrainerMamaSynthForeground)


if __name__ == "__main__":
    unittest.main()
