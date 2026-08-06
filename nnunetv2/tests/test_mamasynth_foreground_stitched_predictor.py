import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from nnunetv2.inference import predict_mamasynth_translate_foreground_stitched as fg


class TestForegroundStitchedCompositing(unittest.TestCase):
    def test_hard_masks_choose_breast_model_then_restore_pre_background(self):
        pre = np.zeros((2, 2), dtype=np.float32)
        lesion = np.full((2, 2), 10, dtype=np.float32)
        image = np.full((2, 2), 2, dtype=np.float32)
        breast = np.array([[1, 0], [1, 0]], dtype=np.uint8)
        foreground = np.array([[1, 1], [0, 0]], dtype=np.uint8)

        final, stitched, breast_soft, foreground_soft = (
            fg.composite_foreground_stitched_slice(
                pre, lesion, image, breast, foreground, feather=0
            )
        )

        np.testing.assert_array_equal(
            stitched, np.array([[10, 2], [10, 2]], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            final, np.array([[10, 2], [0, 0]], dtype=np.float32)
        )
        np.testing.assert_array_equal(breast_soft, breast.astype(np.float32))
        np.testing.assert_array_equal(foreground_soft, foreground.astype(np.float32))

    def test_foreground_background_uses_original_pre_values(self):
        pre = np.array([[1, 2], [3, 4]], dtype=np.float32)
        lesion = np.full((2, 2), 10, dtype=np.float32)
        image = np.full((2, 2), 20, dtype=np.float32)
        breast = np.ones((2, 2), dtype=np.uint8)
        foreground = np.array([[1, 0], [0, 1]], dtype=np.uint8)

        final, _, _, _ = fg.composite_foreground_stitched_slice(
            pre, lesion, image, breast, foreground, feather=0
        )

        np.testing.assert_array_equal(
            final, np.array([[10, 2], [3, 10]], dtype=np.float32)
        )


class TestCheckpointResolver(unittest.TestCase):
    def test_accepts_direct_pth_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "model.pth"
            checkpoint.touch()

            self.assertEqual(
                fg.resolve_checkpoint(checkpoint, None), checkpoint.resolve()
            )

    def test_accepts_folder_and_default_checkpoint_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            checkpoint = folder / "checkpoint_best.pth"
            checkpoint.touch()

            self.assertEqual(
                fg.resolve_checkpoint(folder, "checkpoint_best.pth"),
                checkpoint.resolve(),
            )

    def test_folder_requires_explicit_checkpoint_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                fg.resolve_checkpoint(Path(tmp), None, "--image-weights")

    def test_errors_when_folder_checkpoint_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                fg.resolve_checkpoint(Path(tmp), "checkpoint_best.pth")


class TestForegroundStitchedCli(unittest.TestCase):
    def _make_input_and_weights(self, root: Path):
        import tifffile

        input_dir = root / "input"
        output_dir = root / "output"
        input_dir.mkdir()
        image = np.arange(16, dtype=np.float32).reshape(4, 4)
        tifffile.imwrite(input_dir / "case.tif", image)

        weight_names = [
            ("image", "checkpoint_best.pth"),
            ("lesion", "checkpoint_best_dice.pth"),
            ("foreground", "checkpoint_best.pth"),
            ("breast", "checkpoint_final.pth"),
        ]
        checkpoints = {}
        for name, checkpoint_name in weight_names:
            folder = root / name
            folder.mkdir()
            checkpoint = folder / checkpoint_name
            checkpoint.touch()
            checkpoints[name] = checkpoint
        return input_dir, output_dir, checkpoints

    def test_cli_writes_final_only_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir, output_dir, checkpoints = self._make_input_and_weights(root)

            def fake_predict(*args, **kwargs):
                return np.ones((4, 4), dtype=np.float32)

            argv = [
                "prog",
                "-i",
                str(input_dir),
                "-o",
                str(output_dir),
                "--image-weights",
                str(checkpoints["image"]),
                "--lesion-image-weights",
                str(checkpoints["lesion"]),
                "--foreground-weights",
                str(checkpoints["foreground"]),
                "--breast-weights",
                str(checkpoints["breast"]),
                "-device",
                "cpu",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    fg, "load_translation_network", return_value=(object(), 1)
                ),
                patch.object(fg, "load_seg_network", return_value=(object(), 2)),
                patch.object(
                    fg, "predict_foreground_stitched_array", side_effect=fake_predict
                ),
            ):
                fg.main()

            self.assertEqual(
                sorted(path.name for path in output_dir.iterdir()), ["case.tif"]
            )

    def test_cli_writes_requested_masks_and_intermediates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir, output_dir, checkpoints = self._make_input_and_weights(root)

            def fake_predict(*args, **kwargs):
                final = np.ones((4, 4), dtype=np.float32)
                extras = {
                    "breast_mask": np.ones((4, 4), dtype=np.uint8),
                    "foreground_mask": np.ones((4, 4), dtype=np.uint8),
                    "breast_soft": np.ones((4, 4), dtype=np.float32),
                    "foreground_soft": np.ones((4, 4), dtype=np.float32),
                    "lesion_synth": np.ones((4, 4), dtype=np.float32),
                    "image_synth": np.ones((4, 4), dtype=np.float32),
                    "stitched": np.ones((4, 4), dtype=np.float32),
                }
                return final, extras

            argv = [
                "prog",
                "-i",
                str(input_dir),
                "-o",
                str(output_dir),
                "--image-weights",
                str(checkpoints["image"]),
                "--lesion-image-weights",
                str(checkpoints["lesion"]),
                "--foreground-weights",
                str(checkpoints["foreground"]),
                "--breast-weights",
                str(checkpoints["breast"]),
                "--save-masks",
                "--save-intermediates",
                "-device",
                "cpu",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    fg, "load_translation_network", return_value=(object(), 1)
                ),
                patch.object(fg, "load_seg_network", return_value=(object(), 2)),
                patch.object(
                    fg, "predict_foreground_stitched_array", side_effect=fake_predict
                ),
            ):
                fg.main()

            expected = {
                "case.tif",
                "case_breast_mask.tif",
                "case_foreground_mask.tif",
                "case_breast_soft.tif",
                "case_foreground_soft.tif",
                "case_lesion_synth.tif",
                "case_image_synth.tif",
                "case_stitched.tif",
            }
            self.assertEqual(set(path.name for path in output_dir.iterdir()), expected)


if __name__ == "__main__":
    unittest.main()
