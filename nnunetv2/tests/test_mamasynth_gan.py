import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from nnunetv2.training.logging.nnunet_logger import LocalLogger
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.mamasynth.data_loader import (
    MamaSynthTranslationFixedValDataLoader,
)
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerMamaSynth import (
    MamaSynthPatchDiscriminator,
    MamaSynthResizeConvUNetDecoder,
    nnUNetTrainerMamaSynthTranslationLPIPSCuriaTokenMSE_LPIPS05_Curia05_BS_48,
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDice_LPIPS05_BS_48,
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuria_LPIPS05_Curia05_BS_48,
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuria_LPIPS05_Curia05_DS256128_BS_48,
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuria_LPIPS05_Curia05_ResizeConv_BS_48,
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuria_LPIPS10_Curia05_BS_48,
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuriaGAN,
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuriaGAN_LPIPS05_Curia05_GAN02_BS_48,
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceTverskyA02B08_LPIPS05_BS_48,
    nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceTverskyA02B08DilatedGT_LPIPS05_BS_48,
)
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name


class _CheckpointLogger:
    def __init__(self):
        self.local_logger = SimpleNamespace(
            my_fantastic_logging={
                "val_objective_losses": [2.0],
                "val_total_objectives": [2.0],
                "val_total_objective_emas": [2.0],
            }
        )

    def get_checkpoint(self):
        return {
            key: list(value)
            for key, value in self.local_logger.my_fantastic_logging.items()
        }

    def load_checkpoint(self, state):
        self.local_logger.my_fantastic_logging = {
            key: list(value) for key, value in state.items()
        }

    def get_value(self, key, step=None):
        values = self.local_logger.my_fantastic_logging.get(key, [])
        return values if step is None else values[step]

    def log(self, key, value, epoch):
        self.local_logger.my_fantastic_logging.setdefault(key, []).append(value)


class _CheckpointScaler:
    def __init__(self, value):
        self.value = value

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state):
        self.value = state["value"]


class _TinyDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Conv2d(2, 1, 1)

    def forward(self, value):
        return self.network(value)


def _bare_trainer():
    trainer = object.__new__(nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuriaGAN)
    trainer.device = torch.device("cpu")
    trainer.current_epoch = 0
    trainer.discriminator = _TinyDiscriminator()
    trainer.discriminator_optimizer = torch.optim.Adam(
        trainer.discriminator.parameters(), lr=2.5e-4, betas=(0.5, 0.999)
    )
    trainer.discriminator_grad_scaler = None
    trainer.adversarial_loss = nn.BCEWithLogitsLoss()
    return trainer


class TestMamaSynthGAN(unittest.TestCase):
    def test_patchgan_shape_capacity_and_regularization(self):
        discriminator = MamaSynthPatchDiscriminator()
        with torch.no_grad():
            output = discriminator(torch.randn(1, 2, 512, 512))
        self.assertEqual(tuple(output.shape), (1, 1, 62, 62))
        convolutions = [
            module
            for module in discriminator.modules()
            if isinstance(module, nn.Conv2d)
        ]
        self.assertEqual(
            [module.out_channels for module in convolutions], [32, 64, 128, 256, 1]
        )
        self.assertTrue(all(hasattr(module, "weight_orig") for module in convolutions))
        dropouts = [
            module
            for module in discriminator.modules()
            if isinstance(module, nn.Dropout2d)
        ]
        self.assertEqual(len(dropouts), 4)
        self.assertTrue(all(module.p == 0.3 for module in dropouts))

    def test_reconstruction_mask_includes_acquired_background_and_excludes_padding(
        self,
    ):
        prediction = torch.tensor([[[[2.0, 3.0, 100.0]]]])
        target = torch.zeros_like(prediction)
        acquired_image_mask = torch.tensor([[[[1.0, 1.0, 0.0]]]])

        loss = nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuriaGAN._masked_mse(
            prediction, target, acquired_image_mask
        )

        self.assertEqual(float(loss), 6.5)

    def test_masking_noise_decay_and_labels(self):
        trainer = _bare_trainer()
        image = torch.ones(1, 1, 4, 4)
        acquired_image_mask = torch.zeros_like(image)
        acquired_image_mask[:, :, :2] = 1
        torch.manual_seed(0)
        discriminator_input = trainer._discriminator_input(
            image, acquired_image_mask, noise_std=0.1
        )
        self.assertEqual(tuple(discriminator_input.shape), (1, 2, 4, 4))
        self.assertTrue(
            torch.equal(discriminator_input[:, :1, 2:], torch.zeros(1, 1, 2, 4))
        )
        self.assertTrue(torch.equal(discriminator_input[:, 1:], acquired_image_mask))
        self.assertFalse(torch.equal(discriminator_input[:, :1, :2], image[:, :, :2]))
        self.assertEqual(trainer.discriminator_real_label, 0.9)
        self.assertEqual(trainer.gan_weight, 0.01)
        self.assertAlmostEqual(trainer._discriminator_noise_level(), 0.1)
        trainer.current_epoch = 12.5
        self.assertAlmostEqual(trainer._discriminator_noise_level(), 0.05)
        trainer.current_epoch = 25
        self.assertEqual(trainer._discriminator_noise_level(), 0.0)

    def test_discriminator_and_generator_gradients_are_isolated(self):
        trainer = _bare_trainer()
        prediction = torch.randn(1, 1, 8, 8, requires_grad=True)
        target = torch.randn_like(prediction)
        valid = torch.ones_like(prediction)
        trainer._discriminator_step(prediction, target, valid)
        self.assertIsNone(prediction.grad)

        trainer.discriminator_optimizer.zero_grad(set_to_none=True)
        trainer._set_requires_grad(trainer.discriminator, False)
        gan_loss = trainer._generator_gan_loss(prediction, valid)
        gan_loss.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in trainer.discriminator.parameters()
            )
        )

    def test_generator_pass_does_not_change_spectral_norm_state(self):
        trainer = _bare_trainer()
        trainer.discriminator = MamaSynthPatchDiscriminator(base_channels=4)
        trainer.discriminator.train()
        trainer._set_requires_grad(trainer.discriminator, False)
        prediction = torch.randn(1, 1, 64, 64, requires_grad=True)
        valid = torch.ones_like(prediction)
        before = {
            key: value.clone()
            for key, value in trainer.discriminator.state_dict().items()
        }
        trainer._generator_gan_loss(prediction, valid).backward()
        self.assertTrue(trainer.discriminator.training)
        for key, value in before.items():
            self.assertTrue(
                torch.equal(trainer.discriminator.state_dict()[key], value), key
            )

    def test_train_step_adds_weighted_gan_loss(self):
        trainer = _bare_trainer()
        trainer.network = nn.Conv2d(1, 1, 1)
        trainer.optimizer = torch.optim.AdamW(trainer.network.parameters(), lr=1e-3)
        trainer.grad_scaler = None
        trainer.lpips_weight = 1.0
        trainer.ssim_weight = 1.0
        trainer.dice_weight = 1.0
        trainer.curia_weight = 0.2
        trainer._masked_mse = (
            lambda prediction, target, mask: prediction.mean() * 0 + 1.0
        )
        trainer._lpips_loss = (
            lambda prediction, target, mask: prediction.mean() * 0 + 2.0
        )
        trainer._ssim_loss = (
            lambda prediction, target, mask: prediction.mean() * 0 + 3.0
        )
        trainer._segmentation_dice_loss = (
            lambda prediction, tumor, valid: prediction.mean() * 0 + 4.0
        )
        trainer._curia_feature_loss = (
            lambda prediction, target, tumor, valid: prediction.mean() * 0 + 5.0
        )
        batch = {
            "data": torch.randn(1, 1, 8, 8),
            "target": torch.randn(1, 1, 8, 8),
            "acquired_image_mask": torch.ones(1, 1, 8, 8),
            "lesion_mask": torch.ones(1, 1, 8, 8),
        }
        output = trainer.train_step(batch)
        expected = 1.0 + 2.0 + 3.0 + 4.0 + 0.2 * 5.0 + 0.01 * float(output["gan"])
        self.assertAlmostEqual(float(output["loss"]), expected, places=5)

    def test_checkpoint_round_trip_restores_gan_state(self):
        trainer = _bare_trainer()
        trainer.network = nn.Conv2d(1, 1, 1)
        trainer.optimizer = torch.optim.AdamW(trainer.network.parameters(), lr=1e-3)
        trainer.grad_scaler = _CheckpointScaler(11)
        trainer.discriminator_grad_scaler = _CheckpointScaler(22)
        trainer.discriminator_lr_scheduler = PolyLRScheduler(
            trainer.discriminator_optimizer, 2.5e-4, 100
        )
        trainer.logger = _CheckpointLogger()
        trainer.local_rank = 0
        trainer.disable_checkpointing = False
        trainer._best_ema = 2.0
        trainer.my_init_kwargs = {"fold": 0}
        trainer.inference_allowed_mirroring_axes = (0, 1)
        trainer.current_epoch = 7

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pth"
            trainer.save_checkpoint(str(checkpoint_path))
            expected_network = {
                key: value.clone()
                for key, value in trainer.network.state_dict().items()
            }
            expected_discriminator = {
                key: value.clone()
                for key, value in trainer.discriminator.state_dict().items()
            }
            with torch.no_grad():
                for parameter in trainer.network.parameters():
                    parameter.add_(10)
                for parameter in trainer.discriminator.parameters():
                    parameter.add_(10)
            trainer.was_initialized = True
            trainer.grad_scaler.value = -1
            trainer.discriminator_grad_scaler.value = -1
            trainer.load_checkpoint(str(checkpoint_path))

        for key, value in expected_network.items():
            self.assertTrue(torch.equal(trainer.network.state_dict()[key], value))
        for key, value in expected_discriminator.items():
            self.assertTrue(torch.equal(trainer.discriminator.state_dict()[key], value))
        self.assertEqual(trainer.current_epoch, 8)
        self.assertEqual(trainer.discriminator_lr_scheduler.max_steps, 100)
        self.assertEqual(trainer.grad_scaler.value, 11)
        self.assertEqual(trainer.discriminator_grad_scaler.value, 22)

    def test_raw_and_ema_best_checkpoints_are_selected_independently(self):
        trainer = object.__new__(nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuriaGAN)
        trainer.validation_objective_ema_decay = 0.8
        trainer.validation_objective_ema = None
        trainer.best_raw_validation_objective = None
        trainer._best_ema = None
        trainer.current_epoch = 0
        trainer.output_folder = "/tmp/test-selection"
        trainer.logger = _CheckpointLogger()
        trainer.logger.local_logger.my_fantastic_logging = {
            "val_total_objective_emas": []
        }
        saved = []
        trainer.save_checkpoint = lambda filename: saved.append(Path(filename).name)
        trainer.print_to_log_file = lambda *args, **kwargs: None

        trainer._update_best_translation_checkpoints(10.0)
        trainer.current_epoch = 1
        trainer._update_best_translation_checkpoints(20.0)
        trainer.current_epoch = 2
        trainer._update_best_translation_checkpoints(9.0)

        self.assertEqual(saved.count("checkpoint_best_raw.pth"), 2)
        self.assertEqual(saved.count("checkpoint_best.pth"), 1)
        self.assertAlmostEqual(trainer.validation_objective_ema, 11.4)
        self.assertEqual(trainer.best_raw_validation_objective, 9.0)
        self.assertEqual(trainer._best_ema, 10.0)

    def test_mse_and_dice_save_ema_best_checkpoints_only(self):
        trainer = object.__new__(nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuriaGAN)
        trainer.validation_objective_ema_decay = 0.8
        trainer.validation_objective_ema = None
        trainer.best_raw_validation_objective = None
        trainer._best_ema = None
        trainer.validation_mse_ema = None
        trainer.best_validation_mse_ema = None
        trainer.validation_dice_ema = None
        trainer.best_validation_dice_ema = None
        trainer.current_epoch = 0
        trainer.output_folder = "/tmp/test-metric-selection"
        trainer.logger = _CheckpointLogger()
        trainer.logger.local_logger.my_fantastic_logging = {
            "val_total_objective_emas": [],
            "val_post_mse_emas": [],
            "val_frozen_segmenter_soft_dice_loss_emas": [],
        }
        saved = []
        trainer.save_checkpoint = lambda filename: saved.append(Path(filename).name)
        trainer.print_to_log_file = lambda *args, **kwargs: None

        trainer._update_best_translation_checkpoints(
            10.0, validation_mse=5.0, validation_dice=0.6
        )
        trainer.current_epoch = 1
        trainer._update_best_translation_checkpoints(
            20.0, validation_mse=6.0, validation_dice=0.7
        )
        trainer.current_epoch = 2
        trainer._update_best_translation_checkpoints(
            9.0, validation_mse=4.0, validation_dice=0.5
        )

        self.assertEqual(saved.count("checkpoint_best_mse.pth"), 2)
        self.assertEqual(saved.count("checkpoint_best_dice.pth"), 2)
        self.assertFalse(any("mse_raw" in name or "dice_raw" in name for name in saved))
        self.assertAlmostEqual(trainer.validation_mse_ema, 4.96)
        self.assertAlmostEqual(trainer.validation_dice_ema, 0.596)

    def test_legacy_metric_history_is_migrated_and_ema_is_backfilled(self):
        trainer = object.__new__(nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuriaGAN)
        trainer.validation_objective_ema_decay = 0.8
        trainer.logger = _CheckpointLogger()
        trainer.logger.local_logger.my_fantastic_logging = {
            "train_losses": [3.0, 2.0],
            "val_losses": [4.0, 3.0],
            "train_baseline_mses": [5.0, 5.0],
            "val_baseline_mses": [6.0, 6.0],
            "train_objective_losses": [7.0, 5.0],
            "val_objective_losses": [8.0, 6.0],
        }

        trainer._ensure_translation_logger_keys()
        metrics = trainer.logger.local_logger.my_fantastic_logging
        self.assertEqual(metrics["val_post_mses"], [4.0, 3.0])
        self.assertEqual(metrics["val_total_objectives"], [8.0, 6.0])
        self.assertEqual(metrics["val_total_objective_emas"], [8.0, 7.6])

    def test_curia_token_mse_metrics_are_registered_for_strict_local_logger(self):
        trainer = object.__new__(
            nnUNetTrainerMamaSynthTranslationLPIPSCuriaTokenMSE_LPIPS05_Curia05_BS_48
        )
        trainer.validation_objective_ema_decay = 0.8
        trainer.current_epoch = 0
        trainer.logger = SimpleNamespace(local_logger=LocalLogger())
        trainer.logger.log = trainer.logger.local_logger.log

        trainer._ensure_translation_logger_keys()
        trainer._log_translation_metric(
            "train_curia_losses", "train_curia2_token_mses", 0.25
        )
        trainer._log_translation_metric(
            "val_curia_losses", "val_curia2_token_mses", 0.5
        )

        metrics = trainer.logger.local_logger.my_fantastic_logging
        self.assertEqual(metrics["train_curia_losses"], [0.25])
        self.assertEqual(metrics["train_curia2_token_mses"], [0.25])
        self.assertEqual(metrics["val_curia_losses"], [0.5])
        self.assertEqual(metrics["val_curia2_token_mses"], [0.5])

    def test_trainer_is_discoverable(self):
        name = "nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuriaGAN"
        self.assertEqual(recursive_find_trainer_class_by_name(name).__name__, name)

    def test_reweighted_bs48_trainer_configuration_and_discovery(self):
        trainer_class = nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuriaGAN_LPIPS05_Curia05_GAN02_BS_48
        self.assertEqual(trainer_class.batch_size_override, 48)
        self.assertEqual(trainer_class.lpips_weight, 0.5)
        self.assertEqual(trainer_class.curia_weight, 0.5)
        self.assertEqual(trainer_class.gan_weight, 0.02)
        self.assertEqual(
            recursive_find_trainer_class_by_name(trainer_class.__name__),
            trainer_class,
        )

    def test_matched_non_gan_ablation_configuration_and_discovery(self):
        trainer_class = (
            nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuria_LPIPS05_Curia05_BS_48
        )
        self.assertEqual(trainer_class.batch_size_override, 48)
        self.assertEqual(trainer_class.lpips_weight, 0.5)
        self.assertEqual(trainer_class.curia_weight, 0.5)
        self.assertFalse(
            issubclass(
                trainer_class, nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuriaGAN
            )
        )
        self.assertEqual(
            recursive_find_trainer_class_by_name(trainer_class.__name__),
            trainer_class,
        )

    def test_resize_conv_decoder_and_trainer_configuration(self):
        original_decoder = nn.Module()
        original_decoder.deep_supervision = False
        original_decoder.encoder = nn.Module()
        original_decoder.encoder.strides = [(1, 1), (2, 2)]
        original_decoder.num_classes = 1
        original_decoder.transpconvs = nn.ModuleList([nn.ConvTranspose2d(8, 4, 2, 2)])
        original_decoder.stages = nn.ModuleList([nn.Conv2d(8, 4, 3, padding=1)])
        original_decoder.seg_layers = nn.ModuleList([nn.Conv2d(4, 1, 1)])

        decoder = MamaSynthResizeConvUNetDecoder(original_decoder)
        output = decoder([torch.randn(1, 4, 15, 17), torch.randn(1, 8, 8, 9)])

        self.assertEqual(tuple(output.shape), (1, 1, 15, 17))
        self.assertFalse(
            any(isinstance(module, nn.ConvTranspose2d) for module in decoder.modules())
        )
        self.assertEqual(decoder.upsample_convs[0].kernel_size, (3, 3))
        trainer_class = nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuria_LPIPS05_Curia05_ResizeConv_BS_48
        self.assertEqual(trainer_class.batch_size_override, 48)
        self.assertEqual(trainer_class.lpips_weight, 0.5)
        self.assertEqual(trainer_class.curia_weight, 0.5)
        self.assertIsInstance(
            trainer_class.__dict__["build_network_architecture"], staticmethod
        )
        self.assertEqual(
            recursive_find_trainer_class_by_name(trainer_class.__name__), trainer_class
        )

    def test_multiscale_reconstruction_trainer_configuration_and_discovery(self):
        trainer_class = nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuria_LPIPS05_Curia05_DS256128_BS_48
        self.assertEqual(trainer_class.batch_size_override, 48)
        self.assertEqual(trainer_class.lpips_weight, 0.5)
        self.assertEqual(trainer_class.curia_weight, 0.5)
        self.assertEqual(trainer_class.auxiliary_mse_256_weight, 0.25)
        self.assertEqual(trainer_class.auxiliary_mse_128_weight, 0.125)
        self.assertEqual(
            recursive_find_trainer_class_by_name(trainer_class.__name__),
            trainer_class,
        )

    def test_multiscale_mse_uses_largest_first_outputs_and_fractional_masks(self):
        trainer = object.__new__(
            nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuria_LPIPS05_Curia05_DS256128_BS_48
        )
        data = torch.zeros(1, 1, 8, 8)
        target_post = torch.ones_like(data)
        valid = torch.zeros_like(data)
        valid[:, :, :6] = 1
        outputs = [
            torch.full_like(data, 3.0),
            torch.zeros(1, 1, 4, 4),
            torch.zeros(1, 1, 2, 2),
        ]

        full_output, mse_256, mse_128 = trainer._multiscale_mse_losses(
            outputs, data, target_post, valid
        )

        self.assertIs(full_output, outputs[0])
        self.assertAlmostEqual(float(mse_256), 1.0)
        self.assertAlmostEqual(float(mse_128), 1.0)
        with self.assertRaisesRegex(RuntimeError, "512, 256, and 128"):
            trainer._multiscale_mse_losses(outputs[:2], data, target_post, valid)

    def test_multiscale_total_objective_adds_weighted_auxiliary_mse(self):
        trainer = object.__new__(
            nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuria_LPIPS05_Curia05_DS256128_BS_48
        )
        trainer.network = lambda data: [
            torch.zeros_like(data),
            torch.zeros(data.shape[0], 1, 4, 4),
            torch.zeros(data.shape[0], 1, 2, 2),
        ]
        trainer._lpips_loss = (
            lambda prediction, target, mask: prediction.mean() * 0 + 2.0
        )
        trainer._ssim_loss = (
            lambda prediction, target, mask: prediction.mean() * 0 + 3.0
        )
        trainer._segmentation_dice_loss = (
            lambda prediction, tumor, valid: prediction.mean() * 0 + 4.0
        )
        trainer._curia_feature_loss = (
            lambda prediction, target, tumor, valid: prediction.mean() * 0 + 5.0
        )
        data = torch.zeros(1, 1, 8, 8)
        target = torch.ones_like(data)
        valid = torch.ones_like(data)

        losses = trainer._forward_multiscale_losses(data, target, valid, valid)

        expected = 1.0 + 0.5 * 2.0 + 3.0 + 4.0 + 0.5 * 5.0 + 0.25 * 1.0 + 0.125 * 1.0
        self.assertAlmostEqual(float(losses["loss"]), expected)

    def test_curia_ablation_configuration_and_discovery(self):
        trainer_class = nnUNetTrainerMamaSynthTranslationLPIPSSSIMDice_LPIPS05_BS_48
        self.assertEqual(trainer_class.batch_size_override, 48)
        self.assertEqual(trainer_class.lpips_weight, 0.5)
        self.assertFalse(hasattr(trainer_class, "curia_weight"))
        self.assertEqual(
            recursive_find_trainer_class_by_name(trainer_class.__name__),
            trainer_class,
        )

    def test_tversky_variants_configuration_and_discovery(self):
        trainer_class = (
            nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceTverskyA02B08_LPIPS05_BS_48
        )
        self.assertEqual(trainer_class.tversky_alpha, 0.2)
        self.assertEqual(trainer_class.tversky_beta, 0.8)
        self.assertEqual(trainer_class.tversky_weight, 1.0)
        self.assertEqual(trainer_class.lesion_mask_dilation_radius, 0)
        self.assertEqual(
            recursive_find_trainer_class_by_name(trainer_class.__name__), trainer_class
        )

        dilated_class = nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceTverskyA02B08DilatedGT_LPIPS05_BS_48
        self.assertEqual(dilated_class.lesion_mask_dilation_radius, 1)
        self.assertEqual(
            recursive_find_trainer_class_by_name(dilated_class.__name__), dilated_class
        )

    def test_tversky_loss_uses_alpha_for_false_positives_and_beta_for_false_negatives(
        self,
    ):
        trainer = object.__new__(
            nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceTverskyA02B08_LPIPS05_BS_48
        )
        prediction = torch.tensor([[[[0.8, 0.6], [0.4, 0.2]]]])
        target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        trainer._segmentation_probabilities_and_target = lambda *args: (
            prediction,
            target,
        )

        _, tversky_loss = trainer._segmentation_losses(
            prediction, target, torch.ones_like(target)
        )

        expected_index = (1.2 + 1.0) / (1.2 + 0.2 * 0.8 + 0.8 * 0.8 + 1.0)
        self.assertAlmostEqual(float(tversky_loss), 1.0 - expected_index, places=6)

    def test_dilated_tversky_variant_expands_target_by_one_pixel_within_acquired_support(
        self,
    ):
        trainer = object.__new__(
            nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceTverskyA02B08DilatedGT_LPIPS05_BS_48
        )
        lesion = torch.zeros(1, 1, 5, 5)
        lesion[:, :, 2, 2] = 1
        acquired = torch.ones_like(lesion)
        acquired[:, :, 1, 1] = 0

        target = trainer._segmentation_target(lesion, acquired, torch.float32)

        self.assertEqual(float(target.sum()), 8.0)
        self.assertEqual(float(target[0, 0, 1, 1]), 0.0)
        self.assertEqual(float(target[0, 0, 2, 2]), 1.0)

    def test_lpips_restoration_configuration_and_discovery(self):
        trainer_class = (
            nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceCuria_LPIPS10_Curia05_BS_48
        )
        self.assertEqual(trainer_class.batch_size_override, 48)
        self.assertEqual(trainer_class.lpips_weight, 1.0)
        self.assertEqual(trainer_class.curia_weight, 0.5)
        self.assertEqual(
            recursive_find_trainer_class_by_name(trainer_class.__name__),
            trainer_class,
        )


class TestMamaSynthFixedValidationLoader(unittest.TestCase):
    def test_reset_repeats_every_slice_once_in_the_same_order(self):
        loader = object.__new__(MamaSynthTranslationFixedValDataLoader)
        loader.batch_size = 2
        loader.pre_mean = 0.0
        loader.pre_std = 1.0
        case = SimpleNamespace(case_id="case")
        loader.samples = [(case, slice_idx, (256, 256)) for slice_idx in range(5)]
        loader.num_batches = 3
        loader._cursor = 0

        def load_slice(case, slice_idx, pre_mean, pre_std, crop_center_yx):
            image = np.full((4, 4), slice_idx, dtype=np.float32)
            valid = np.ones((4, 4), dtype=np.float32)
            mask = np.zeros((4, 4), dtype=np.uint8)
            return image, image, image, valid, mask

        epoch_keys = []
        with patch(
            "nnunetv2.training.mamasynth.data_loader.load_pre_post_delta_slice",
            side_effect=load_slice,
        ):
            for _ in range(2):
                loader.reset()
                keys = []
                for _ in range(loader.num_batches):
                    keys.extend(loader.generate_train_batch()["keys"])
                epoch_keys.append(keys)

        expected = [f"case_s{slice_idx}" for slice_idx in range(5)]
        self.assertEqual(epoch_keys, [expected, expected])
        self.assertEqual(loader._cursor, 0)


if __name__ == "__main__":
    unittest.main()
