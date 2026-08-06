import os
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from nnunetv2.architectures.curia.common import (
    CURIA_DS_SCALES,
    RatioPreservingPolyLRScheduler,
    build_curia_param_groups,
)
from nnunetv2.architectures.curia.dpt import CuriaDPT
from nnunetv2.architectures.curia.upernet import CuriaUPerNet
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerCuria import (
    nnUNetTrainerCuriaDPT,
)
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name


class _MockBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.dense = nn.Linear(hidden, hidden)

    def forward(self, x):
        return x + self.dense(self.norm(x))


class _MockEmbeddings(nn.Module):
    def __init__(self, hidden: int, patch: int):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden))
        self.position_embeddings = nn.Parameter(torch.zeros(1, 1, hidden))
        self.patch_embeddings = nn.Conv2d(1, hidden, patch, patch)

    def forward(self, pixel_values):
        patches = self.patch_embeddings(pixel_values).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(patches.shape[0], -1, -1)
        return torch.cat([cls, patches], dim=1) + self.position_embeddings


class MockCuriaModel(nn.Module):
    """Stand-in for the transformers Dinov2Model: same module names, config and forward contract."""

    def __init__(self, hidden: int = 768, num_layers: int = 12, patch: int = 16):
        super().__init__()
        self.config = SimpleNamespace(
            patch_size=patch, hidden_size=hidden, num_hidden_layers=num_layers
        )
        self.embeddings = _MockEmbeddings(hidden, patch)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList(
            _MockBlock(hidden) for _ in range(num_layers)
        )
        self.layernorm = nn.LayerNorm(hidden)

    def forward(self, pixel_values, output_hidden_states=False, **kwargs):
        x = self.embeddings(pixel_values)
        hidden_states = [x]
        for block in self.encoder.layer:
            x = block(x)
            hidden_states.append(x)
        return SimpleNamespace(
            hidden_states=tuple(hidden_states), last_hidden_state=self.layernorm(x)
        )


def _expected_ds_shapes(num_classes, h, w):
    return [
        (1, num_classes, round(h * sy), round(w * sx)) for sy, sx in CURIA_DS_SCALES
    ]


def _group_of(groups, param):
    return next(g for g in groups if any(p is param for p in g["params"]))


class TestCuriaArchitectures(unittest.TestCase):
    def test_rejects_multi_channel_input(self):
        with self.assertRaisesRegex(ValueError, "exactly one grayscale"):
            CuriaDPT(2, 3, encoder=MockCuriaModel())

    def test_deep_supervision_shapes(self):
        for model_cls, num_classes in ((CuriaDPT, 3), (CuriaUPerNet, 2)):
            model = model_cls(1, num_classes, True, MockCuriaModel())
            with torch.no_grad():
                outputs = model(torch.randn(1, 1, 256, 256))
            self.assertEqual(
                [tuple(o.shape) for o in outputs],
                _expected_ds_shapes(num_classes, 256, 256),
            )

    def test_backward_and_encoder_is_finetuned(self):
        for model_cls in (CuriaDPT, CuriaUPerNet):
            model = model_cls(1, 2, True, MockCuriaModel())
            outputs = model(torch.randn(1, 1, 128, 128))
            sum(o.float().sum() for o in outputs).backward()
            grad = model.encoder.model.encoder.layer[0].dense.weight.grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())

    def test_non_divisible_input_is_padded_then_cropped(self):
        for model_cls in (CuriaDPT, CuriaUPerNet):
            model = model_cls(1, 2, True, MockCuriaModel())
            with torch.no_grad():
                outputs = model(torch.randn(1, 1, 250, 260))
            self.assertEqual(
                [tuple(o.shape) for o in outputs], _expected_ds_shapes(2, 250, 260)
            )

    def test_inference_mode_returns_single_full_res_map(self):
        model = CuriaDPT(1, 4, deep_supervision=False, encoder=MockCuriaModel())
        with torch.no_grad():
            output = model(torch.randn(1, 1, 128, 96))
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(tuple(output.shape), (1, 4, 128, 96))

    def test_toggle_deep_supervision_at_runtime(self):
        model = CuriaUPerNet(1, 2, True, MockCuriaModel())
        model.decoder.deep_supervision = False
        with torch.no_grad():
            self.assertIsInstance(model(torch.randn(1, 1, 64, 64)), torch.Tensor)

    def test_llrd_param_groups(self):
        model = CuriaDPT(1, 2, encoder=MockCuriaModel())
        groups = build_curia_param_groups(
            model, decoder_lr=1e-4, encoder_lr=1e-5, weight_decay=5e-2, layer_decay=0.5
        )

        # Every trainable parameter is assigned exactly once.
        assigned = sum(len(g["params"]) for g in groups)
        self.assertEqual(
            assigned, sum(1 for p in model.parameters() if p.requires_grad)
        )

        # Decoder LR is the largest; the top encoder block sits at exactly encoder_lr.
        encoder_lrs = sorted({g["lr"] for g in groups if g["lr"] < 1e-4})
        self.assertAlmostEqual(max(encoder_lrs), 1e-5)
        self.assertGreater(1e-4, max(encoder_lrs))

        # Embeddings: deepest-from-top -> smallest LR, and never weight-decayed.
        cls_group = _group_of(groups, model.encoder.model.embeddings.cls_token)
        self.assertEqual(cls_group["weight_decay"], 0.0)
        self.assertAlmostEqual(cls_group["lr"], 1e-5 * 0.5**12)

        # Decoder conv weight: decoder LR and weight-decayed; its bias is not.
        head = model.decoder.heads[0]
        self.assertAlmostEqual(_group_of(groups, head.weight)["lr"], 1e-4)
        self.assertEqual(_group_of(groups, head.weight)["weight_decay"], 5e-2)
        self.assertEqual(_group_of(groups, head.bias)["weight_decay"], 0.0)

    def test_ratio_preserving_scheduler(self):
        first, second = nn.Parameter(torch.ones(1)), nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW(
            [{"params": [first], "lr": 1e-5}, {"params": [second], "lr": 1e-4}]
        )
        scheduler = RatioPreservingPolyLRScheduler(optimizer, 10)
        scheduler.step(5)
        self.assertAlmostEqual(
            optimizer.param_groups[1]["lr"] / optimizer.param_groups[0]["lr"], 10
        )
        self.assertLess(optimizer.param_groups[0]["lr"], 1e-5)

    def test_trainer_ds_scales_match_network_outputs(self):
        trainer = object.__new__(nnUNetTrainerCuriaDPT)
        trainer.enable_deep_supervision = True
        scales = trainer._get_deep_supervision_scales()
        self.assertEqual(scales, [list(s) for s in CURIA_DS_SCALES])
        model = CuriaDPT(1, 2, True, MockCuriaModel())
        with torch.no_grad():
            self.assertEqual(len(model(torch.randn(1, 1, 128, 128))), len(scales))

    def test_trainers_are_discoverable(self):
        for trainer in ("nnUNetTrainerCuriaDPT", "nnUNetTrainerCuriaUPerNet"):
            self.assertEqual(
                recursive_find_trainer_class_by_name(trainer).__name__, trainer
            )

    @unittest.skipUnless(
        os.environ.get("RUN_CURIA_HUB_SMOKE_TEST") == "1",
        "requires cached or downloadable Hub weights",
    )
    def test_hub_checkpoint_smoke(self):
        image = torch.randn(1, 1, 512, 512)
        for model_cls in (CuriaDPT, CuriaUPerNet):
            with self.subTest(architecture=model_cls.__name__):
                model = model_cls(1, 2, deep_supervision=False)
                with torch.no_grad():
                    self.assertEqual(tuple(model(image).shape), (1, 2, 512, 512))


if __name__ == "__main__":
    unittest.main()
