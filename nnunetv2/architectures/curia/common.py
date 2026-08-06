from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler

CURIA_MODEL_ID = "raidium/curia-2"

# Deep-supervision scales (high resolution first) emitted by every Curia decoder.
# These are decoupled from the nnU-Net plan on purpose: a plain ViT decoder has a
# fixed /1../16 pyramid, so we pin the scales to the genuine decoder stages and make
# the trainer report the same list (so targets, loss weights and outputs all agree).
CURIA_DS_SCALES: List[List[float]] = [
    [1.0, 1.0],
    [0.5, 0.5],
    [0.25, 0.25],
    [0.125, 0.125],
    [0.0625, 0.0625],
]


def load_curia_encoder() -> nn.Module:
    try:
        from transformers import AutoModel
    except ImportError as e:
        raise ImportError(
            "Curia architectures require `transformers`. Install the project dependencies "
            "before constructing a Curia trainer."
        ) from e
    return AutoModel.from_pretrained(CURIA_MODEL_ID)


class CuriaEncoder(nn.Module):
    """Wraps the published grayscale Curia-2 DINOv2 model and exposes 4 multi-depth token maps.

    All taps live on the same ``/patch_size`` grid (a plain ViT has no spatial downsampling);
    the decoders turn them into a feature pyramid via the DPT/UPerNet reassemble trick.
    """

    def __init__(
        self,
        input_channels: int = 1,
        encoder: nn.Module | None = None,
        num_taps: int = 4,
    ):
        super().__init__()
        if input_channels != 1:
            raise ValueError(
                f"Curia-2 expects exactly one grayscale input channel, got {input_channels}."
            )
        self.model = encoder if encoder is not None else load_curia_encoder()
        config = self.model.config
        self.patch_size: int = config.patch_size
        self.hidden_size: int = config.hidden_size
        num_layers: int = config.num_hidden_layers
        # Evenly spaced taps across depth, always including the final layer, e.g. (3, 6, 9, 12).
        # hidden_states index 0 is the embedding output; index i is the output of block i.
        self.tap_layers: Tuple[int, ...] = tuple(
            round(num_layers * k / num_taps) for k in range(1, num_taps + 1)
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        grid_h, grid_w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        hidden_states = self.model(
            pixel_values=x, output_hidden_states=True
        ).hidden_states
        if hidden_states is None or len(hidden_states) <= self.tap_layers[-1]:
            raise RuntimeError(
                "Curia-2 did not return the expected per-layer hidden states."
            )
        num_patches = grid_h * grid_w
        maps = []
        for layer in self.tap_layers:
            # Drop the CLS token (and any leading register tokens) by keeping the trailing patch tokens.
            tokens = hidden_states[layer][:, -num_patches:]
            maps.append(
                tokens.transpose(1, 2).reshape(
                    tokens.shape[0], self.hidden_size, grid_h, grid_w
                )
            )
        return maps


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.GELU(),
        )


class Reassemble(nn.Module):
    """Project a ViT tap to ``dim`` channels and resample it to a target stride (DPT reassemble)."""

    def __init__(self, in_channels: int, dim: int, scale: float):
        super().__init__()
        self.project = nn.Conv2d(in_channels, dim, 1)
        if scale == 4:
            self.resample: nn.Module = nn.Sequential(
                nn.ConvTranspose2d(dim, dim, 2, 2),
                nn.GELU(),
                nn.ConvTranspose2d(dim, dim, 2, 2),
            )
        elif scale == 2:
            self.resample = nn.ConvTranspose2d(dim, dim, 2, 2)
        elif scale == 1:
            self.resample = nn.Identity()
        elif scale == 0.5:
            self.resample = nn.Conv2d(dim, dim, 3, stride=2, padding=1)
        else:
            raise ValueError(f"Unsupported reassemble scale {scale}.")

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.resample(self.project(tokens))


class UpsampleBlock(nn.Module):
    """Learned 2x upsample followed by a conv-norm-act refinement."""

    def __init__(self, dim: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(dim, dim, 2, 2)
        self.refine = ConvNormAct(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.up(x))


def add_resized(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    """Resize ``x`` to ``skip`` and add (top-down fusion)."""
    return (
        F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        + skip
    )


def build_curia_param_groups(
    network: nn.Module,
    decoder_lr: float,
    encoder_lr: float,
    weight_decay: float,
    layer_decay: float = 0.75,
) -> List[dict]:
    """AdamW param groups with layer-wise LR decay on the ViT and no weight decay on norms/embeddings.

    The encoder is reached at ``network.encoder.model`` (a DINOv2 ``Dinov2Model``). Later transformer
    blocks get a higher LR than earlier ones (``encoder_lr * layer_decay ** depth_from_top``); the
    decoder uses ``decoder_lr``. Biases, all 1-D parameters (norms, LayerScale) and the position/CLS
    embeddings are excluded from weight decay, as is standard for ViT fine-tuning.
    """
    num_layers = network.encoder.model.config.num_hidden_layers

    def layer_index(name: str) -> int | None:
        if "encoder.model.embeddings" in name:
            return 0
        if ".layer." in name:
            return int(name.split(".layer.")[1].split(".")[0]) + 1
        if "encoder.model.layernorm" in name:
            return num_layers  # final encoder norm, treated as the top block
        return None  # decoder

    no_decay_keys = ("cls_token", "position_embeddings", "mask_token")
    groups: dict[tuple, dict] = {}
    for name, parameter in network.named_parameters():
        if not parameter.requires_grad:
            continue
        index = layer_index(name)
        lr = (
            decoder_lr
            if index is None
            else encoder_lr * layer_decay ** (num_layers - index)
        )
        no_decay = parameter.ndim <= 1 or any(key in name for key in no_decay_keys)
        wd = 0.0 if no_decay else weight_decay
        key = (lr, wd)
        groups.setdefault(key, {"params": [], "lr": lr, "weight_decay": wd})[
            "params"
        ].append(parameter)
    return list(groups.values())


class RatioPreservingPolyLRScheduler(_LRScheduler):
    """Poly decay that scales every param group by the same factor, preserving the LLRD ratios."""

    def __init__(self, optimizer, max_steps: int, exponent: float = 0.9):
        self.max_steps = max(max_steps, 1)
        self.exponent = exponent
        self.base_group_lrs = [group["lr"] for group in optimizer.param_groups]
        super().__init__(optimizer, -1)

    def step(self, current_step: int | None = None):
        step = 0 if current_step is None else current_step
        factor = (1 - min(step, self.max_steps) / self.max_steps) ** self.exponent
        for base_lr, group in zip(self.base_group_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * factor
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
