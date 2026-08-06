from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from nnunetv2.architectures.curia.base import CuriaDecoderBase, CuriaSegmentor
from nnunetv2.architectures.curia.common import ConvNormAct, CuriaEncoder, add_resized


class RefineBlock(nn.Module):
    """RefineNet-style fusion unit: add the upsampled coarser map to the skip, then refine."""

    def __init__(self, dim: int):
        super().__init__()
        self.unit = nn.Sequential(ConvNormAct(dim, dim), ConvNormAct(dim, dim))

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        if skip is not None:
            x = add_resized(x, skip)
        return self.unit(x)


class DPTDecoder(CuriaDecoderBase):
    """Dense Prediction Transformer decoder: reassemble taps to a pyramid, fuse top-down."""

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        dim: int = 256,
        deep_supervision: bool = True,
    ):
        super().__init__(hidden_size, num_classes, dim, deep_supervision)
        self.refine = nn.ModuleList(
            RefineBlock(dim) for _ in range(4)
        )  # strides 4, 8, 16, 32

    def fuse(
        self, p4: torch.Tensor, p8: torch.Tensor, p16: torch.Tensor, p32: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r32 = self.refine[3](p32)
        r16 = self.refine[2](r32, p16)
        r8 = self.refine[1](r16, p8)
        r4 = self.refine[0](r8, p4)
        return r4, r8, r16


class CuriaDPT(CuriaSegmentor):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        deep_supervision: bool = True,
        encoder: nn.Module | None = None,
        dim: int = 256,
    ):
        encoder = CuriaEncoder(input_channels, encoder)
        decoder = DPTDecoder(encoder.hidden_size, num_classes, dim, deep_supervision)
        super().__init__(encoder, decoder)
