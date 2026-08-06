from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from nnunetv2.architectures.curia.base import CuriaDecoderBase, CuriaSegmentor
from nnunetv2.architectures.curia.common import ConvNormAct, CuriaEncoder, add_resized


class PyramidPooling(nn.Module):
    """UPerNet's PSP module: pool at several bin sizes, then fuse with the input feature."""

    def __init__(self, dim: int, bins: Tuple[int, ...] = (1, 2, 3, 6)):
        super().__init__()
        self.bins = bins
        self.branches = nn.ModuleList(nn.Conv2d(dim, dim // 4, 1) for _ in bins)
        self.fuse = ConvNormAct(dim + (dim // 4) * len(bins), dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = [x]
        for bin_size, branch in zip(self.bins, self.branches):
            feature = branch(F.adaptive_avg_pool2d(x, bin_size))
            pooled.append(
                F.interpolate(
                    feature, size=x.shape[-2:], mode="bilinear", align_corners=False
                )
            )
        return self.fuse(torch.cat(pooled, dim=1))


class UPerNetDecoder(CuriaDecoderBase):
    """UPerNet decoder: PSP on the coarsest level, FPN top-down fusion, multi-level merge."""

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        dim: int = 256,
        deep_supervision: bool = True,
    ):
        super().__init__(hidden_size, num_classes, dim, deep_supervision)
        self.ppm = PyramidPooling(dim)
        self.lateral = nn.ModuleList(
            nn.Conv2d(dim, dim, 1) for _ in range(3)
        )  # strides 4, 8, 16
        self.smooth = nn.ModuleList(
            ConvNormAct(dim, dim) for _ in range(3)
        )  # strides 4, 8, 16
        self.merge = ConvNormAct(dim * 4, dim)

    def fuse(
        self, p4: torch.Tensor, p8: torch.Tensor, p16: torch.Tensor, p32: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l32 = self.ppm(p32)
        l16 = add_resized(l32, self.lateral[2](p16))
        l8 = add_resized(l16, self.lateral[1](p8))
        l4 = add_resized(l8, self.lateral[0](p4))
        f16, f8, f4 = self.smooth[2](l16), self.smooth[1](l8), self.smooth[0](l4)
        merged = self.merge(
            torch.cat(
                [f4]
                + [
                    F.interpolate(
                        f, size=f4.shape[-2:], mode="bilinear", align_corners=False
                    )
                    for f in (f8, f16, l32)
                ],
                dim=1,
            )
        )
        return merged, f8, f16


class CuriaUPerNet(CuriaSegmentor):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        deep_supervision: bool = True,
        encoder: nn.Module | None = None,
        dim: int = 256,
    ):
        encoder = CuriaEncoder(input_channels, encoder)
        decoder = UPerNetDecoder(
            encoder.hidden_size, num_classes, dim, deep_supervision
        )
        super().__init__(encoder, decoder)
