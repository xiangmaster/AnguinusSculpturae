from __future__ import annotations

from typing import List, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from nnunetv2.architectures.curia.common import (
    CURIA_DS_SCALES,
    CuriaEncoder,
    Reassemble,
    UpsampleBlock,
)


class CuriaDecoderBase(nn.Module):
    """Shared decoder scaffold for the Curia segmentation heads.

    Subclasses implement :meth:`fuse`, which turns the four reassembled pyramid levels
    (strides 4, 8, 16, 32 relative to the input) into three fused maps at strides 4, 8 and 16.
    This base owns the reassemble stem, the learned /4 -> /2 -> /1 upsampling tail and the
    five segmentation heads, and assembles the deep-supervision list (strides 1, 2, 4, 8, 16,
    matching :data:`CURIA_DS_SCALES`).
    """

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        dim: int = 256,
        deep_supervision: bool = True,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        # tap (stride /16) -> pyramid levels at strides 4, 8, 16, 32
        self.reassemble = nn.ModuleList(
            Reassemble(hidden_size, dim, scale) for scale in (4, 2, 1, 0.5)
        )
        self.up_to_half = UpsampleBlock(dim)  # /4 -> /2
        self.up_to_full = UpsampleBlock(dim)  # /2 -> /1
        # heads for strides 1, 2, 4, 8, 16 (high resolution first)
        self.heads = nn.ModuleList(nn.Conv2d(dim, num_classes, 1) for _ in range(5))

    def fuse(
        self, p4: torch.Tensor, p8: torch.Tensor, p16: torch.Tensor, p32: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward(
        self, taps: List[torch.Tensor]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        p4, p8, p16, p32 = (
            reassemble(tap) for reassemble, tap in zip(self.reassemble, taps)
        )
        f4, f8, f16 = self.fuse(p4, p8, p16, p32)
        half = self.up_to_half(f4)
        full = self.up_to_full(half)
        main = self.heads[0](full)
        if not self.deep_supervision:
            return main
        return [
            main,
            self.heads[1](half),
            self.heads[2](f4),
            self.heads[3](f8),
            self.heads[4](f16),
        ]


class CuriaSegmentor(nn.Module):
    """Curia encoder + pluggable decoder, with patch-multiple padding and exact-size deep supervision."""

    def __init__(self, encoder: CuriaEncoder, decoder: CuriaDecoderBase):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.patch_size = encoder.patch_size

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        size = x.shape[-2:]
        pad_h, pad_w = (-size[0]) % self.patch_size, (-size[1]) % self.patch_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        outputs = self.decoder(self.encoder(x))
        if isinstance(outputs, list):
            return [
                self._to_target(output, size, scale)
                for output, scale in zip(outputs, CURIA_DS_SCALES)
            ]
        return self._to_target(outputs, size, CURIA_DS_SCALES[0])

    @staticmethod
    def _to_target(logits: torch.Tensor, size, scale) -> torch.Tensor:
        target = (round(size[0] * scale[0]), round(size[1] * scale[1]))
        if logits.shape[-2:] == target:
            return logits
        if logits.shape[-2] >= target[0] and logits.shape[-1] >= target[1]:
            return logits[
                ..., : target[0], : target[1]
            ]  # crop padding (top-left aligned)
        return F.interpolate(logits, size=target, mode="bilinear", align_corners=False)
