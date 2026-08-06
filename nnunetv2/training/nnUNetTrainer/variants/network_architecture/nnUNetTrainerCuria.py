from __future__ import annotations

import torch
from torch import nn

from nnunetv2.architectures.curia.common import (
    CURIA_DS_SCALES,
    RatioPreservingPolyLRScheduler,
    build_curia_param_groups,
)
from nnunetv2.architectures.curia.dpt import CuriaDPT
from nnunetv2.architectures.curia.upernet import CuriaUPerNet
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)


class AbstractnnUNetTrainerCuria(nnUNetTrainer):
    """Fine-tunes a Curia-2 (DINOv2) encoder with a dense segmentation decoder.

    The networks conform to nnU-Net's output contract (a deep-supervision list when enabled,
    a single map otherwise) and expose ``network.decoder.deep_supervision``, so the entire base
    training/validation/inference loop is reused unchanged. Only the optimizer (layer-wise LR
    decay + AdamW) and the deep-supervision scales (pinned to the decoder's /1../16 pyramid)
    are customized.
    """

    def __init__(
        self, plans, configuration, fold, dataset_json, device=torch.device("cuda")
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-4  # decoder LR
        self.encoder_lr = (
            1e-5  # base LR of the top transformer block (decayed downwards)
        )
        self.layer_decay = 0.75  # layer-wise LR decay factor across encoder depth
        self.weight_decay = 5e-2

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        raise NotImplementedError

    def _do_i_compile(self):
        # transformers + dynamic input sizes do not play well with torch.compile here.
        return False

    def _get_deep_supervision_scales(self):
        # Pin targets/weights to the decoder's genuine output scales (see CURIA_DS_SCALES).
        return (
            [list(scale) for scale in CURIA_DS_SCALES]
            if self.enable_deep_supervision
            else None
        )

    def configure_optimizers(self):
        network = self.network.module if self.is_ddp else self.network
        param_groups = build_curia_param_groups(
            network,
            self.initial_lr,
            self.encoder_lr,
            self.weight_decay,
            self.layer_decay,
        )
        optimizer = torch.optim.AdamW(param_groups)
        scheduler = RatioPreservingPolyLRScheduler(optimizer, self.num_epochs)
        return optimizer, scheduler


class nnUNetTrainerCuriaDPT(AbstractnnUNetTrainerCuria):
    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=True,
    ):
        return CuriaDPT(
            num_input_channels, num_output_channels, enable_deep_supervision
        )


class nnUNetTrainerCuriaUPerNet(AbstractnnUNetTrainerCuria):
    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=True,
    ):
        return CuriaUPerNet(
            num_input_channels, num_output_channels, enable_deep_supervision
        )
