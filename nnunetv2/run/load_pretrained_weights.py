import torch
import torch.distributed as dist
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP


def _adapt_input_conv_weight(
    pretrained_weight: torch.Tensor, target_shape: torch.Size
) -> torch.Tensor | None:
    if pretrained_weight.ndim < 3 or len(target_shape) != pretrained_weight.ndim:
        return None
    if (
        pretrained_weight.shape[0] != target_shape[0]
        or pretrained_weight.shape[2:] != target_shape[2:]
    ):
        return None
    if pretrained_weight.shape[1] == target_shape[1]:
        return pretrained_weight

    target_channels = target_shape[1]
    source_channels = pretrained_weight.shape[1]
    if source_channels == 1:
        return (
            pretrained_weight.repeat(
                1, target_channels, *([1] * (pretrained_weight.ndim - 2))
            )
            / target_channels
        )
    if target_channels == 1:
        return pretrained_weight.mean(1, keepdim=True)
    return None


def load_pretrained_weights(network, fname, verbose=False):
    """
    Transfers all weights between matching keys in state_dicts. matching is done by name and we only transfer if the
    shape is also the same. Segmentation layers (the 1x1(x1) layers that produce the segmentation maps)
    identified by keys ending with '.seg_layers') are not transferred!

    If the pretrained weights were obtained with a training outside nnU-Net and DDP or torch.optimize was used,
    you need to change the keys of the pretrained state_dict. DDP adds a 'module.' prefix and torch.optim adds
    '_orig_mod'. You DO NOT need to worry about this if pretraining was done with nnU-Net as
    nnUNetTrainer.save_checkpoint takes care of that!

    """
    if dist.is_initialized():
        saved_model = torch.load(
            fname,
            map_location=torch.device("cuda", dist.get_rank()),
            weights_only=False,
        )
    else:
        saved_model = torch.load(fname, weights_only=False)
    pretrained_dict = saved_model["network_weights"]

    skip_strings_in_pretrained = [
        ".seg_layers.",
    ]

    if isinstance(network, DDP):
        mod = network.module
    else:
        mod = network
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    model_dict = mod.state_dict()

    # fun fact: in principle this allows loading from parameters that do not cover the entire network. For example pretrained
    # encoders. Not supported by this function though (see assertions above)

    # commenting out this abomination of a dict comprehension for preservation in the archives of 'what not to do'
    # pretrained_dict = {'module.' + k if is_ddp else k: v
    #                    for k, v in pretrained_dict.items()
    #                    if (('module.' + k if is_ddp else k) in model_dict) and
    #                    all([i not in k for i in skip_strings_in_pretrained])}

    skipped = []
    loaded_pretrained_dict = {}
    for key, value in pretrained_dict.items():
        if key not in model_dict or any(i in key for i in skip_strings_in_pretrained):
            continue
        if model_dict[key].shape == value.shape:
            loaded_pretrained_dict[key] = value
            continue
        adapted_value = _adapt_input_conv_weight(value, model_dict[key].shape)
        if adapted_value is not None:
            loaded_pretrained_dict[key] = adapted_value
            continue
        skipped.append((key, value.shape, model_dict[key].shape))

    model_dict.update(loaded_pretrained_dict)

    print(
        "################### Loading pretrained weights from file ",
        fname,
        "###################",
    )
    if verbose:
        print(
            "Below is the list of overlapping blocks in pretrained model and nnUNet architecture:"
        )
        for key, value in loaded_pretrained_dict.items():
            print(key, "shape", value.shape)
        if len(skipped) > 0:
            print(
                "Below is the list of overlapping blocks that were skipped because their shapes differ:"
            )
            for key, pretrained_shape, model_shape in skipped:
                print(
                    key,
                    "pretrained shape",
                    pretrained_shape,
                    "model shape",
                    model_shape,
                )
        print("################### Done ###################")
    elif len(skipped) > 0:
        print(
            f"Skipped {len(skipped)} pretrained weights because their shapes differ. "
            f"Run with verbose=True to list them."
        )
    mod.load_state_dict(model_dict)
