# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""ACT vision layout and compatible optimizer-state serialization."""

import functools

import torch


_MARKER = "_turbo_lerobot_act_channels_last"


def act_init_wrapper(original, options):
    del options

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if hasattr(self, "backbone") and hasattr(self, "encoder_img_feat_input_proj"):
            self.backbone.to(memory_format=torch.channels_last)
            self.encoder_img_feat_input_proj.to(memory_format=torch.channels_last)
            setattr(self.backbone.conv1, _MARKER, True)
        return result

    return wrapped


def conv_forward_wrapper(original, options):
    del options

    @functools.wraps(original)
    def wrapped(self, input, *args, **kwargs):
        if getattr(self, _MARKER, False):
            input = input.to(memory_format=torch.channels_last)
        return original(self, input, *args, **kwargs)

    return wrapped


def _contiguous_tensors(value):
    if isinstance(value, torch.Tensor):
        return value if value.is_contiguous() else value.contiguous()
    if isinstance(value, dict):
        return {key: _contiguous_tensors(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_contiguous_tensors(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_contiguous_tensors(item) for item in value)
    return value


def optimizer_save_wrapper(original, options):
    del options

    @functools.wraps(original)
    def wrapped(optimizer, save_dir, optim_state_dict=None):
        state = optimizer.state_dict() if optim_state_dict is None else optim_state_dict
        save_state = {
            key: value if key == "param_groups" else _contiguous_tensors(value)
            for key, value in state.items()
        }
        return original(optimizer, save_dir, optim_state_dict=save_state)

    return wrapped
