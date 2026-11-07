# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Compile the ACT vision backbone while preserving channels-last input."""

import functools

import torch

from turbo_physai.optimizations.models.lerobot_act.channels_last import _MARKER


def compiled_backbone_forward_wrapper(original, options):
    del options
    compiled = None

    @functools.wraps(original)
    def wrapped(self, image, *args, **kwargs):
        nonlocal compiled
        if not getattr(self, _MARKER, False):
            return original(self, image, *args, **kwargs)
        if compiled is None:
            compiled = torch.compile(original, mode="default", dynamic=False)
        image = image.to(memory_format=torch.channels_last)
        return compiled(self, image, *args, **kwargs)

    return wrapped
