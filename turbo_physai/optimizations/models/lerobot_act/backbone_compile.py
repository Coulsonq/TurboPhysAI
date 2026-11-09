# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Compile the ACT vision backbone independently of its memory layout."""

import functools

import torch

_MARKER = "_turbo_lerobot_act_compile_backbone"


def policy_init_wrapper(original, options):
    del options

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if hasattr(self.model, "backbone"):
            setattr(self.model.backbone, _MARKER, True)
        return result

    return wrapped


def compiled_backbone_forward_wrapper(original, options):
    del options
    compiled = None

    @functools.wraps(original)
    def wrapped(self, x, *args, **kwargs):
        nonlocal compiled
        if not getattr(self, _MARKER, False):
            return original(self, x, *args, **kwargs)
        if compiled is None:
            compiled = torch.compile(original, mode="default", dynamic=False)
        return compiled(self, x, *args, **kwargs)

    return wrapped
