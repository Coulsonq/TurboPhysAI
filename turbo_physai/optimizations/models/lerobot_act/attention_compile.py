# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental HCU attention compilation, inspired by the ACT-H20 hotspot."""

import functools

import torch


_MARKER = "_turbo_lerobot_act_compile_attention"


def encoder_init_wrapper(original, options):
    del options

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if not self.is_vae_encoder:
            for layer in self.layers:
                setattr(layer.self_attn, _MARKER, True)
        return result

    return wrapped


def attention_forward_wrapper(original, options):
    del options
    compiled = None

    @functools.wraps(original)
    def wrapped(self, query, key, value, *args, **kwargs):
        nonlocal compiled
        eligible = (
            getattr(self, _MARKER, False)
            and self.training
            and self.embed_dim == 512
            and self.num_heads == 8
            and self.dropout == 0.1
            and not getattr(self, "batch_first", False)
            and query.ndim == 3
            and tuple(query.shape) == (902, 8, 512)
            and query.is_cuda
            and query.dtype == torch.float32
            and not torch.compiler.is_compiling()
        )
        if not eligible:
            return original(self, query, key, value, *args, **kwargs)
        if compiled is None:
            compiled = torch.compile(original, mode="default", dynamic=False)
        return compiled(self, query, key, value, *args, **kwargs)

    return wrapped
