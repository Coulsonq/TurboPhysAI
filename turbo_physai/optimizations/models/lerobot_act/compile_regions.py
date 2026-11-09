# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Optional ACT region compilation; no changes to the upstream model."""

import functools

import torch


def _option(options, name):
    value = options.get(name, False)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _compile(original):
    return torch.compile(original, dynamic=False, options={"fallback_random": True})


def wrap_encoder(original, options):
    main = _option(options, "main_encoder")
    vae = _option(options, "vae_encoder")
    compiled = None

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        nonlocal compiled
        selected = vae if self.is_vae_encoder else main
        if not selected or not self.training or torch.compiler.is_compiling():
            return original(self, *args, **kwargs)
        if compiled is None:
            compiled = _compile(original)
        return compiled(self, *args, **kwargs)

    return wrapped


def wrap_decoder(original, options):
    del options
    compiled = None

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        nonlocal compiled
        if not self.training or torch.compiler.is_compiling():
            return original(self, *args, **kwargs)
        if compiled is None:
            compiled = _compile(original)
        return compiled(self, *args, **kwargs)

    return wrapped


def wrap_ffn(original, options):
    del options
    compiled = None

    @functools.wraps(original)
    def wrapped(layer, *args, **kwargs):
        nonlocal compiled
        if not layer.training or torch.compiler.is_compiling():
            return original(layer, *args, **kwargs)
        if compiled is None:
            compiled = _compile(original)
        return compiled(layer, *args, **kwargs)

    return wrapped
