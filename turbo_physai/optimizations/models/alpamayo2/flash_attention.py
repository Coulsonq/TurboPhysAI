# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Select FlashAttention 2 before constructing the Alpamayo model."""

from functools import wraps


def flash_attention_wrapper(original, options):
    @wraps(original)
    def wrapped(model, config, *args, **kwargs):
        config._attn_implementation = "flash_attention_2"
        return original(model, config, *args, **kwargs)

    return wrapped
