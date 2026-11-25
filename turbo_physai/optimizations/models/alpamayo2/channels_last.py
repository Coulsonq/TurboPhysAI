# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Use channels-last 3D inputs for Qwen3-VL's visual patch projection."""

import os
from functools import wraps


def patch_embed_init_wrapper(original, options):
    @wraps(original)
    def wrapped(module, *args, **kwargs):
        import torch
        for name in ('MIOPEN_FIND_MODE', 'PYTORCH_MIOPEN_SUGGEST_NHWC'):
            if os.environ.get(name) != '1':
                raise RuntimeError(f'{name}=1 must be set before starting the process')
        result = original(module, *args, **kwargs)
        if not isinstance(module.proj, torch.nn.Conv3d):
            raise RuntimeError('Expected Qwen3-VL patch projection to be Conv3d')
        module.proj._alpamayo_channels_last_3d = True
        return result
    return wrapped


def conv3d_forward_wrapper(original, options):
    @wraps(original)
    def wrapped(module, input, *args, **kwargs):
        if getattr(module, '_alpamayo_channels_last_3d', False):
            import torch
            input = input.contiguous(memory_format=torch.channels_last_3d)
        return original(module, input, *args, **kwargs)
    return wrapped
