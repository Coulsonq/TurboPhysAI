# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Strict Linear-module compilation for the Alpamayo2 VLM training path."""

import os
from functools import wraps


def compile_module_wrapper(original, options):
    if os.environ.get('TURBO_PHYSAI_DISABLE_TORCH_COMPILE') == '1':
        return original
    import torch
    import torch._dynamo

    compiled = torch.compile(original, backend='inductor', fullgraph=True, dynamic=True)
    profile = os.environ.get('ALPAMAYO_COMPILE_TRACE') == '1'

    @wraps(original)
    def wrapped(*args, **kwargs):
        with torch._dynamo.config.patch(suppress_errors=False, fail_on_recompile_limit_hit=True, recompile_limit=64):
            if profile:
                with torch.profiler.record_function('ALPA_COMPILE ' + original.__qualname__):
                    return compiled(*args, **kwargs)
            return compiled(*args, **kwargs)
    return wrapped
