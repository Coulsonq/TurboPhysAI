# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""ACT-specific DDP configuration."""

import functools
import inspect


def ddp_find_unused_wrapper(original, options):
    signature = inspect.signature(original)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        from lerobot.policies.act.modeling_act import ACTPolicy

        bound = signature.bind(*args, **kwargs)
        if isinstance(bound.arguments.get("module"), ACTPolicy):
            if bound.arguments.get("find_unused_parameters") is True:
                bound.arguments["find_unused_parameters"] = False
            for key in ("static_graph", "broadcast_buffers"):
                if key in options:
                    bound.arguments[key] = options[key]
        return original(*bound.args, **bound.kwargs)

    return wrapped
