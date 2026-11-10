# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Enable non-blocking Accelerate data transfers when not explicitly configured."""

import functools
import inspect


def accelerator_init_wrapper(original, options):
    del options
    signature = inspect.signature(original)
    supports_config = "dataloader_config" in signature.parameters

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        if supports_config:
            bound = signature.bind_partial(*args, **kwargs)
            if "dataloader_config" not in bound.arguments:
                from accelerate.utils import DataLoaderConfiguration

                kwargs["dataloader_config"] = DataLoaderConfiguration(non_blocking=True)
        return original(*args, **kwargs)

    return wrapped
