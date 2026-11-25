# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Enable non-blocking device transfers for the training accelerator."""

from functools import wraps


def non_blocking_wrapper(original, options):
    @wraps(original)
    def wrapped(trainer):
        trainer.args.accelerator_config.non_blocking = True
        return original(trainer)

    return wrapped
