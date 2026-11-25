# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Configure Alpamayo training workers and PyTorch CPU threads."""

import os
from functools import wraps


def _positive_integer(options, name, default):
    value = options.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def train_dataloader_wrapper(original, options):
    workers = _positive_integer(options, "num_workers", 2)

    @wraps(original)
    def wrapped(trainer, *args, **kwargs):
        previous = trainer.args.dataloader_num_workers
        trainer.args.dataloader_num_workers = workers
        try:
            return original(trainer, *args, **kwargs)
        finally:
            trainer.args.dataloader_num_workers = previous

    return wrapped


def omp_threads_wrapper(original, options):
    threads = _positive_integer(options, "omp_num_threads", 4)

    @wraps(original)
    def wrapped(training_args, *args, **kwargs):
        import torch

        os.environ["OMP_NUM_THREADS"] = str(threads)
        torch.set_num_threads(threads)
        return original(training_args, *args, **kwargs)

    return wrapped
