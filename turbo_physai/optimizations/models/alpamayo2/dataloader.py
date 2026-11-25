# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Use spawned workers for the Alpamayo training DataLoader."""

import os
import inspect
from functools import wraps


def train_dataloader_wrapper(original, options):
    workers = options.get("num_workers", 2)
    if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
        raise ValueError("num_workers must be a positive integer")

    @wraps(original)
    def wrapped(trainer, *args, **kwargs):
        previous = trainer.args.dataloader_num_workers
        trainer.args.dataloader_num_workers = workers
        try:
            return original(trainer, *args, **kwargs)
        finally:
            trainer.args.dataloader_num_workers = previous

    return wrapped


def dataloader_wrapper(original, options):
    class SpawnDataLoader(original):
        def __init__(self, *args, **kwargs):
            if kwargs.get("worker_init_fn") is not None:
                kwargs["multiprocessing_context"] = "spawn"
            super().__init__(*args, **kwargs)
            if kwargs.get("worker_init_fn") is not None and os.environ.get("RANK", "0") == "0":
                print(f"TURBO_ALPAMAYO_DATALOADER workers={self.num_workers} context=spawn", flush=True)

    SpawnDataLoader.__signature__ = inspect.signature(original)
    return SpawnDataLoader
