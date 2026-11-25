# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Alpamayo 2 Super optimization declarations."""

from turbo_physai.engine.definitions import group, wrap


SPAWN_DATALOADER = group(
    "alpamayo2.spawn_dataloader",
    wrap(
        target="alpamayo2_sft.trainer.Alpamayo2SuperTrainer.get_train_dataloader",
        replacement="turbo_physai.optimizations.models.alpamayo2.dataloader.train_dataloader_wrapper",
    ),
    wrap(
        target="transformers.trainer.DataLoader",
        replacement="turbo_physai.optimizations.models.alpamayo2.dataloader.dataloader_wrapper",
    ),
)

FLASH_ATTENTION = group(
    "alpamayo2.flash_attention",
    wrap(
        target="alpamayo2_sft.models.trainable_alpamayo2_super.TrainableAlpamayo2Super.__init__",
        replacement="turbo_physai.optimizations.models.alpamayo2.flash_attention.flash_attention_wrapper",
    ),
)

NON_BLOCKING_H2D = group(
    "alpamayo2.non_blocking_h2d",
    wrap(
        target="transformers.trainer.Trainer.create_accelerator_and_postprocess",
        replacement="turbo_physai.optimizations.models.alpamayo2.non_blocking.non_blocking_wrapper",
    ),
)
