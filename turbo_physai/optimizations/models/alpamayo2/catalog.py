# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Alpamayo 2 Super optimization declarations."""

from turbo_physai.engine.definitions import group, wrap


DATALOADER_WORKERS_OMP = group(
    "alpamayo2.dataloader_workers_omp",
    wrap(
        target="alpamayo2_sft.trainer.Alpamayo2SuperTrainer.get_train_dataloader",
        replacement="turbo_physai.optimizations.models.alpamayo2.dataloader.train_dataloader_wrapper",
    ),
    wrap(
        target="alpamayo2_sft.trainer.TrainingArguments.__post_init__",
        replacement="turbo_physai.optimizations.models.alpamayo2.dataloader.omp_threads_wrapper",
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

DISABLE_CPU_OFFLOAD = group(
    "alpamayo2.disable_cpu_offload",
    wrap(
        target="transformers.integrations.deepspeed.HfTrainerDeepSpeedConfig.__init__",
        replacement="turbo_physai.optimizations.models.alpamayo2.cpu_offload.disable_cpu_offload_wrapper",
    ),
)


CHANNELS_LAST_VISION = group(
    "alpamayo2.channels_last_vision",
    wrap(
        target="transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionPatchEmbed.__init__",
        replacement="turbo_physai.optimizations.models.alpamayo2.channels_last.patch_embed_init_wrapper",
    ),
    wrap(
        target="torch.nn.modules.conv.Conv3d.forward",
        replacement="turbo_physai.optimizations.models.alpamayo2.channels_last.conv3d_forward_wrapper",
    ),
)


BF16_COMMUNICATION = group(
    "alpamayo2.bf16_communication",
    wrap(
        target="accelerate.utils.dataclasses.DeepSpeedPlugin.set_mixed_precision",
        replacement="turbo_physai.optimizations.models.alpamayo2.bf16_communication.bf16_communication_wrapper",
    ),
)


ZERO3_COMM_OVERLAP = group(
    "alpamayo2.zero3_comm_overlap",
    wrap(
        target="deepspeed.runtime.zero.stage3.DeepSpeedZeroOptimizer_Stage3.__init__",
        replacement="turbo_physai.optimizations.models.alpamayo2.zero3_overlap.optimizer_init_wrapper",
    ),
    wrap(
        target="deepspeed.runtime.zero.stage3.DeepSpeedZeroOptimizer_Stage3._DeepSpeedZeroOptimizer_Stage3__avg_scatter_grads",
        replacement="turbo_physai.optimizations.models.alpamayo2.zero3_overlap.gradient_reduce_wrapper",
    ),
    wrap(
        target="deepspeed.runtime.zero.stage3.DeepSpeedZeroOptimizer_Stage3.destroy",
        replacement="turbo_physai.optimizations.models.alpamayo2.zero3_overlap.optimizer_destroy_wrapper",
    ),
)
