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


APEX_FUSED_ADAMW = group(
    "alpamayo2.apex_fused_adamw",
    wrap(
        target="alpamayo2_sft.trainer.Alpamayo2SuperTrainer.create_optimizer",
        replacement="turbo_physai.optimizations.models.alpamayo2.apex_adamw.optimizer_wrapper",
    ),
    wrap(
        target="deepspeed.runtime.zero.stage3.DeepSpeedZeroOptimizer_Stage3._optimizer_step",
        replacement="turbo_physai.optimizations.models.alpamayo2.apex_adamw.zero3_step_wrapper",
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


MODULE_COMPILE = group(
    "alpamayo2.module_compile",
    wrap(
        target="torch.nn.modules.linear.Linear.forward",
        replacement="turbo_physai.optimizations.models.alpamayo2.module_compile.compile_module_wrapper",
    ),
)
