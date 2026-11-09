# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Declarative optimization groups for LeRobot ACT."""

from ....engine.definitions import group, wrap


DDP_FIND_UNUSED = group(
    "lerobot_act.ddp_find_unused",
    wrap(
        target="torch.nn.parallel.distributed.DistributedDataParallel.__init__",
        replacement="turbo_physai.optimizations.models.lerobot_act.ddp.ddp_find_unused_wrapper",
    ),
)

CHANNELS_LAST_COMPILE = group(
    "lerobot_act.channels_last_backbone_compile",
    wrap(
        target="lerobot.policies.act.modeling_act.ACT.__init__",
        replacement="turbo_physai.optimizations.models.lerobot_act.channels_last.act_init_wrapper",
    ),
    wrap(
        target="torch.nn.modules.conv.Conv2d.forward",
        replacement="turbo_physai.optimizations.models.lerobot_act.channels_last.conv_forward_wrapper",
    ),
    wrap(
        target="lerobot.optim.optimizers._save_single_optimizer_state",
        replacement="turbo_physai.optimizations.models.lerobot_act.channels_last.optimizer_save_wrapper",
    ),
    wrap(
        target="lerobot.policies.act.modeling_act.ACTPolicy.__init__",
        replacement="turbo_physai.optimizations.models.lerobot_act.backbone_compile.policy_init_wrapper",
    ),
    wrap(
        target="torchvision.models._utils.IntermediateLayerGetter.forward",
        replacement="turbo_physai.optimizations.models.lerobot_act.backbone_compile.compiled_backbone_forward_wrapper",
    ),
)

SAMPLER = group(
    "lerobot_act.sampler_vector",
    wrap(
        target="lerobot.datasets.sampler.EpisodeAwareSampler._iter_epoch",
        replacement="turbo_physai.optimizations.models.lerobot_act.sampler.sampler_wrapper",
    ),
)

FFN_BACKWARD = group(
    "lerobot_act.ffn_backward_fusion",
    wrap(
        target="lerobot.policies.act.modeling_act.ACTEncoderLayer.forward",
        replacement="turbo_physai.optimizations.models.lerobot_act.ffn_fusion.encoder_wrapper",
    ),
    wrap(
        target="lerobot.policies.act.modeling_act.ACTDecoderLayer.forward",
        replacement="turbo_physai.optimizations.models.lerobot_act.ffn_fusion.decoder_wrapper",
    ),
)

ATTENTION_COMPILE = group(
    "lerobot_act.attention_compile",
    wrap(
        target="lerobot.policies.act.modeling_act.ACTEncoder.__init__",
        replacement="turbo_physai.optimizations.models.lerobot_act.attention_compile.encoder_init_wrapper",
    ),
    wrap(
        target="torch.nn.modules.activation.MultiheadAttention.forward",
        replacement="turbo_physai.optimizations.models.lerobot_act.attention_compile.attention_forward_wrapper",
    ),
)

POOL_GATHER = group(
    "lerobot_act.pool_backward_gather",
    wrap(
        target="lerobot.policies.act.modeling_act.ACT.forward",
        replacement="turbo_physai.optimizations.models.lerobot_act.pool_gather.act_forward_wrapper",
    ),
    wrap(
        target="torch.nn.modules.pooling.MaxPool2d.forward",
        replacement="turbo_physai.optimizations.models.lerobot_act.pool_gather.pool_forward_wrapper",
    ),
)

ACT_LAYERNORM_BACKWARD = group(
    "lerobot_act.layernorm_apex_backward",
    wrap(
        target="lerobot.policies.act.modeling_act.ACTPolicy.forward",
        replacement="turbo_physai.optimizations.models.lerobot_act.layer_norm.wrap_act_policy",
    ),
    wrap(
        target="torch.nn.modules.normalization.LayerNorm.forward",
        replacement="turbo_physai.optimizations.models.lerobot_act.layer_norm.wrap_layer_norm",
    ),
)

APEX_FUSED_ADAMW = group(
    "lerobot_act.apex_fused_adamw",
    wrap(
        target="lerobot.optim.optimizers.AdamWConfig.build",
        replacement="turbo_physai.optimizations.models.lerobot_act.adamw.wrap_apex_fused",
    ),
)
