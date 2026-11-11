# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-Python catalog declarations for LeRobot ACT HCU optimizations."""

from __future__ import annotations

from ....engine.definitions import group, wrap


BASE = "turbo_physai.optimizations.models.lerobot_act"
ACT = "lerobot.policies.act.modeling_act"


CHANNELS_LAST = group(
    "lerobot_act.reference_channels_last",
    wrap(
        target=f"{ACT}.ACT.__init__",
        replacement=f"{BASE}.reference_model.act_init_wrapper",
    ),
    wrap(
        target="torch.nn.modules.conv.Conv2d.forward",
        replacement=f"{BASE}.reference_model.conv_forward_wrapper",
    ),
    wrap(
        target="lerobot.optim.optimizers._save_single_optimizer_state",
        replacement=f"{BASE}.reference_model.optimizer_save_wrapper",
    ),
)

BACKBONE = group(
    "lerobot_act.reference_backbone_compile",
    wrap(
        target=f"{ACT}.ACTPolicy.__init__",
        replacement=f"{BASE}.reference_model.policy_init_wrapper",
    ),
    wrap(
        target="torchvision.models._utils.IntermediateLayerGetter.forward",
        replacement=f"{BASE}.reference_model.backbone_forward_wrapper",
    ),
)

NO_ATTN_AVERAGE = group(
    "lerobot_act.reference_no_attn_average",
    wrap(
        target=f"{ACT}.ACTEncoder.__init__",
        replacement=f"{BASE}.reference_model.encoder_init_wrapper",
    ),
    wrap(
        target=f"{ACT}.ACTDecoder.__init__",
        replacement=f"{BASE}.reference_model.decoder_init_wrapper",
    ),
    wrap(
        target="torch.nn.modules.activation.MultiheadAttention.forward",
        replacement=f"{BASE}.reference_model.attention_forward_wrapper",
    ),
)

FFN = group(
    "lerobot_act.reference_ffn_compile",
    wrap(
        target=f"{ACT}.ACTEncoderLayer.forward",
        replacement=f"{BASE}.reference_model.encoder_ffn_wrapper",
    ),
    wrap(
        target=f"{ACT}.ACTDecoderLayer.forward",
        replacement=f"{BASE}.reference_model.decoder_ffn_wrapper",
    ),
)

ATTENTION_PROBS = group(
    "lerobot_act.reference_attention_probs",
    wrap(
        target="torch.nn.functional.multi_head_attention_forward",
        replacement=f"{BASE}.reference_attention.wrap_attention",
    ),
)

DDP = group(
    "lerobot_act.reference_ddp",
    wrap(
        target="torch.nn.parallel.distributed.DistributedDataParallel.__init__",
        replacement=f"{BASE}.reference_training.ddp_kwargs_wrapper",
    ),
)

H2D = group(
    "lerobot_act.reference_nonblocking_h2d",
    wrap(
        target="accelerate.accelerator.Accelerator.__init__",
        replacement=f"{BASE}.reference_training.accelerator_wrapper",
    ),
)

ASYNC_SCALARS = group(
    "lerobot_act.reference_async_scalars",
    wrap(
        target=f"{ACT}.ACTPolicy.forward",
        replacement=f"{BASE}.policy_metrics_layernorm.wrap_policy",
    ),
    wrap(
        target="torch.nn.modules.normalization.LayerNorm.forward",
        replacement=f"{BASE}.policy_metrics_layernorm.wrap_layer_norm",
    ),
)

SAMPLER_VECTOR = group(
    "lerobot_act.sampler_vector",
    wrap(
        target="lerobot.datasets.sampler.EpisodeAwareSampler._iter_epoch",
        replacement=f"{BASE}.sampler_vector.sampler_wrapper",
    ),
)

VAE_NO_CLONE = group(
    "lerobot_act.vae_no_clone",
    wrap(
        target=f"{ACT}.ACT.forward",
        replacement=f"{BASE}.vae_no_clone.vae_forward_wrapper",
    ),
)

APEX_FUSED_ADAMW = group(
    "lerobot_act.apex_fused_adamw",
    wrap(
        target="lerobot.optim.optimizers.AdamWConfig.build",
        replacement=f"{BASE}.apex_fused_adamw.wrap_apex_fused",
    ),
)

POOL_GATHER = group(
    "lerobot_act.pool_backward_gather",
    wrap(
        target="torch.nn.modules.pooling.MaxPool2d.forward",
        replacement=f"{BASE}.pool_backward_gather.pool_forward_wrapper",
    ),
    depends_on=("lerobot_act.reference_channels_last",),
)
