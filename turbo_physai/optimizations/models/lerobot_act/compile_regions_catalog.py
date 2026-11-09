# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Optional region-level compilation candidates for LeRobot ACT."""

from ....engine.definitions import group, wrap


ENCODER_REGIONS = group(
    "lerobot_act.encoder_regions_compile",
    wrap(
        target="lerobot.policies.act.modeling_act.ACTEncoder.forward",
        replacement="turbo_physai.optimizations.models.lerobot_act.compile_regions.wrap_encoder",
    ),
)

DECODER_REGION = group(
    "lerobot_act.decoder_region_compile",
    wrap(
        target="lerobot.policies.act.modeling_act.ACTDecoder.forward",
        replacement="turbo_physai.optimizations.models.lerobot_act.compile_regions.wrap_decoder",
    ),
)

FFN_REGION = group(
    "lerobot_act.ffn_region_compile",
    wrap(
        target="turbo_physai.optimizations.models.lerobot_act.ffn_fusion._ffn",
        replacement="turbo_physai.optimizations.models.lerobot_act.compile_regions.wrap_ffn",
    ),
    depends_on=("lerobot_act.ffn_backward_fusion",),
)
