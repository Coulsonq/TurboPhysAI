# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
"""Compose ACT deferred metrics and an experimental scoped LayerNorm path."""

from . import apex_layernorm, reference_training


def _switch(options, name, default):
    value = options.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def wrap_policy(original, options):
    deferred = _switch(options, "deferred_metrics", True)
    layernorm = _switch(options, "layernorm_apex", False)
    target = reference_training.act_policy_forward if deferred else original
    return apex_layernorm.wrap_act_policy(target, {}) if layernorm else target


def wrap_layer_norm(original, options):
    _switch(options, "deferred_metrics", True)
    layernorm = _switch(options, "layernorm_apex", False)
    return apex_layernorm.wrap_layer_norm(original, {}) if layernorm else original
