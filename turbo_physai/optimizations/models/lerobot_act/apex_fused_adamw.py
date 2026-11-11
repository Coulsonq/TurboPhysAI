# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
"""Optional Apex AdamW candidate, preserving unsupported optimizer inputs."""

from dataclasses import asdict
from functools import wraps

import torch


def _prepare_params(params):
    if isinstance(params, (torch.Tensor, dict, set)):
        return params, False
    try:
        items = list(params)
    except TypeError:
        return params, False

    prepared, tensors = [], []
    supported = bool(items)
    grouped = all(isinstance(item, dict) for item in items)
    if any(isinstance(item, dict) for item in items) and not grouped:
        supported = False
    for item in items:
        if not isinstance(item, dict):
            prepared.append(item)
            tensors.append(item)
            continue
        group = dict(item)
        values = group.get('params')
        if values is None or isinstance(values, set):
            prepared.append(group)
            supported = False
            continue
        try:
            values = [values] if isinstance(values, torch.Tensor) else list(values)
        except TypeError:
            prepared.append(group)
            supported = False
            continue
        group['params'] = values
        prepared.append(group)
        tensors.extend(values)
        if 'fused' in group or 'foreach' in group or any(
            group.get(key, False) for key in ('differentiable', 'amsgrad', 'maximize', 'capturable')
        ):
            supported = False
    devices = set()
    for param in tensors:
        if not isinstance(param, torch.Tensor) or param.device.type != 'cuda' or param.dtype != torch.float32:
            supported = False
        else:
            devices.add(param.device)
    return prepared, supported and bool(tensors) and len(devices) == 1


def _require_all_gradients(optimizer, args, kwargs):
    # Apex has group-level step counters; AdamW has per-parameter counters.
    if any(param.grad is None for group in optimizer.param_groups for param in group['params']):
        raise RuntimeError('Apex FusedAdam received a missing gradient; disable the Apex AdamW group.')


def wrap_apex_fused(original, options):
    del options

    @wraps(original)
    def wrapped(config, params):
        prepared, supported = _prepare_params(params)
        kwargs = asdict(config)
        kwargs.pop('grad_clip_norm')
        if any(key in kwargs for key in ('fused', 'foreach', 'differentiable', 'amsgrad', 'maximize', 'capturable')):
            supported = False
        if not supported:
            return original(config, prepared)
        from apex.optimizers import FusedAdam

        optimizer = FusedAdam(prepared, adam_w_mode=True, bias_correction=True, **kwargs)
        for group in optimizer.param_groups:
            group.setdefault('step', 0)
        optimizer.register_step_pre_hook(_require_all_gradients)
        return optimizer

    return wrapped
