# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
"""Combinable ACT-scoped Apex LayerNorm backward, adapted from v6."""

from contextvars import ContextVar
from functools import wraps

import torch
import torch.nn.functional as F

try:
    import fused_layer_norm_cuda
except (ImportError, OSError):
    fused_layer_norm_cuda = None


_ACT_POLICY_ACTIVE = ContextVar('turbo_lerobot_act_policy_active', default=False)


class _ApexBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, weight, bias, normalized_shape, eps):
        output, mean, rstd = torch.ops.aten.native_layer_norm.default(
            value, normalized_shape, weight, bias, eps
        )
        ctx.save_for_backward(value, weight, bias, mean, rstd)
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx, grad_output):
        value, weight, bias, mean, rstd = ctx.saved_tensors
        if torch.is_grad_enabled():
            tensors = (value, weight, bias)
            needed = ctx.needs_input_grad[:3]
            selected = [tensor for tensor, required in zip(tensors, needed) if required]
            with torch.enable_grad():
                output = F.layer_norm(value, ctx.normalized_shape, weight, bias, ctx.eps)
                selected_grads = torch.autograd.grad(
                    output, selected, grad_output, create_graph=True, allow_unused=True
                )
            iterator = iter(selected_grads)
            gradients = [next(iterator) if required else None for required in needed]
            return *gradients, None, None
        grad_value, grad_weight, grad_bias = fused_layer_norm_cuda.backward_affine(
            grad_output.contiguous(), mean, rstd, value, ctx.normalized_shape,
            weight, bias, ctx.eps,
        )
        return grad_value, grad_weight, grad_bias, None, None


def _is_eligible(module, value):
    if (
        fused_layer_norm_cuda is None
        or torch.compiler.is_compiling()
        or not torch.is_grad_enabled()
        or torch.are_deterministic_algorithms_enabled()
        or not isinstance(value, torch.Tensor)
        or value.device.type != 'cuda'
        or value.dtype != torch.float32
        or value.ndim != 3
        or value.shape[0] != 902
        or value.shape[1] <= 0
        or value.shape[2] != 512
        or not value.is_contiguous()
        or tuple(module.normalized_shape) != (512,)
        or module.weight is None
        or module.bias is None
        or module.weight.dtype != torch.float32
        or module.bias.dtype != torch.float32
        or module.weight.device != value.device
        or module.bias.device != value.device
        or not module.weight.is_contiguous()
        or not module.bias.is_contiguous()
    ):
        return False
    return torch.autograd.forward_ad.unpack_dual(value).tangent is None


def wrap_act_policy(original, options):
    del options
    @wraps(original)
    def wrapped(*args, **kwargs):
        token = _ACT_POLICY_ACTIVE.set(True)
        try:
            return original(*args, **kwargs)
        finally:
            _ACT_POLICY_ACTIVE.reset(token)
    return wrapped


def wrap_layer_norm(original, options):
    del options
    @wraps(original)
    def wrapped(module, value):
        if not _ACT_POLICY_ACTIVE.get() or not _is_eligible(module, value):
            return original(module, value)
        return _ApexBackward.apply(value, module.weight, module.bias, (512,), module.eps)
    return wrapped
