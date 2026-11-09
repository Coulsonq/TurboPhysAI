# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental HCU ACT max-pool backward gather, inspired by ACT-H20."""

import functools

import torch
import triton
import triton.language as tl


_MARKER = "_turbo_lerobot_act_pool_gather"


@triton.jit
def _pool_backward_gather(indices, grad, output, N: tl.constexpr, C: tl.constexpr,
                          H: tl.constexpr, W: tl.constexpr, BLOCK: tl.constexpr):
    position = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = position < N * C * H * W
    channel = position % C
    column = (position // C) % W
    row = (position // (C * W)) % H
    batch = position // (C * W * H)
    OH: tl.constexpr = (H + 1) // 2
    OW: tl.constexpr = (W + 1) // 2
    result = tl.full((BLOCK,), 0.0, tl.float32)
    for dy in tl.static_range(2):
        for dx in tl.static_range(2):
            out_y = row // 2 + dy
            out_x = column // 2 + dx
            active = valid & (out_y < OH) & (out_x < OW)
            active = active & ((dy == 0) | (row % 2 == 1))
            active = active & ((dx == 0) | (column % 2 == 1))
            offset = ((batch * OH + out_y) * OW + out_x) * C + channel
            selected = tl.load(indices + offset, mask=active, other=-1) == row * W + column
            value = tl.load(grad + offset, mask=active & selected, other=0.0)
            result += value
    tl.store(output + position, result, mask=valid)


class _PoolGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        output, indices = torch.ops.aten.max_pool2d_with_indices.default(
            value, [3, 3], [2, 2], [1, 1], [1, 1], False
        )
        ctx.save_for_backward(value, indices)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        value, indices = ctx.saved_tensors
        if torch.is_grad_enabled():
            return torch.ops.aten.max_pool2d_with_indices_backward.default(
                grad_output, value, [3, 3], [2, 2], [1, 1], [1, 1], False, indices
            )
        n, c, h, w = value.shape
        indices = indices.contiguous(memory_format=torch.channels_last)
        grad_output = grad_output.contiguous(memory_format=torch.channels_last)
        output = torch.empty_like(value, memory_format=torch.channels_last)
        _pool_backward_gather[(triton.cdiv(value.numel(), 256),)](
            indices, grad_output, output, N=n, C=c, H=h, W=w, BLOCK=256, num_warps=4
        )
        return output


def act_forward_wrapper(original, options):
    del options

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        backbone = getattr(self, "backbone", None)
        pool = getattr(backbone, "maxpool", None)
        if pool is not None:
            setattr(pool, _MARKER, True)
        return original(self, *args, **kwargs)

    return wrapped


def _pair(value):
    if value is None:
        return None
    return (value, value) if isinstance(value, int) else tuple(value)


def pool_forward_wrapper(original, options):
    del options

    @functools.wraps(original)
    def wrapped(self, value):
        eligible = (
            getattr(self, _MARKER, False)
            and self.training
            and _pair(self.kernel_size) == (3, 3)
            and _pair(self.stride) == (2, 2)
            and _pair(self.padding) == (1, 1)
            and _pair(self.dilation) == (1, 1)
            and not self.ceil_mode
            and not self.return_indices
            and value.ndim == 4
            and value.shape[0] > 0
            and tuple(value.shape[1:]) == (64, 240, 320)
            and value.numel() < 2**31
            and value.is_cuda
            and value.dtype == torch.float32
            and value.requires_grad
            and value.is_contiguous(memory_format=torch.channels_last)
        )
        return _PoolGather.apply(value) if eligible else original(self, value)

    return wrapped
