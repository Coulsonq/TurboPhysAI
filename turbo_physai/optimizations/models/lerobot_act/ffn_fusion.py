# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# The layer forward bodies are adapted from
# https://github.com/huggingface/lerobot/blob/3f2179f3b69708b6ad009b2e7685dd9d05269ee1/src/lerobot/policies/act/modeling_act.py
# The native-forward / fused-backward implementation is also adapted from a
# local ACT-H20 experiment whose ownership and license remain to be confirmed.

"""Fuse only the ReLU/dropout backward pair for validated ACT FP32 shapes."""

import functools

import torch
import torch.nn.functional as F


_SEQUENCE_LENGTHS = {902, 102, 100}
_compiled_backward = None


def _native_backward(gradient, mask, rectified, scale):
    dropped = torch.ops.aten.native_dropout_backward.default(gradient, mask, scale)
    return torch.ops.aten.threshold_backward.default(dropped, rectified, 0)


class _ReluDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, probability):
        rectified = F.relu(tensor)
        output, mask = torch.ops.aten.native_dropout.default(rectified, probability, True)
        ctx.save_for_backward(rectified, mask)
        ctx.scale = 1.0 / (1.0 - probability)
        return output

    @staticmethod
    def backward(ctx, gradient):
        global _compiled_backward
        rectified, mask = ctx.saved_tensors
        if torch.is_grad_enabled():
            result = _native_backward(gradient, mask, rectified, ctx.scale)
        else:
            if _compiled_backward is None:
                _compiled_backward = torch.compile(_native_backward, fullgraph=True, dynamic=False)
            result = _compiled_backward(gradient, mask, rectified, ctx.scale)
        return result, None


def _can_fuse(value, probability, training):
    return (
        value.is_cuda and value.dtype == torch.float32 and value.ndim == 3
        and value.shape[0] in _SEQUENCE_LENGTHS and value.shape[1] > 0 and value.shape[2] == 3200
        and value.is_contiguous() and value.requires_grad and probability == 0.1 and training
        and not torch.compiler.is_compiling()
    )


def _eligible(layer, value):
    return (
        value.is_cuda and value.dtype == torch.float32 and torch.is_grad_enabled()
        and layer.training and layer.activation is F.relu and type(layer.dropout) is torch.nn.Dropout
        and layer.dropout.p == 0.1 and not layer.dropout.inplace and layer.dropout.training
    )


def _ffn(layer, value):
    hidden = layer.linear1(value)
    if _can_fuse(hidden, layer.dropout.p, layer.dropout.training):
        hidden = _ReluDropout.apply(hidden, layer.dropout.p)
    else:
        hidden = layer.dropout(layer.activation(hidden))
    return layer.linear2(hidden)


def encoder_wrapper(original, options):
    del options

    @functools.wraps(original)
    def wrapped(self, x, pos_embed=None, key_padding_mask=None):
        if not _eligible(self, x):
            return original(self, x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)[0]
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = _ffn(self, x)
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x

    return wrapped


def decoder_wrapper(original, options):
    del options

    @functools.wraps(original)
    def wrapped(self, x, encoder_out, decoder_pos_embed=None, encoder_pos_embed=None):
        if not _eligible(self, x):
            return original(self, x, encoder_out, decoder_pos_embed=decoder_pos_embed,
                            encoder_pos_embed=encoder_pos_embed)
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = _ffn(self, x)
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x

    return wrapped
