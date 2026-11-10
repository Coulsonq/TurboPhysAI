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

"""ACT FFN backward fusion and optional packed Q/K projection."""

import functools
import math

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


def _qk_packed_attention(mha, query, value, key_padding_mask):
    """Evaluate ACT self-attention with Q and K sharing one projection."""
    seq, batch, embed = query.shape
    bias = mha.in_proj_bias
    qk = F.linear(query, mha.in_proj_weight[: 2 * embed], None if bias is None else bias[: 2 * embed])
    q, k = qk.chunk(2, dim=-1)
    v = F.linear(value, mha.in_proj_weight[2 * embed :], None if bias is None else bias[2 * embed :])
    head_dim = embed // mha.num_heads

    def heads(tensor):
        return tensor.reshape(seq, batch * mha.num_heads, head_dim).transpose(0, 1)

    q, k, v = heads(q), heads(k), heads(v)
    scores = torch.bmm(q / math.sqrt(head_dim), k.transpose(1, 2))
    if key_padding_mask is not None:
        mask = key_padding_mask[:, None, None, :].expand(batch, mha.num_heads, 1, seq)
        mask = mask.reshape(batch * mha.num_heads, 1, seq)
        scores = scores.masked_fill(mask, float("-inf")) if mask.dtype == torch.bool else scores + mask
    weights = F.dropout(torch.softmax(scores, dim=-1), p=mha.dropout, training=mha.training)
    result = torch.bmm(weights, v).transpose(0, 1).contiguous().view(seq * batch, embed)
    return F.linear(result, mha.out_proj.weight, mha.out_proj.bias).view(seq, batch, embed)


def _qk_supported(mha, query, value, key_padding_mask):
    return (
        not mha.batch_first and mha.in_proj_weight is not None
        and mha.bias_k is None and mha.bias_v is None and not mha.add_zero_attn
        and query.ndim == 3 and query.shape == value.shape and query.shape[-1] == mha.embed_dim
        and (key_padding_mask is None or (
            key_padding_mask.shape == (query.shape[1], query.shape[0])
            and (key_padding_mask.dtype == torch.bool or torch.is_floating_point(key_padding_mask))
        ))
    )


def encoder_wrapper(original, options):
    qk_projection = options.get("qk_projection", True)
    if not isinstance(qk_projection, bool):
        raise ValueError("qk_projection must be a boolean")

    @functools.wraps(original)
    def wrapped(self, x, pos_embed=None, key_padding_mask=None):
        if not _eligible(self, x):
            return original(self, x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        if qk_projection and pos_embed is not None and _qk_supported(self.self_attn, q, x, key_padding_mask):
            x = _qk_packed_attention(self.self_attn, q, x, key_padding_mask)
        else:
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
