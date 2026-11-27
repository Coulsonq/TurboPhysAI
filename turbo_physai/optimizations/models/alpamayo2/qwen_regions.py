# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Modified by Hygon. Adapted from HuggingFace Transformers v4.57.1,
# src/transformers/models/qwen3_vl/modeling_qwen3_vl.py, commit
# a47e6666f8f9a61e44a5d149c5e3d230e6c8053c.

"""Complete Qwen3-VL compute regions; batch metadata is prepared outside Dynamo.

No learned value is cached. The original attention projections, FlashAttention
kernel, block checkpoint calls, positional embedding and normalization are used.
"""

from functools import wraps
from typing import NamedTuple

import torch


class AttentionMetadata(NamedTuple):
    indices: torch.Tensor | None
    cu_seqlens: torch.Tensor
    max_length: int
    cu_seqlens_k: torch.Tensor


class VisionMetadata(NamedTuple):
    indices: torch.Tensor
    weights: torch.Tensor
    order: torch.Tensor
    rotary: torch.Tensor
    attention: AttentionMetadata


def prepare_vision_metadata(model, grid_thw, device, dtype):
    """Prepare only non-trainable geometry, with upstream interpolation rounding."""
    grids = grid_thw.detach().to(device="cpu", dtype=torch.long).tolist()
    merge = model.spatial_merge_size
    side = model.num_grid_per_side
    indices = [[] for _ in range(4)]
    weights = [[] for _ in range(4)]
    orders, coords, lengths = [], [], []
    offset = 0
    if not grids:
        raise ValueError("empty vision grid")
    for t, h, w in grids:
        if min(t, h, w) <= 0 or h % merge or w % merge:
            raise ValueError("vision grid must be positive and divisible by spatial_merge_size")
        hi, wi = torch.linspace(0, side - 1, h), torch.linspace(0, side - 1, w)
        hf, wf = hi.int(), wi.int()
        hc, wc = (hf + 1).clip(max=side - 1), (wf + 1).clip(max=side - 1)
        dh, dw = hi - hf, wi - wf
        corners = [(hf * side, wf), (hf * side, wc), (hc * side, wf), (hc * side, wc)]
        coeffs = [(1 - dh, 1 - dw), (1 - dh, dw), (dh, 1 - dw), (dh, dw)]
        for i, ((row, col), (a, b)) in enumerate(zip(corners, coeffs)):
            indices[i].extend((row[:, None] + col[None, :]).flatten().tolist())
            weights[i].extend((a[:, None] * b[None, :]).flatten().tolist())
        order = (torch.arange(h * w) + offset).repeat(t)
        orders.append(order.view(t, h // merge, merge, w // merge, merge)
                      .permute(0, 1, 3, 2, 4).flatten())
        rows = torch.arange(h).view(h // merge, merge)
        cols = torch.arange(w).view(w // merge, merge)
        rr = rows[:, None, :, None].expand(h // merge, w // merge, merge, merge)
        cc = cols[None, :, None, :].expand(h // merge, w // merge, merge, merge)
        coords.append(torch.stack((rr.flatten(), cc.flatten()), -1).repeat(t, 1))
        lengths.extend([h * w] * t)
        offset += h * w
    coords = torch.cat(coords).to(device)
    freq = model.rotary_pos_emb(max(max(h, w) for _, h, w in grids))
    rotary = freq[coords].flatten(1)
    cu = torch.tensor([0] + lengths, device=device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    return VisionMetadata(
        torch.tensor(indices, dtype=torch.long, device=device),
        torch.tensor(weights, dtype=dtype, device=device),
        torch.cat(orders).to(device), rotary,
        AttentionMetadata(None, cu, max(lengths), cu.clone()))


def vision_region(model, hidden_states, metadata, **kwargs):
    """Patch embedding + every Encoder block + all DeepStack/patch mergers."""
    hidden_states = model.patch_embed(hidden_states)
    positions = model.pos_embed(metadata.indices) * metadata.weights[:, :, None]
    positions = positions[0] + positions[1] + positions[2] + positions[3]
    hidden_states = hidden_states + positions.index_select(0, metadata.order)
    seq_len = hidden_states.shape[0]
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary = metadata.rotary.reshape(seq_len, -1)
    emb = torch.cat((rotary, rotary), -1)
    position_embeddings = (emb.cos(), emb.sin())
    deepstack = []
    for index, block in enumerate(model.blocks):
        hidden_states = block(hidden_states, cu_seqlens=metadata.attention.cu_seqlens,
            position_embeddings=position_embeddings,
            _alpamayo_fa_metadata=metadata.attention, **kwargs)
        if index in model.deepstack_visual_indexes:
            merger = model.deepstack_merger_list[model.deepstack_visual_indexes.index(index)]
            deepstack.append(merger(hidden_states))
    return model.merger(hidden_states), deepstack


def prepare_text_metadata(attention_mask, batch_size, seq_len, device):
    """Prepare the same unpadding indices/cu/max once, rather than in every layer."""
    if attention_mask is None:
        lengths = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
        indices = None
        max_length = seq_len
    else:
        if attention_mask.ndim != 2 or tuple(attention_mask.shape) != (batch_size, seq_len):
            raise ValueError("compiled FA requires the official two-dimensional padding mask")
        if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
            raise ValueError("compiled FA requires a binary padding mask")
        lengths = attention_mask.sum(-1, dtype=torch.int32)
        if bool((lengths == 0).any()):
            raise ValueError("compiled FA does not support empty sequences")
        max_length = int(lengths.max().item())
        indices = attention_mask.flatten().nonzero(as_tuple=False).flatten().to(device)
    cu = torch.nn.functional.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
    # Dynamo's autograd.Function HOP requires distinct tensor input identities.
    # Query/key lengths have identical values, but must not be the same object.
    return AttentionMetadata(indices, cu, max_length, cu.clone())


@torch.compiler.allow_in_graph
def _vendor_flash_varlen(*args, **kwargs):
    # Keep the legacy autograd.Function out of Dynamo's nested checkpoint tracer.
    # AOTAutograd still traces its original forward/backward into vendor custom
    # ops. Every tensor is an explicit argument; no parameters or state are closed over.
    from flash_attn import flash_attn_varlen_func
    return flash_attn_varlen_func(*args, **kwargs)


def flash_import_wrapper(original, options):
    from .zero3_module_regions import disabled
    from flash_attn import flash_attn_varlen_func

    @wraps(original)
    def wrapped(implementation=None, *args, **kwargs):
        result = original(implementation, *args, **kwargs)
        if disabled():
            return result
        (padded, varlen, pad, unpad), process_kwargs = result
        if varlen is not flash_attn_varlen_func:
            return result
        # Preserve signature-derived kwargs from the actual installed FA library.
        return (padded, _vendor_flash_varlen, pad, unpad), process_kwargs
    return wrapped


def flash_metadata_wrapper(original, options):
    """Use original vendor FA; move only padding/scalar preparation out of graphs."""
    @wraps(original)
    def wrapped(q, k, v, attention_mask, query_length, is_causal, *args,
                _alpamayo_fa_metadata=None, **kwargs):
        meta = _alpamayo_fa_metadata
        if meta is None:
            return original(q, k, v, attention_mask, query_length, is_causal, *args, **kwargs)
        kwargs.update(position_ids=None, cu_seq_lens_q=meta.cu_seqlens,
            cu_seq_lens_k=meta.cu_seqlens_k, max_length_q=meta.max_length, max_length_k=meta.max_length)
        if meta.indices is None:
            return original(q, k, v, None, query_length, is_causal, *args, **kwargs)
        shape = q.shape
        q = q.flatten(0, 1).index_select(0, meta.indices).unsqueeze(0)
        k = k.flatten(0, 1).index_select(0, meta.indices).unsqueeze(0)
        v = v.flatten(0, 1).index_select(0, meta.indices).unsqueeze(0)
        output = original(q, k, v, None, query_length, is_causal, *args, **kwargs)
        flat = output.reshape(-1, shape[-2], shape[-1])
        return flat.new_zeros((shape[0] * shape[1], shape[-2], shape[-1])).index_copy(
            0, meta.indices, flat).reshape(shape)
    return wrapped


def text_region(model, hidden_states, attention_mask, position_ids, text_position_ids,
                cache_position, visual_indices, deepstack_visual_embeds, fa_metadata, **kwargs):
    """Shared RoPE + every checkpointed Qwen decoder block + final norm."""
    position_embeddings = model.rotary_emb(hidden_states, position_ids)
    for index, layer in enumerate(model.layers):
        hidden_states = layer(hidden_states, attention_mask=attention_mask,
            position_ids=text_position_ids, past_key_values=None, cache_position=cache_position,
            position_embeddings=position_embeddings, _alpamayo_fa_metadata=fa_metadata, **kwargs)
        if deepstack_visual_embeds is not None and index < len(deepstack_visual_embeds):
            features = deepstack_visual_embeds[index].to(hidden_states.device, hidden_states.dtype)
            flat = hidden_states.flatten(0, 1)
            updates = flat.index_select(0, visual_indices).clone() + features
            hidden_states = flat.index_copy(0, visual_indices, updates).reshape(hidden_states.shape)
    return model.norm(hidden_states)


def vision_forward_wrapper(original, options):
    @wraps(original)
    def wrapped(model, hidden_states, grid_thw, **kwargs):
        region = getattr(model, "_alpamayo_compiled_region", None)
        if region is None:
            return original(model, hidden_states, grid_thw, **kwargs)
        metadata = prepare_vision_metadata(model, grid_thw, hidden_states.device,
                                           model._alpamayo_region_dtype)
        with torch.profiler.record_function("ALPA_COMPILE/encoder"):
            return region(model, hidden_states, metadata, **kwargs)
    return wrapped


def text_forward_wrapper(original, options):
    @wraps(original)
    def wrapped(model, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, cache_position=None, visual_pos_masks=None,
                deepstack_visual_embeds=None, **kwargs):
        region = getattr(model, "_alpamayo_compiled_region", None)
        if region is None:
            return original(model, input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
                cache_position=cache_position, visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds, **kwargs)
        from transformers.modeling_outputs import BaseModelOutputWithPast
        from transformers.masking_utils import create_causal_mask

        if use_cache or past_key_values is not None:
            raise ValueError("Alpamayo module compile supports Stage-1 use_cache=False training only")
        for name in ("output_attentions", "output_hidden_states"):
            if kwargs.pop(name, False) or getattr(model.config, name, False):
                raise ValueError(f"compiled Stage-1 does not support {name}")
        if kwargs.pop("return_dict", True) is False:
            raise ValueError("compiled Stage-1 requires return_dict=True")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("specify exactly one of input_ids and inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = model.embed_tokens(input_ids)
        batch, length = inputs_embeds.shape[:2]
        if cache_position is None:
            cache_position = torch.arange(length, device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, batch, -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None].expand(3, -1, -1)
        if position_ids.shape[0] == 4:
            text_positions, position_ids = position_ids[0], position_ids[1:]
        else:
            text_positions = position_ids[0]
        # An absent mask plus reset positions has packing semantics in upstream FA.
        if attention_mask is None and bool((text_positions[:, 1:] < text_positions[:, :-1]).any()):
            raise ValueError("packed text positions are outside the Stage-1 compile contract")
        fa_metadata = prepare_text_metadata(attention_mask, batch, length, inputs_embeds.device)
        mask = create_causal_mask(config=model.config, input_embeds=inputs_embeds,
            attention_mask=attention_mask, cache_position=cache_position, past_key_values=None,
            position_ids=text_positions)
        visual_indices = None
        if deepstack_visual_embeds is not None:
            if visual_pos_masks is None:
                raise ValueError("DeepStack features require visual_pos_masks")
            visual_indices = visual_pos_masks.to(inputs_embeds.device).flatten().nonzero().flatten()
            if any(t.shape[0] != visual_indices.numel() for t in deepstack_visual_embeds):
                raise ValueError("DeepStack feature count differs from visual positions")
        with torch.profiler.record_function("ALPA_COMPILE/qwen"):
            result = region(model, inputs_embeds, mask, position_ids, text_positions,
                cache_position, visual_indices, deepstack_visual_embeds, fa_metadata, **kwargs)
        return BaseModelOutputWithPast(last_hidden_state=result, past_key_values=None)
    return wrapped


def parameter_forward_wrapper(original, options):
    """Only the marked embedding and LM head; other Linear/Embedding calls retain original behavior."""
    @wraps(original)
    def wrapped(model, *args, **kwargs):
        region = getattr(model, "_alpamayo_compiled_parameter_region", None)
        if region is None:
            return original(model, *args, **kwargs)
        return region(model, *args, **kwargs)
    return wrapped
