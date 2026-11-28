# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import copy

import pytest
import torch

def vision_model():
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

    config = Qwen3VLVisionConfig(depth=2, hidden_size=16, intermediate_size=32,
        num_heads=2, patch_size=2, temporal_patch_size=1, spatial_merge_size=2,
        out_hidden_size=16, num_position_embeddings=16, deepstack_visual_indexes=[0])
    config._attn_implementation = "sdpa"
    return Qwen3VLVisionModel(config).train()


@pytest.mark.parametrize("grid", [[[1, 4, 4]], [[2, 4, 4], [1, 2, 4]]])
def test_vision_preparation_preserves_outputs_and_all_parameter_gradients(grid):
    from turbo_physai.optimizations.models.alpamayo2.qwen_regions import prepare_vision_metadata, vision_region
    torch.manual_seed(31)
    eager = vision_model()
    candidate = copy.deepcopy(eager)
    grid = torch.tensor(grid)
    tokens = int(grid.prod(-1).sum())
    pixels = torch.randn(tokens, 12)
    metadata = prepare_vision_metadata(candidate, grid, pixels.device, torch.float32)
    expected = eager(pixels, grid)
    actual = vision_region(candidate, pixels, metadata)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    (expected[0].square().sum() + sum(t.square().sum() for t in expected[1])).backward()
    (actual[0].square().sum() + sum(t.square().sum() for t in actual[1])).backward()
    for (name, p), (_, q) in zip(eager.named_parameters(), candidate.named_parameters()):
        assert p.grad is not None, name
        torch.testing.assert_close(q.grad, p.grad, msg=name)
    # Metadata must not contain a cached learned positional embedding.
    with torch.no_grad():
        candidate.pos_embed.weight.add_(0.1)
    assert not torch.allclose(vision_region(candidate, pixels, metadata)[0], actual[0])


def reference_varlen(q, k, v, cu, causal):
    token = torch.arange(q.shape[0], device=q.device)
    segment = torch.bucketize(token, cu[1:], right=True)
    mask = segment[:, None] == segment[None, :]
    if causal:
        mask = mask & (token[:, None] >= token[None, :])
    out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
        attn_mask=mask[None, None], enable_gqa=True)
    return out[0].transpose(0, 1)


def test_precomputed_padding_preserves_attention_and_input_gradients():
    from turbo_physai.optimizations.models.alpamayo2.qwen_regions import prepare_text_metadata, flash_metadata_wrapper
    torch.manual_seed(5)
    mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    meta = prepare_text_metadata(mask, 2, 4, mask.device)
    source = [torch.randn(2, 4, 2, 8, requires_grad=True) for _ in range(3)]
    other = [x.detach().clone().requires_grad_() for x in source]

    def original(q, k, v, attention_mask, query_length, is_causal, **kwargs):
        assert attention_mask is None and kwargs["position_ids"] is None
        result = reference_varlen(q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1),
                                 kwargs["cu_seq_lens_q"], is_causal)
        return result.reshape(q.shape)

    actual = flash_metadata_wrapper(original, {})(*source, mask, query_length=4,
        is_causal=True, _alpamayo_fa_metadata=meta)
    q, k, v = [x.transpose(1, 2) for x in other]
    causal = torch.ones(4, 4, dtype=torch.bool).tril()[None, None]
    valid = mask[:, None, None, :].bool() & causal
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=valid)
    expected = expected.transpose(1, 2) * mask[:, :, None, None]
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for x, y in zip(source, other):
        torch.testing.assert_close(x.grad, y.grad)


def test_empty_sequences_are_rejected_before_capture():
    from turbo_physai.optimizations.models.alpamayo2.qwen_regions import prepare_text_metadata
    with pytest.raises(ValueError, match="empty"):
        prepare_text_metadata(torch.tensor([[0, 0], [1, 1]]), 2, 2, torch.device("cpu"))


def test_flash_autograd_metadata_inputs_have_distinct_identity_and_equal_values():
    from turbo_physai.optimizations.models.alpamayo2.qwen_regions import prepare_text_metadata, flash_metadata_wrapper
    mask = torch.tensor([[0, 1, 1]])
    meta = prepare_text_metadata(mask, 1, 3, mask.device)
    def original(q, k, v, attention_mask, query_length, is_causal, **kwargs):
        assert kwargs["cu_seq_lens_q"] is not kwargs["cu_seq_lens_k"]
        torch.testing.assert_close(kwargs["cu_seq_lens_q"], kwargs["cu_seq_lens_k"], atol=0, rtol=0)
        return q
    x = [torch.randn(1, 3, 2, 8) for _ in range(3)]
    flash_metadata_wrapper(original, {})(*x, mask, 3, True, _alpamayo_fa_metadata=meta)


def test_checkpointed_qwen_region_is_one_real_inductor_graph():
    from turbo_physai.optimizations.models.alpamayo2.qwen_regions import text_region
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

    torch.manual_seed(17)
    config = Qwen3VLTextConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]})
    config._attn_implementation = "sdpa"
    eager = Qwen3VLTextModel(config).train()
    eager.gradient_checkpointing_enable({"use_reentrant": False})
    candidate = copy.deepcopy(eager)
    x = torch.randn(1, 8, 32, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    visual_indices = torch.tensor([2, 3])
    deep = [torch.randn(2, 32, requires_grad=True)]
    deep2 = [t.detach().clone().requires_grad_() for t in deep]
    pos = torch.arange(8)[None, None].expand(3, 1, -1)
    mask = torch.ones(8, 8, dtype=torch.bool).tril()[None, None]
    graphs = []

    def backend(gm, inputs):
        graphs.append(gm)
        return torch._inductor.compile(gm, inputs)

    compiled = torch.compile(text_region, fullgraph=True, backend=backend)
    visual_mask = torch.zeros(1, 8, dtype=torch.bool)
    visual_mask[0, visual_indices] = True
    expected = eager(inputs_embeds=x, attention_mask=mask, position_ids=pos,
        use_cache=False, visual_pos_masks=visual_mask, deepstack_visual_embeds=deep).last_hidden_state
    actual = compiled(candidate, y, mask, pos, pos[0], torch.arange(8),
                      visual_indices, deep2, None)
    torch.testing.assert_close(actual, expected)
    expected.square().sum().backward()
    actual.square().sum().backward()
    assert len(graphs) == 1
    assert any("tag_activation_checkpoint" in str(n.target) for n in graphs[0].graph.nodes)
    torch.testing.assert_close(y.grad, x.grad)
    torch.testing.assert_close(deep2[0].grad, deep[0].grad)
    for (name, p), (_, q) in zip(eager.named_parameters(), candidate.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(q.grad, p.grad, atol=2e-5, rtol=2e-4, msg=name)


def test_checkpointed_complete_encoder_is_one_real_inductor_graph(monkeypatch):
    from turbo_physai.optimizations.models.alpamayo2.qwen_regions import (
        prepare_vision_metadata, vision_region, flash_metadata_wrapper)
    import transformers.modeling_flash_attention_utils as fa
    import transformers.integrations.flash_attention as integration

    def varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal, **kw):
        return reference_varlen(q, k, v, cu_seqlens_q, causal)

    def padded(q, k, v, causal, **kw):
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=causal, enable_gqa=True).transpose(1, 2)

    monkeypatch.setattr(fa, "lazy_import_flash_attention", lambda *args: (
        (padded, varlen, None, None), lambda **kw: {"causal": kw["is_causal"]}))
    torch.manual_seed(29)
    eager = vision_model().to(dtype=torch.bfloat16)
    eager.config._attn_implementation = "flash_attention_2"
    eager.gradient_checkpointing_enable({"use_reentrant": False})
    candidate = copy.deepcopy(eager)
    grid = torch.tensor([[2, 4, 4], [1, 2, 4]])
    pixels = torch.randn(40, 12, dtype=torch.bfloat16)
    metadata = prepare_vision_metadata(candidate, grid, pixels.device, torch.bfloat16)
    expected = eager(pixels, grid)
    monkeypatch.setattr(integration, "_flash_attention_forward", flash_metadata_wrapper(fa._flash_attention_forward, {}))
    prepared = vision_region(candidate, pixels, metadata)
    torch.testing.assert_close(prepared[0], expected[0], atol=0, rtol=0)
    torch.testing.assert_close(prepared[1], expected[1], atol=0, rtol=0)
    graphs = []

    def backend(gm, inputs):
        graphs.append(gm)
        with torch._inductor.config.patch(emulate_precision_casts=True):
            return torch._inductor.compile(gm, inputs)

    compiled = torch.compile(vision_region, backend=backend, fullgraph=True)
    actual = compiled(candidate, pixels, metadata)
    torch.testing.assert_close(actual[0], expected[0], atol=0.004, rtol=0.016)
    torch.testing.assert_close(actual[1], expected[1], atol=0.004, rtol=0.016)
    (expected[0].square().sum() + expected[1][0].square().sum()).backward()
    (actual[0].square().sum() + actual[1][0].square().sum()).backward()
    assert len(graphs) == 1
    assert sum("tag_activation_checkpoint" in str(n.target) for n in graphs[0].graph.nodes) == 2
    for (name, p), (_, q) in zip(eager.named_parameters(), candidate.named_parameters()):
        torch.testing.assert_close(q.grad, p.grad, atol=2e-2, rtol=2e-2, msg=name)


@pytest.mark.parametrize("region", ["encoder", "qwen"])
def test_vendor_fa2_is_traced_through_checkpoint_into_aot_graphs(monkeypatch, region):
    """Capture the installed vendor forward/backward without launching HCU kernels."""
    from torch._dynamo.backends.common import aot_autograd
    from torch._functorch.partitioners import min_cut_rematerialization_partition
    from turbo_physai.optimizations.models.alpamayo2.qwen_regions import (
        prepare_vision_metadata, vision_region, prepare_text_metadata, text_region,
        flash_metadata_wrapper, flash_import_wrapper)
    import transformers.modeling_flash_attention_utils as fa
    import transformers.integrations.flash_attention as integration

    fa.lazy_import_flash_attention("flash_attention_2")
    monkeypatch.setattr(fa, "lazy_import_flash_attention", flash_import_wrapper(fa.lazy_import_flash_attention, {}))
    monkeypatch.setattr(integration, "_flash_attention_forward", flash_metadata_wrapper(fa._flash_attention_forward, {}))
    if region == "encoder":
        model = vision_model().to(dtype=torch.bfloat16)
        pixels = torch.randn(16, 12, dtype=torch.bfloat16)
        metadata = prepare_vision_metadata(model, torch.tensor([[1, 4, 4]]), pixels.device, pixels.dtype)
        function, inputs = vision_region, (model, pixels, metadata)
    else:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
        config = Qwen3VLTextConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]})
        config._attn_implementation = "sdpa"
        model = Qwen3VLTextModel(config).train().to(dtype=torch.bfloat16)
        hidden = torch.randn(1, 8, 32, dtype=torch.bfloat16, requires_grad=True)
        pos = torch.arange(8)[None, None].expand(3, 1, -1)
        metadata = prepare_text_metadata(None, 1, 8, hidden.device)
        function, inputs = text_region, (model, hidden, None, pos, pos[0], torch.arange(8), None, None, metadata)
    model.config._attn_implementation = "flash_attention_2"
    model.gradient_checkpointing_enable({"use_reentrant": False})
    graphs = {}

    def partition(joint, inputs, **kwargs):
        fw, bw = min_cut_rematerialization_partition(joint, inputs, **kwargs)
        graphs["fw"], graphs["bw"] = fw, bw
        return fw, bw

    def stop_before_execution(gm, inputs):
        raise RuntimeError("vendor FA graph captured; do not execute CPU kernels")

    aot = aot_autograd(fw_compiler=stop_before_execution, partition_fn=partition)

    def backend(gm, inputs):
        graphs["dynamo"] = gm
        return aot(gm, inputs)

    with pytest.raises(torch._dynamo.exc.BackendCompilerFailed, match="vendor FA graph captured"):
        torch.compile(function, backend=backend, fullgraph=True, dynamic=True)(*inputs)
    assert sum("tag_activation_checkpoint" in str(n.target) for n in graphs["dynamo"].graph.nodes) == 2
    assert any("flash_attn2_c_op.varlen_fwd" in str(n.target) for n in graphs["fw"].graph.nodes)
    backward_op = torch.ops.flash_attn._flash_attn_varlen_backward.default
    assert any(n.target is backward_op or any(a is backward_op for a in n.args)
               for gm in graphs["bw"].modules() if isinstance(gm, torch.fx.GraphModule)
               for n in gm.graph.nodes), graphs["bw"].code


def test_disabling_compile_keeps_original_flash_imports(monkeypatch):
    from turbo_physai.optimizations.models.alpamayo2.qwen_regions import flash_import_wrapper
    monkeypatch.setenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "1")
    result = ((object(), object(), object(), object()), object())
    original = lambda *args, **kwargs: result
    assert flash_import_wrapper(original, {})("flash_attention_2") is result
