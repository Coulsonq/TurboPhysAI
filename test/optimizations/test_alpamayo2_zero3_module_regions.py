# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial
from types import SimpleNamespace

import pytest
import torch


def test_unsupported_zero_is_rejected_before_native_mutation():
    from turbo_physai.optimizations.models.alpamayo2.zero3_module_regions import validate_engine
    engine = SimpleNamespace(zero_optimization_stage=lambda: 2)
    with pytest.raises(ValueError, match="ZeRO-3"):
        validate_engine(engine)


def test_native_initializer_preserves_existing_gradient_group():
    from turbo_physai.optimizations.models.alpamayo2.zero3_module_regions import native_init_wrapper
    group = SimpleNamespace(group_name="existing-gradient-group")
    calls = []
    dc = SimpleNamespace(set_reduce_process_group=lambda pg: calls.append(pg),
                         get_reduce_group_name=lambda: group.group_name)
    engine = SimpleNamespace(optimizer=SimpleNamespace(_alpamayo_gradient_reduce_group=group), module=object())
    assert native_init_wrapper(lambda *args: dc, {})(engine, object()) is dc
    assert calls == [group]


def test_unpatched_native_library_is_rejected_before_removing_zero_hooks():
    from turbo_physai.optimizations.models.alpamayo2.zero3_module_regions import native_init_wrapper
    engine = SimpleNamespace(optimizer=SimpleNamespace(_alpamayo_gradient_reduce_group=object()))
    with pytest.raises(RuntimeError, match="Turbo-managed native extension"):
        native_init_wrapper(lambda *a: object(), {})(engine, object())


def test_reentrant_checkpoint_is_rejected_and_nonreentrant_rng_policy_preserved():
    from turbo_physai.optimizations.models.alpamayo2.zero3_module_regions import validate_checkpointing
    from torch.utils.checkpoint import checkpoint
    block = torch.nn.Linear(4, 4)
    block.gradient_checkpointing = True
    block._gradient_checkpointing_func = partial(checkpoint, use_reentrant=True)
    with pytest.raises(ValueError, match="use_reentrant=False"):
        validate_checkpointing([block])
    block._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
    original = block._gradient_checkpointing_func
    validate_checkpointing([block])
    assert block._gradient_checkpointing_func is original
    block._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False, preserve_rng_state=False)
    with pytest.raises(ValueError, match="preserve_rng_state"):
        validate_checkpointing([block])


def test_native_backend_cannot_be_replaced_with_vanilla_inductor():
    from turbo_physai.optimizations.models.alpamayo2.zero3_module_regions import root_compile_wrapper
    called = []
    wrapper = root_compile_wrapper(lambda *a, **kw: called.append(kw), {})
    with pytest.raises(RuntimeError, match="DeepCompile"):
        wrapper(object(), backend="inductor", fullgraph=True)
    assert not called


def test_disable_environment_leaves_config_and_prepare_unchanged(monkeypatch):
    from turbo_physai.optimizations.models.alpamayo2.zero3_module_regions import (
        deepspeed_config_wrapper, accelerator_prepare_wrapper)
    monkeypatch.setenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "1")
    config = {"zero_optimization": {"stage": 3}}
    wrapper = deepspeed_config_wrapper(lambda self, c: c, {})
    assert wrapper(object(), config) is config
    obj = object()
    assert accelerator_prepare_wrapper(lambda self, *a: a[0], {})(obj, obj) is obj


def test_embedding_backward_needs_reduction_but_does_not_read_weights():
    from turbo_physai.optimizations.models.alpamayo2.zero3_module_regions import CompileState
    model = torch.nn.Linear(2, 2)
    for i, p in enumerate(model.parameters()):
        p.ds_id = i
    engine = SimpleNamespace(module=model, global_rank=0)
    state = CompileState(engine, {})
    state.graph_labels[7] = "embedding"
    gm = SimpleNamespace(graph=SimpleNamespace(nodes=[
        SimpleNamespace(op="call_function", target="dc.reduce_grad.default")]))
    state.audit_pass(gm, 7, None, None, None, None, None, True)
    assert state.pass_count["embedding", "backward"] == 1


def test_zero3_full_shape_guard_keeps_dynamic_inputs_in_one_graph():
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent('''
        import torch
        from deepspeed.compile.patch_fake_tensor import patch_fake_tensor
        from deepspeed.compile.init_z3 import _allow_dynamo_dynamic_parameter_shapes_for_z3
        from turbo_physai.optimizations.models.alpamayo2.zero3_module_regions import static_parameter_policy_wrapper
        from turbo_physai.optimizations.models.alpamayo2 import deepspeed_compat as compat
        from torch._dynamo.guards import GuardBuilder
        from torch._subclasses.fake_tensor import FakeTensorMode
        torch._dynamo.variables.builder.wrap_to_fake_tensor_and_record = compat.fake_record_wrapper(torch._dynamo.variables.builder.wrap_to_fake_tensor_and_record,{})
        GuardBuilder.TENSOR_MATCH = compat.tensor_match_wrapper(GuardBuilder.TENSOR_MATCH,{})
        FakeTensorMode.from_tensor = compat.fake_from_tensor_wrapper(FakeTensorMode.from_tensor,{})
        compat.activate_fake_compat_wrapper(patch_fake_tensor,{})()
        config = torch._dynamo.config
        before = (config.force_parameter_static_shapes, config.force_nn_module_property_static_shapes)
        release = static_parameter_policy_wrapper(_allow_dynamo_dynamic_parameter_shapes_for_z3, {})(dict(dynamic=True))
        model = torch.nn.Embedding(16, 4)
        p = model.weight
        full = p.detach().clone()
        p.ds_id = 0
        p.ds_shape = p.shape
        p.ds_persist = False
        p.data = torch.empty(0)
        graphs = []
        def backend(gm, inputs):
            graphs.append(gm)
            def execute(*args):
                p.data = full
                try:
                    return gm.forward(*args)
                finally:
                    p.data = torch.empty(0)
            return execute
        compiled = torch.compile(torch.nn.Embedding.forward, backend=backend, fullgraph=True, dynamic=True)
        for size in (3, 5):
            indices = torch.arange(size)[None]
            actual = compiled(model, indices)
            torch.testing.assert_close(actual, torch.nn.functional.embedding(indices, full))
        assert len(graphs) == 1, f'capture_count={len(graphs)}'
        release()
        assert before == (config.force_parameter_static_shapes, config.force_nn_module_property_static_shapes), 'config not restored'
        print('PASS dynamic_input_static_zero_parameter')
    ''')
    import os
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            env={**os.environ, "TORCH_LOGS": "recompiles"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS dynamic_input_static_zero_parameter" in result.stdout
