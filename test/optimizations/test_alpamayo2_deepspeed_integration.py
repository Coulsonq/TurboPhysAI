# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
from types import SimpleNamespace
import torch

def test_keyword_tensor_producer_is_scheduled_before_consumer():
    from turbo_physai.optimizations.models.alpamayo2.deepspeed_compat import node_requirements
    g=torch.fx.Graph();x=g.placeholder('x'); mask=g.call_function(torch.neg,(x,))
    y=g.call_function(torch.add,(x,),{'other':mask})
    assert node_requirements(y,[x]) == [mask,y]

def test_normal_input_preserves_symbolic_guard_metadata():
    from turbo_physai.optimizations.models.alpamayo2.deepspeed_compat import fake_record_wrapper
    from turbo_physai.optimizations.models.alpamayo2.deepspeed_compat import activate_fake_compat_wrapper
    activate_fake_compat_wrapper(lambda:None,{})()
    source=object(); metadata={'size':('symbolic_batch',4),'stride':(4,1)}
    tx=SimpleNamespace(output=SimpleNamespace(input_source_to_sizes_strides={source:metadata}))
    t=torch.ones(2,4)
    def original(t,tx,source):return t
    assert fake_record_wrapper(original,{})(t,tx=tx,source=source) is t
    assert tx.output.input_source_to_sizes_strides[source] is metadata

def test_compile_disable_keeps_original_loader(monkeypatch):
    from turbo_physai.optimizations.models.alpamayo2.native_deepspeed import native_load_wrapper
    monkeypatch.setenv('TURBO_PHYSAI_DISABLE_TORCH_COMPILE','1')
    value=object()
    assert native_load_wrapper(lambda builder,verbose=True:value,{})(object()) is value

def test_native_cache_does_not_modify_installed_source(tmp_path):
    import deepspeed
    from pathlib import Path
    from turbo_physai.optimizations.models.alpamayo2.native_deepspeed import prepare_sources
    source=Path(deepspeed.__file__).parent/'ops/csrc'
    originals={p.relative_to(source):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    prepared=prepare_sources(source,tmp_path/'native')
    assert 'set_reduce_process_group' in (prepared/'compile/deepcompile.cpp').read_text()
    assert originals == {p.relative_to(source):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    assert prepare_sources(source,tmp_path/'native') == prepared


def test_zero_parameter_guard_uses_released_shape():
    from turbo_physai.optimizations.models.alpamayo2.deepspeed_compat import fake_record_wrapper,activate_fake_compat_wrapper
    activate_fake_compat_wrapper(lambda:None,{})()
    t=torch.nn.Parameter(torch.empty(0));t.ds_id=1;t.ds_shape=(3,4)
    source=object();tx=SimpleNamespace(output=SimpleNamespace(input_source_to_sizes_strides={}))
    ret=fake_record_wrapper(lambda tensor,**kw:tensor,{}) (t,tx=tx,source=source)
    assert ret.shape == (3,4)
    assert tx.output.input_source_to_sizes_strides[source]['size'] == (0,)


def test_native_sources_are_prepared_without_patch_command(tmp_path, monkeypatch):
    import subprocess
    import deepspeed
    from pathlib import Path
    from turbo_physai.optimizations.models.alpamayo2.native_deepspeed import prepare_sources
    original = subprocess.run
    def reject_patch(args, *a, **kw):
        if args[0] == 'patch':
            raise AssertionError('native optimization must build Turbo C++ sources without patch')
        return original(args, *a, **kw)
    monkeypatch.setattr(subprocess, 'run', reject_patch)
    source = Path(deepspeed.__file__).parent/'ops/csrc'
    prepare_sources(source, tmp_path/'native')
