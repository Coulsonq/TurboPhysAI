# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
from turbo_physai.optimizations.models.alpamayo2 import module_compile


def test_strict_module_compile_preserves_call_and_propagates_errors(monkeypatch):
    seen = {}
    def fake_compile(fn, **kwargs):
        seen.update(kwargs)
        return fn
    monkeypatch.delenv('TURBO_PHYSAI_DISABLE_TORCH_COMPILE', raising=False)
    monkeypatch.setattr(torch, 'compile', fake_compile)
    def original(x):
        assert torch._dynamo.config.suppress_errors is False
        assert torch._dynamo.config.fail_on_recompile_limit_hit is True
        assert torch._dynamo.config.recompile_limit == 64
        return x + 1
    wrapped = module_compile.compile_module_wrapper(original, {})
    assert wrapped(2) == 3
    assert seen == {'backend': 'inductor', 'fullgraph': True, 'dynamic': True}
    def error(x):
        raise ValueError('compile failure')
    failing = module_compile.compile_module_wrapper(error, {})
    with pytest.raises(ValueError, match='compile failure'):
        failing(0)


def test_explicit_compile_debug_switch(monkeypatch):
    monkeypatch.setenv('TURBO_PHYSAI_DISABLE_TORCH_COMPILE', '1')
    original = lambda x: x + 1
    assert module_compile.compile_module_wrapper(original, {}) is original
