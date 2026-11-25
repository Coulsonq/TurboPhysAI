# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace
import pytest
import torch
from turbo_physai.optimizations.models.alpamayo2 import channels_last


def test_mark_vision_projection_and_require_runtime_settings(monkeypatch):
    module = SimpleNamespace()
    def initialize(obj):
        obj.proj = torch.nn.Conv3d(3, 4, 2)
    wrapped = channels_last.patch_embed_init_wrapper(initialize, {})
    monkeypatch.delenv('MIOPEN_FIND_MODE', raising=False)
    with pytest.raises(RuntimeError, match='MIOPEN_FIND_MODE'):
        wrapped(module)
    monkeypatch.setenv('MIOPEN_FIND_MODE', '1')
    monkeypatch.setenv('PYTORCH_MIOPEN_SUGGEST_NHWC', '1')
    wrapped(module)
    assert module.proj._alpamayo_channels_last_3d is True


def test_only_marked_conv_receives_channels_last_and_gradients():
    x = torch.randn(5, 3, 2, 16, 16, requires_grad=True)
    conv = channels_last.conv3d_forward_wrapper(lambda obj, t: t, {})
    assert conv(SimpleNamespace(), x) is x
    assert conv(SimpleNamespace(), input=x) is x
    output = conv(SimpleNamespace(_alpamayo_channels_last_3d=True), x)
    assert output.is_contiguous(memory_format=torch.channels_last_3d)
    torch.testing.assert_close(output, x, rtol=0, atol=0)
    output.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)
