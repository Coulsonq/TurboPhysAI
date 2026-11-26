# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import pytest

from turbo_physai.optimizations.models.alpamayo2.bf16_communication import bf16_communication_wrapper


def test_bf16_override_precedes_accelerate_autoselection():
    seen = []
    plugin = SimpleNamespace(deepspeed_config={'communication_data_type': 'fp32'})

    def original(instance, mixed_precision):
        seen.append((mixed_precision, instance.deepspeed_config['communication_data_type']))
        return 'configured'

    wrapped = bf16_communication_wrapper(original, {})
    assert wrapped(plugin, 'bf16') == 'configured'
    assert seen == [('bf16', 'bf16')]
    assert plugin.deepspeed_config['communication_data_type'] == 'bf16'


def test_reject_non_bf16_without_mutating_config():
    plugin = SimpleNamespace(deepspeed_config={'communication_data_type': 'fp32'})
    wrapped = bf16_communication_wrapper(lambda *_: None, {})
    with pytest.raises(ValueError, match='bf16'):
        wrapped(plugin, 'fp16')
    assert plugin.deepspeed_config['communication_data_type'] == 'fp32'


@pytest.mark.parametrize('initial', [{}, {'communication_data_type': 'fp32'}])
def test_original_failure_restores_previous_communication_setting(initial):
    plugin = SimpleNamespace(deepspeed_config=initial.copy())

    def fail(*_):
        raise RuntimeError('configuration failed')

    wrapped = bf16_communication_wrapper(fail, {})
    with pytest.raises(RuntimeError, match='configuration failed'):
        wrapped(plugin, 'bf16')
    assert plugin.deepspeed_config == initial


def test_actual_accelerate_plugin_receives_bf16_override(monkeypatch):
    import accelerate.utils.dataclasses as dataclasses

    monkeypatch.setattr(dataclasses, 'is_rocm_available', lambda: True)
    config = {'zero_optimization': {'stage': 3}, 'bf16': {'enabled': 'auto'}}
    baseline = dataclasses.DeepSpeedPlugin(hf_ds_config=config.copy())
    assert baseline.deepspeed_config is baseline.hf_ds_config.config
    baseline.set_mixed_precision('bf16')
    assert baseline.deepspeed_config['communication_data_type'] == 'fp32'

    overridden = dataclasses.DeepSpeedPlugin(hf_ds_config={
        'zero_optimization': {'stage': 3}, 'bf16': {'enabled': 'auto'}
    })
    wrapped = bf16_communication_wrapper(dataclasses.DeepSpeedPlugin.set_mixed_precision, {})
    wrapped(overridden, 'bf16')
    assert overridden.hf_ds_config.config['communication_data_type'] == 'bf16'
