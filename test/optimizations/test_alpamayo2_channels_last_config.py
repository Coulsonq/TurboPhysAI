# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from importlib import resources

import yaml

from turbo_physai.runtime import load_runtime_config, prepare_environment


def test_default_channels_last_group_and_launch_environment(monkeypatch):
    config_dir = resources.files('turbo_physai.optimizations.models.alpamayo2').joinpath('configs')
    for name in ('recipe.yaml', 'optimization.yaml'):
        config = yaml.safe_load(config_dir.joinpath(name).read_text(encoding='utf-8'))
        group = next(item for item in config['optimization_groups']
                     if item['id'] == 'alpamayo2.channels_last_vision')
        assert group['enabled'] is True

    monkeypatch.setenv('MIOPEN_FIND_MODE', 'inherited')
    monkeypatch.setenv('PYTORCH_MIOPEN_SUGGEST_NHWC', 'inherited')
    runtime = load_runtime_config(str(config_dir.joinpath('runtime.yaml')))
    effective_env = prepare_environment(runtime)
    assert effective_env['MIOPEN_FIND_MODE'] == '1'
    assert effective_env['PYTORCH_MIOPEN_SUGGEST_NHWC'] == '1'
    assert effective_env['HIP_BUFFER_EXTRA_SIZE'] == '0'


def test_removed_experimental_groups_are_not_registered():
    from turbo_physai.optimizations.models.alpamayo2 import catalog

    removed = {'alpamayo2.apex_fused_adamw', 'alpamayo2.module_compile'}
    config_dir = resources.files('turbo_physai.optimizations.models.alpamayo2').joinpath('configs')
    for name in ('recipe.yaml', 'optimization.yaml'):
        config = yaml.safe_load(config_dir.joinpath(name).read_text(encoding='utf-8'))
        assert not removed.intersection(item['id'] for item in config['optimization_groups'])
    assert not hasattr(catalog, 'APEX_FUSED_ADAMW')
    assert not hasattr(catalog, 'MODULE_COMPILE')
