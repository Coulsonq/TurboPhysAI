# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import pytest
import torch

from turbo_physai.optimizations.models.alpamayo2.zero3_overlap import (
    gradient_reduce_wrapper,
    optimizer_destroy_wrapper,
    optimizer_init_wrapper,
)


@pytest.fixture
def distributed(monkeypatch):
    group = object()
    created, destroyed = [], []
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda *a, **k: 8)
    monkeypatch.setattr(torch.distributed, 'get_process_group_ranks', lambda _: list(range(8)))
    monkeypatch.setattr(torch.distributed, 'get_backend', lambda _: 'nccl')
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 8)

    def new_group(*, ranks, backend):
        created.append((ranks, backend))
        return group

    monkeypatch.setattr(torch.distributed, 'new_group', new_group)
    monkeypatch.setattr(torch.distributed, 'destroy_process_group', destroyed.append)
    return SimpleNamespace(group=group, created=created, destroyed=destroyed)


def initialize(instance, **overrides):
    instance.dp_process_group = object()
    instance.reduce_scatter = True
    instance.sequence_parallel_size = 1
    instance.all2all_process_group = None
    instance.__dict__.update(overrides)


def test_constructor_keeps_original_arguments_and_creates_equal_rank_group(distributed):
    seen = []

    def original(instance, marker):
        seen.append(marker)
        initialize(instance)

    wrapped = optimizer_init_wrapper(original, {})
    instance = SimpleNamespace()
    wrapped(instance, 'original argument')
    assert seen == ['original argument']
    assert instance._alpamayo_gradient_reduce_group is distributed.group
    assert distributed.created == [(list(range(8)), 'nccl')]
    assert wrapped.__wrapped__ is original


@pytest.mark.parametrize('overrides', [
    {'reduce_scatter': False},
    {'sequence_parallel_size': 2},
    {'all2all_process_group': object()},
])
def test_unsupported_parallel_modes_reject_before_group_creation(distributed, overrides):
    wrapped = optimizer_init_wrapper(lambda instance: initialize(instance, **overrides), {})
    with pytest.raises(ValueError, match='ZeRO-3 overlap'):
        wrapped(SimpleNamespace())
    assert distributed.created == []


def test_reordered_or_partial_dp_group_is_rejected(distributed, monkeypatch):
    monkeypatch.setattr(torch.distributed, 'get_process_group_ranks', lambda _: [1, 0, 2, 3, 4, 5, 6, 7])
    wrapped = optimizer_init_wrapper(initialize, {})
    with pytest.raises(ValueError, match='WORLD'):
        wrapped(SimpleNamespace())
    assert distributed.created == []


@pytest.mark.parametrize('predivide,averaging_world_size', [(1.0, 8), (2.0, 8), (2.0, 4)])
def test_reduction_uses_instance_group_and_keeps_dtype_scaling(
        distributed, monkeypatch, predivide, averaging_world_size):
    import deepspeed.runtime.comm.coalesced_collectives as collectives

    dp_group = object()
    parameter = torch.nn.Parameter(torch.zeros(8))
    parameter.grad = torch.arange(8, 16, dtype=torch.float32)
    seen = []

    def reduce(tensors, group):
        seen.append((group, tensors[0].dtype))
        return [tensors[0][:1].clone() / 8]

    monkeypatch.setattr(collectives, 'reduce_scatter_coalesced', reduce)
    instance = SimpleNamespace(
        dp_process_group=dp_group, _alpamayo_gradient_reduce_group=distributed.group,
        dtype=torch.float32, gradient_accumulation_dtype=torch.float32,
        postscale_gradients=True, gradient_predivide_factor=predivide,
        _get_param_partition_group=lambda _: dp_group,
        _assert_same_partition_group=lambda _: None,
        _gradient_averaging_world_size=lambda *a: averaging_world_size,
    )
    wrapped = gradient_reduce_wrapper(lambda *_: pytest.fail('original reduction called'), {})
    result = wrapped(instance, [parameter], torch.bfloat16)
    assert seen == [(distributed.group, torch.bfloat16)]
    assert result[0].dtype == torch.float32
    torch.testing.assert_close(result[0], torch.tensor([8.0 / averaging_world_size]))
    torch.testing.assert_close(parameter.grad, torch.arange(8, 16, dtype=torch.float32))


def test_unregistered_parameter_group_is_rejected(distributed):
    instance = SimpleNamespace(
        dp_process_group=object(), _alpamayo_gradient_reduce_group=distributed.group,
        _get_param_partition_group=lambda _: object(),
        _assert_same_partition_group=lambda _: None,
    )
    with pytest.raises(ValueError, match='parameter group'):
        gradient_reduce_wrapper(lambda *_: None, {})(instance, [object()], torch.float32)


def test_destroy_releases_group_after_original_cleanup(distributed):
    instance = SimpleNamespace(_alpamayo_gradient_reduce_group=distributed.group)

    def original(optimizer):
        assert distributed.destroyed == []
        assert optimizer._alpamayo_gradient_reduce_group is distributed.group
        return 'destroyed'

    assert optimizer_destroy_wrapper(original, {})(instance) == 'destroyed'
    assert distributed.destroyed == [distributed.group]
    assert instance._alpamayo_gradient_reduce_group is None


def test_packaged_configs_enable_overlap_and_declare_all_three_targets():
    from importlib import resources
    import yaml
    from turbo_physai.optimizations.models.alpamayo2 import catalog

    config_dir = resources.files('turbo_physai.optimizations.models.alpamayo2').joinpath('configs')
    for name in ('recipe.yaml', 'optimization.yaml'):
        config = yaml.safe_load(config_dir.joinpath(name).read_text(encoding='utf-8'))
        entry = next(item for item in config['optimization_groups']
                     if item['id'] == 'alpamayo2.zero3_comm_overlap')
        assert entry['enabled'] is True
    assert catalog.ZERO3_COMM_OVERLAP.group_id == 'alpamayo2.zero3_comm_overlap'
    assert len(catalog.ZERO3_COMM_OVERLAP.replacements) == 3
