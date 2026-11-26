# Copyright (c) Microsoft Corporation.
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Modified by Hygon.

"""Separate ZeRO-3 gradient reduction from parameter-fetch collectives."""

from functools import wraps


def optimizer_init_wrapper(original, options):
    @wraps(original)
    def wrapped(optimizer, *args, **kwargs):
        import torch
        result = original(optimizer, *args, **kwargs)
        dist = torch.distributed
        if not dist.is_initialized():
            raise RuntimeError('ZeRO-3 overlap requires an initialized process group')
        if (not optimizer.reduce_scatter or optimizer.sequence_parallel_size != 1
                or optimizer.all2all_process_group is not None):
            raise ValueError('ZeRO-3 overlap requires standard data-parallel reduce-scatter')
        source = optimizer.dp_process_group or dist.group.WORLD
        ranks = list(range(dist.get_world_size()))
        if dist.get_process_group_ranks(source) != ranks:
            raise ValueError('ZeRO-3 overlap requires the full WORLD rank order')
        if dist.get_backend(source) != 'nccl' or torch.cuda.device_count() != len(ranks):
            raise ValueError('ZeRO-3 overlap requires single-node NCCL data parallelism')
        optimizer._alpamayo_gradient_reduce_group = dist.new_group(ranks=ranks, backend='nccl')
        return result

    return wrapped


def gradient_reduce_wrapper(original, options):
    @wraps(original)
    def wrapped(optimizer, params_to_reduce, communication_data_type):
        import torch
        from deepspeed.runtime.comm.coalesced_collectives import reduce_scatter_coalesced

        optimizer._assert_same_partition_group(params_to_reduce)
        source = optimizer._get_param_partition_group(params_to_reduce[0])
        if source not in (optimizer.dp_process_group, None, torch.distributed.group.WORLD):
            raise ValueError('ZeRO-3 overlap does not support this parameter group')
        group = getattr(optimizer, '_alpamayo_gradient_reduce_group', None)
        if group is None:
            raise RuntimeError('ZeRO-3 overlap gradient group was not initialized')

        # Retain stage3.__avg_scatter_grads scaling, casts and return contract.
        full_grads = [p.grad for p in params_to_reduce]
        if communication_data_type != optimizer.dtype:
            full_grads = [g.to(communication_data_type) for g in full_grads]
        if optimizer.postscale_gradients and optimizer.gradient_predivide_factor != 1.0:
            full_grads = [g.div(optimizer.gradient_predivide_factor) for g in full_grads]

        partitions = reduce_scatter_coalesced(full_grads, group)
        partition_world_size = torch.distributed.get_world_size(group)
        if (optimizer.postscale_gradients and optimizer.gradient_predivide_factor != 1.0
                and optimizer.gradient_predivide_factor != partition_world_size):
            partitions = [g.mul(optimizer.gradient_predivide_factor) for g in partitions]
        averaging_world_size = optimizer._gradient_averaging_world_size(params_to_reduce, partition_world_size)
        if averaging_world_size != partition_world_size:
            scale = partition_world_size / float(averaging_world_size)
            partitions = [g.mul(scale) for g in partitions]
        if communication_data_type != optimizer.gradient_accumulation_dtype:
            partitions = [g.to(optimizer.gradient_accumulation_dtype) for g in partitions]
        return partitions

    return wrapped


def optimizer_destroy_wrapper(original, options):
    @wraps(original)
    def wrapped(optimizer, *args, **kwargs):
        import torch
        result = original(optimizer, *args, **kwargs)
        group = getattr(optimizer, '_alpamayo_gradient_reduce_group', None)
        if group is not None and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group(group)
        optimizer._alpamayo_gradient_reduce_group = None
        return result

    return wrapped
