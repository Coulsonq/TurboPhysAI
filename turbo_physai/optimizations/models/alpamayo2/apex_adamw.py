# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental Apex AdamW with correct DeepSpeed ZeRO-3 subgroup steps."""

from functools import wraps


def optimizer_wrapper(original, options):
    from apex.optimizers import FusedAdam
    from transformers.training_args import OptimizerNames

    @wraps(original)
    def wrapped(trainer, *args, **kwargs):
        trainer.args.optim = OptimizerNames.ADAMW_APEX_FUSED
        optimizer = original(trainer, *args, **kwargs)
        if not isinstance(optimizer, FusedAdam):
            raise RuntimeError('Apex optimizer group requires apex.optimizers.FusedAdam')
        return optimizer

    return wrapped


def zero3_step_wrapper(original, options):
    from apex.optimizers import FusedAdam

    @wraps(original)
    def wrapped(zero, sub_group_id):
        optimizer = zero.optimizer
        if not isinstance(optimizer, FusedAdam):
            return original(zero, sub_group_id)
        if (getattr(zero, 'offload_optimizer', False)
                or getattr(zero, 'swap_optimizer', False)
                or getattr(zero, 'torch_autocast_gradscaler', None) is not None
                or getattr(zero, 'zenflow', False)
                or optimizer.capturable or optimizer.master_weights):
            raise RuntimeError('Apex ZeRO-3 subgroup steps require HCU optimizer state, without swapping or grad scaling')
        parameter = zero.fp32_partitioned_groups_flat[sub_group_id]
        if parameter.grad is None:
            raise RuntimeError('Apex ZeRO-3 subgroup has no gradient')
        state = optimizer.state.get(parameter, {})
        if state and 'step' not in state:
            raise RuntimeError('Existing Apex optimizer state lacks the ZeRO-3 subgroup step')
        group = optimizer.param_groups[zero.sub_group_to_group_id[sub_group_id]]
        group['step'] = state.get('step', 0)
        result = original(zero, sub_group_id)
        optimizer.state[parameter]['step'] = group['step']
        return result

    return wrapped
