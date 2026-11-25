# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import pytest
import torch
from transformers.training_args import OptimizerNames

FusedAdam = pytest.importorskip("apex.optimizers").FusedAdam

from turbo_physai.optimizations.models.alpamayo2 import apex_adamw


def test_selects_apex_and_preserves_returned_optimizer():
    trainer = SimpleNamespace(args=SimpleNamespace(optim=OptimizerNames.ADAMW_TORCH_FUSED))
    optimizer = object.__new__(FusedAdam)
    wrapped = apex_adamw.optimizer_wrapper(lambda obj: optimizer, {})
    assert wrapped(trainer) is optimizer
    assert trainer.args.optim == OptimizerNames.ADAMW_APEX_FUSED


def test_rejects_when_trainer_returns_other_optimizer():
    trainer = SimpleNamespace(args=SimpleNamespace(optim=OptimizerNames.ADAMW_TORCH_FUSED))
    wrapped = apex_adamw.optimizer_wrapper(lambda obj: object(), {})
    with pytest.raises(RuntimeError, match='FusedAdam'):
        wrapped(trainer)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU/CUDA")
def test_zero3_subgroups_match_adamw_updates():
    pytest.importorskip("amp_C")
    deepspeed = pytest.importorskip("deepspeed.runtime.zero.stage3")
    torch.manual_seed(123)
    refs = [torch.nn.Parameter(torch.randn(257, device="cuda")) for _ in range(2)]
    actual = [torch.nn.Parameter(p.detach().clone()) for p in refs]
    reference = torch.optim.AdamW(refs, lr=5e-5, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    optimizer = FusedAdam(actual, lr=5e-5, betas=(0.9, 0.95), weight_decay=0.1)
    optimizer.param_groups[0]["params"] = []
    zero = SimpleNamespace(
        optimizer=optimizer,
        sub_group_to_group_id={0: 0, 1: 0},
        fp32_partitioned_groups_flat=actual,
        torch_autocast_gradscaler=None,
        offload_optimizer=False,
        zenflow=False,
    )
    step = apex_adamw.zero3_step_wrapper(deepspeed.DeepSpeedZeroOptimizer_Stage3._optimizer_step, {})
    for iteration in range(1, 6):
        for index, (ref, param) in enumerate(zip(refs, actual)):
            grad = torch.sin(torch.arange(257, device="cuda") * 0.017 + iteration * 0.31 + index * 0.71)
            ref.grad = grad.clone()
            param.grad = grad.clone()
        reference.step()
        step(zero, 0)
        step(zero, 1)
    for ref, param in zip(refs, actual):
        assert optimizer.state[param]["step"] == 5
        torch.testing.assert_close(param, ref, rtol=1e-5, atol=1e-6)


def test_zero3_cpu_offload_is_rejected():
    optimizer = object.__new__(FusedAdam)
    zero = SimpleNamespace(optimizer=optimizer, offload_optimizer=True)
    wrapped = apex_adamw.zero3_step_wrapper(lambda *_: None, {})
    with pytest.raises(RuntimeError, match="without swapping or grad scaling"):
        wrapped(zero, 0)
