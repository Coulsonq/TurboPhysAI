import os
from pathlib import Path
import tempfile
import unittest

import torch

from lerobot.optim.optimizers import AdamWConfig, load_optimizer_state, save_optimizer_state
from turbo_physai.optimizations.models.lerobot_act.adamw import wrap_apex_fused


@unittest.skipUnless(torch.cuda.is_available(), "HCU is required")
class ApexCheckpointTests(unittest.TestCase):
    def make_parameters(self):
        return [
            torch.nn.Parameter(torch.linspace(-0.2, 0.2, 32, device="cuda")),
            torch.nn.Parameter(torch.linspace(0.3, 0.7, 16, device="cuda")),
        ]

    def make_optimizer(self, params):
        config = AdamWConfig(lr=1e-3, weight_decay=1e-4)
        groups = [{"params": [params[0]]}, {"params": [params[1]], "lr": 2e-3}]
        return wrap_apex_fused(AdamWConfig.build, {})(config, groups)

    def step(self, optimizer, params, index):
        for i, param in enumerate(params):
            param.grad = torch.full_like(param, (index + 1) * (i + 1) * 0.013)
        optimizer.step()
        optimizer.zero_grad()

    def test_official_optimizer_save_and_resume(self):
        params_a = self.make_parameters()
        optimizer_a = self.make_optimizer(params_a)
        for step in range(3):
            self.step(optimizer_a, params_a, step)

        folder = Path(tempfile.mkdtemp(prefix="apex-resume-", dir=os.environ.get("ACT_ADAMW_TEST_DIR")))
        save_optimizer_state(optimizer_a, folder)
        params_b = [torch.nn.Parameter(p.detach().clone()) for p in params_a]
        optimizer_b = self.make_optimizer(params_b)
        try:
            load_optimizer_state(optimizer_b, folder)
        except Exception as exc:
            self.fail(f"Official optimizer checkpoint failed to resume: {type(exc).__name__}: {exc}")

        for step in range(3, 6):
            self.step(optimizer_a, params_a, step)
            self.step(optimizer_b, params_b, step)
        for a, b in zip(params_a, params_b):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            for key in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(optimizer_a.state[a][key], optimizer_b.state[b][key], rtol=0, atol=0)

    def test_missing_gradient_is_rejected_before_any_update(self):
        params = self.make_parameters()
        optimizer = self.make_optimizer(params)
        before = [p.detach().clone() for p in params]
        params[0].grad = torch.ones_like(params[0])
        params[1].grad = None
        with self.assertRaisesRegex(RuntimeError, "missing gradient"):
            optimizer.step()
        for p, original in zip(params, before):
            torch.testing.assert_close(p, original, rtol=0, atol=0)

    def test_float32_updates_match_original_adamw(self):
        params_a = self.make_parameters()
        params_b = [torch.nn.Parameter(p.detach().clone()) for p in params_a]
        config = AdamWConfig(lr=1e-3, weight_decay=1e-4)
        baseline = config.build([{"params": [params_a[0]]}, {"params": [params_a[1]], "lr": 2e-3}])
        optimizer = self.make_optimizer(params_b)
        for step in range(20):
            self.step(baseline, params_a, step)
            self.step(optimizer, params_b, step)
        for a, b in zip(params_a, params_b):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
            for key in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(baseline.state[a][key], optimizer.state[b][key], rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
