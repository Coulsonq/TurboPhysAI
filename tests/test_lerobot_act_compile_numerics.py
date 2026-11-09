import unittest
from unittest.mock import patch

import torch

from turbo_physai.optimizations.models.lerobot_act import compile_regions


class SmallModule(torch.nn.Module):
    def __init__(self, vae=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.25, -0.75]))
        self.is_vae_encoder = vae


def forward(module, x, *, scale=1):
    return (x * module.weight * scale).square().sum()


class CompileNumericsTest(unittest.TestCase):
    def test_cpu_eager_backend_forward_and_backward(self):
        real_compile = torch.compile
        for kind in ("main", "vae", "decoder"):
            with self.subTest(kind=kind):
                model = SmallModule(vae=kind == "vae")
                x = torch.tensor([0.5, 2.0], requires_grad=True)
                eager = forward(model, x, scale=3)
                eager.backward()
                expected = (eager.detach().clone(), x.grad.clone(), model.weight.grad.clone())
                x.grad = None
                model.weight.grad = None
                with patch.object(compile_regions.torch, "compile", side_effect=lambda fn, **kw: real_compile(fn, backend="eager")):
                    wrapper = (compile_regions.wrap_decoder(forward, {}) if kind == "decoder" else
                               compile_regions.wrap_encoder(forward, {"main_encoder": True, "vae_encoder": True}))
                    result = wrapper(model, x, scale=3)
                    result.backward()
                for actual, reference in zip((result.detach(), x.grad, model.weight.grad), expected):
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
