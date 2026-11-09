import unittest
from unittest import mock

import torch
from lerobot.optim.optimizers import AdamWConfig
from turbo_physai.optimizations.models.lerobot_act import adamw


class AdamWTests(unittest.TestCase):
    def setUp(self):
        self.config = AdamWConfig(lr=1e-4, weight_decay=0.01, grad_clip_norm=1.0)

    def cuda_parameter(self, dtype=torch.float32):
        param = mock.Mock(spec=torch.Tensor)
        param.device = torch.device("cuda:0")
        param.dtype = dtype
        return param

    def test_apex_uses_adamw_mode_and_preserves_groups(self):
        p1, p2 = self.cuda_parameter(), self.cuda_parameter()
        groups = [{"params": [p1]}, {"params": [p2], "lr": 1e-5}]
        original = mock.Mock()
        with mock.patch("apex.optimizers.FusedAdam") as constructor:
            constructor.return_value.param_groups = [{"params": [p1]}, {"params": [p2]}]
            result = adamw.wrap_apex_fused(original, {})(self.config, iter(groups))
        self.assertIs(result, constructor.return_value)
        args, kwargs = constructor.call_args
        self.assertEqual(args[0], groups)
        self.assertTrue(kwargs.pop("adam_w_mode"))
        self.assertTrue(kwargs.pop("bias_correction"))
        self.assertEqual(kwargs, {"lr": 1e-4, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01})
        self.assertEqual([group["step"] for group in result.param_groups], [0, 0])
        result.register_step_pre_hook.assert_called_once_with(adamw._require_all_gradients)
        original.assert_not_called()

    def test_cpu_parameter_falls_back_without_losing_generator(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        original = mock.Mock()
        result = adamw.wrap_apex_fused(original, {})(self.config, (p for p in [parameter]))
        self.assertIs(result, original.return_value)
        self.assertEqual(original.call_args.args, (self.config, [parameter]))

    def test_explicit_foreach_falls_back(self):
        parameter = self.cuda_parameter()
        groups = [{"params": [parameter], "foreach": True}]
        original = mock.Mock()
        result = adamw.wrap_apex_fused(original, {})(self.config, groups)
        self.assertIs(result, original.return_value)
        self.assertEqual(original.call_args.args, (self.config, groups))

    def test_lower_precision_falls_back(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                parameter = self.cuda_parameter(dtype)
                original = mock.Mock()
                result = adamw.wrap_apex_fused(original, {})(self.config, [parameter])
                self.assertIs(result, original.return_value)
                self.assertEqual(original.call_args.args, (self.config, [parameter]))


if __name__ == "__main__":
    unittest.main()
