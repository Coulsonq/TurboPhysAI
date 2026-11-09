import importlib
import unittest
from unittest.mock import patch


try:
    regions = importlib.import_module(
        "turbo_physai.optimizations.models.lerobot_act.compile_regions"
    )
except ModuleNotFoundError as error:
    if not error.name.endswith("compile_regions"):
        raise
    regions = None


class Model:
    def __init__(self, vae=False, training=True):
        self.is_vae_encoder = vae
        self.training = training


def forward(model, value, *, offset=0):
    return value + offset


class CompileRegionsTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(regions, "compile_regions is not implemented")

    def test_independent_encoder_switches(self):
        for main, vae in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(main=main, vae=vae), patch.object(
                regions.torch, "compile", side_effect=lambda fn, **kw: fn
            ) as compile_call:
                wrapped = regions.wrap_encoder(forward, {"main_encoder": main, "vae_encoder": vae})
                self.assertEqual(wrapped(Model(False), 2, offset=3), 5)
                self.assertEqual(wrapped(Model(True), 3, offset=4), 7)
                self.assertEqual(compile_call.call_count, int(main or vae))

    def test_eval_and_nested_compile_remain_eager(self):
        with patch.object(regions.torch, "compile") as compile_call:
            wrapped = regions.wrap_encoder(forward, {"main_encoder": True})
            self.assertEqual(wrapped(Model(training=False), 2), 2)
            with patch.object(regions.torch.compiler, "is_compiling", return_value=True):
                self.assertEqual(wrapped(Model(), 2), 2)
            compile_call.assert_not_called()

    def test_decoder_is_independent_and_cached(self):
        with patch.object(regions.torch, "compile", side_effect=lambda fn, **kw: fn) as compile_call:
            wrapped = regions.wrap_decoder(forward, {})
            self.assertEqual(wrapped(Model(), 1, offset=2), 3)
            self.assertEqual(wrapped(Model(), 2), 2)
            compile_call.assert_called_once()

    def test_reject_non_boolean_encoder_options(self):
        with self.assertRaises((TypeError, ValueError)):
            regions.wrap_encoder(forward, {"main_encoder": "false"})


if __name__ == "__main__":
    unittest.main()
