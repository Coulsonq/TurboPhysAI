import unittest
from unittest.mock import patch

from turbo_physai.optimizations.models.lerobot_act import compile_regions


class Layer:
    training = True


class FfnCompileTest(unittest.TestCase):
    def test_lazily_compile_the_ffn_and_skip_nested_compilation(self):
        self.assertTrue(hasattr(compile_regions, "wrap_ffn"), "FFN wrapper missing")
        original = lambda layer, x: x + 1
        with patch.object(compile_regions.torch, "compile", side_effect=lambda fn, **kw: fn) as compile_call:
            wrapped = compile_regions.wrap_ffn(original, {})
            with patch.object(compile_regions.torch.compiler, "is_compiling", return_value=True):
                self.assertEqual(wrapped(Layer(), 2), 3)
            compile_call.assert_not_called()
            self.assertEqual(wrapped(Layer(), 3), 4)
            self.assertEqual(wrapped(Layer(), 4), 5)
            compile_call.assert_called_once()


if __name__ == "__main__":
    unittest.main()
