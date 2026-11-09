import unittest

from turbo_physai.engine.execution.replacements.wrapper import WrapperHandler
from turbo_physai.optimizations.models.lerobot_act import compile_regions_catalog as catalog


class EngineRestoreTest(unittest.TestCase):
    def test_targets_are_restored_by_engine_handler(self):
        handler = WrapperHandler()
        for group, options in (
            (catalog.ENCODER_REGIONS, {"main_encoder": True, "vae_encoder": True}),
            (catalog.DECODER_REGION, {}),
            (catalog.FFN_REGION, {}),
        ):
            with self.subTest(group=group.group_id):
                prepared = handler.prepare(group.specs[0], options)
                target = prepared.targets[0]
                snapshot = handler.snapshot(prepared)
                handler.apply(prepared)
                self.assertIs(getattr(target.parent, target.attribute), prepared.replacement)
                results = handler.restore(snapshot)
                self.assertIs(getattr(target.parent, target.attribute), target.original)
                self.assertEqual(results[0].status.value, "restored")


if __name__ == "__main__":
    unittest.main()
