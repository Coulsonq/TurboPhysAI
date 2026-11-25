import copy
import json
from types import SimpleNamespace

import torch

from turbo_physai.optimizations.models.alpamayo2 import dataloader


def test_workers_are_temporary_and_thread_count_is_applied(monkeypatch):
    trainer = SimpleNamespace(args=SimpleNamespace(dataloader_num_workers=4))
    loader = dataloader.train_dataloader_wrapper(lambda obj: obj.args.dataloader_num_workers, {})
    assert loader(trainer) == 2
    assert trainer.args.dataloader_num_workers == 4
    old_threads = torch.get_num_threads()
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    try:
        thread_setup = dataloader.omp_threads_wrapper(lambda obj: torch.get_num_threads(), {})
        assert thread_setup(trainer) == 4
        assert __import__("os").environ["OMP_NUM_THREADS"] == "4"
    finally:
        torch.set_num_threads(old_threads)


def test_disable_cpu_offload_preserves_other_settings_and_original(tmp_path):
    from turbo_physai.optimizations.models.alpamayo2 import cpu_offload

    original = {
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "offload_param": {"device": "cpu"},
        },
        "bf16": {"enabled": True},
    }
    expected_original = copy.deepcopy(original)
    config_path = tmp_path / "zero.json"
    config_path.write_text(json.dumps(original))
    seen = []
    def init(_self, config):
        seen.append(config)
    wrapped = cpu_offload.disable_cpu_offload_wrapper(init, {})
    wrapped(object(), str(config_path))
    wrapped(object(), original)
    for config in seen:
        assert config["zero_optimization"]["offload_optimizer"]["device"] == "none"
        assert config["zero_optimization"]["offload_param"]["device"] == "none"
        assert config["zero_optimization"]["stage"] == 3
        assert config["bf16"]["enabled"] is True
    assert original == expected_original
    assert json.loads(config_path.read_text()) == expected_original
