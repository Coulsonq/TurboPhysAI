# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Select BF16 DeepSpeed gradient communication for BF16 training."""

from functools import wraps


def bf16_communication_wrapper(original, options):
    @wraps(original)
    def wrapped(plugin, mixed_precision):
        if mixed_precision != "bf16":
            raise ValueError("Alpamayo2 BF16 communication requires bf16 mixed precision")
        config = plugin.deepspeed_config
        if not isinstance(config, dict):
            raise TypeError("DeepSpeed communication configuration must be a dictionary")
        missing = object()
        previous = config.get("communication_data_type", missing)
        config["communication_data_type"] = "bf16"
        try:
            return original(plugin, mixed_precision)
        except BaseException:
            if previous is missing:
                config.pop("communication_data_type", None)
            else:
                config["communication_data_type"] = previous
            raise

    return wrapped
