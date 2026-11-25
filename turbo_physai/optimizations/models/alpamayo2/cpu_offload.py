# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Disable CPU offloading in the DeepSpeed configuration for this rank."""

import base64
import copy
import json
import os
from functools import wraps


def disable_cpu_offload_wrapper(original, options):
    @wraps(original)
    def wrapped(instance, config_file_or_dict):
        if isinstance(config_file_or_dict, dict):
            config = copy.deepcopy(config_file_or_dict)
        elif isinstance(config_file_or_dict, (str, os.PathLike)):
            config_path = os.fspath(config_file_or_dict)
            if os.path.isfile(config_path):
                with open(config_path, encoding="utf-8") as stream:
                    config = json.load(stream)
            else:
                try:
                    config = json.loads(base64.urlsafe_b64decode(config_path).decode("utf-8"))
                except (ValueError, UnicodeDecodeError) as error:
                    raise ValueError(f"Invalid DeepSpeed configuration: {config_path}") from error
        else:
            raise TypeError("DeepSpeed configuration must be a dict, file path, or base64 JSON")
        zero_config = config.get("zero_optimization", {})
        for key in ("offload_optimizer", "offload_param"):
            offload = zero_config.get(key)
            if isinstance(offload, dict) and offload.get("device") == "cpu":
                offload["device"] = "none"
        return original(instance, config)

    return wrapped
