# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Build the small DeepCompile extension in an isolated published-wheel copy.

Run inside the training container after sourcing its DTK environment. The input
directory contains deepspeed/ and its dist-info; output must not already exist.
"""

import argparse
import importlib.machinery
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.package.resolve(), args.output.resolve()
    if not (source / "deepspeed/ops/csrc/compile/deepcompile.cpp").is_file():
        parser.error("--package must be the extracted supported Hygon DeepSpeed wheel")
    if output.exists():
        parser.error("--output must be a new directory")
    if source == output or source in output.parents:
        parser.error("output must be outside the input package")
    os.umask(0o077)
    shutil.copytree(source, output)
    for name in ("native_reduce_group.patch", "zero3_input_guards.patch", "zero3_scheduler_kwargs.patch"):
        patch = Path(__file__).with_name(name)
        subprocess.run(["patch", "--batch", "--forward", "-d", str(output), "-p1", "-i", str(patch)], check=True)
    sys.path.insert(0, str(output))
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(output.parent / "torch_extensions"))
    os.environ.setdefault("MAX_JOBS", "2")
    from deepspeed.ops.op_builder.dc import DeepCompileBuilder

    builder = DeepCompileBuilder()
    # Match the installed extension's PyInit symbol; .load() otherwise uses the
    # prebuilt unmodified wheel, while the default JIT module name is just dc.
    builder.name = "dc_op"
    module = builder.jit_load(verbose=True)
    if not callable(getattr(module, "set_reduce_process_group", None)):
        raise RuntimeError("the built extension lacks the gradient communicator interface")
    target = output / "deepspeed/ops" / ("dc_op" + importlib.machinery.EXTENSION_SUFFIXES[0])
    shutil.copy2(module.__file__, target)
    print(target)


if __name__ == "__main__":
    main()
