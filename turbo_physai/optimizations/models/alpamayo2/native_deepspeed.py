# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
"""Build DeepCompile's communicator bridge from Turbo-owned resources on first use.

Only the private cache is written. The installed DeepSpeed package stays intact.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from functools import wraps

_LOADED = {}

NATIVE_BASELINES = {'compile/deepcompile.cpp': '13b535e5c55db077cdc917e5bf51f81b2b405ea8da68c37210de99984982af1b', 'compile/init.cpp': '50ce992a7b2d3994a557aa8f29340765cd0ba2cfbaccc64a8f63dc7ed8b359bf', 'compile/z3.cpp': '8ff77bba1691ae76249213f6e016083751ca7752c22b65307082b9e4bea141da', 'includes/deepcompile.h': '57c9200c73fb2031aff66f92ca3aa39d5ea3af17aaddcabecb88d8262a5a6ac5'}

def native_sources_root():
    return Path(__file__).with_name('native')/'csrc'


def source_fingerprint(source):
    import torch
    import deepspeed
    h=hashlib.sha256()
    h.update(repr((torch.__version__,torch.version.hip,deepspeed.__version__,sys.implementation.cache_tag,
                   torch._C._GLIBCXX_USE_CXX11_ABI,os.getenv('CXX'))).encode())
    h.update(json.dumps(NATIVE_BASELINES,sort_keys=True).encode())
    for relative in sorted(NATIVE_BASELINES):
        h.update(relative.encode())
        h.update((native_sources_root()/relative).read_bytes())
    for folder in ('compile','includes'):
        for p in sorted((source/folder).rglob('*')):
            if p.is_file():
                h.update(str(p.relative_to(source)).encode());h.update(p.read_bytes())
    return h.hexdigest()

def prepare_sources(source, root):
    source,root=Path(source),Path(root)
    for relative, expected in NATIVE_BASELINES.items():
        original = source/relative
        if not original.is_file() or hashlib.sha256(original.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Turbo native implementation requires the supported original DeepSpeed sources: '+relative)
    key=source_fingerprint(source)
    root.mkdir(parents=True,exist_ok=True,mode=0o700)
    with (root/'source.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        dest=root/'deepspeed/ops/csrc';marker=root/'source.json'
        if marker.exists():
            if json.loads(marker.read_text())['fingerprint'] != key:
                raise RuntimeError('native source cache fingerprint mismatch')
            return dest
        if dest.exists():
            raise RuntimeError(f'incomplete native source cache: {root}; use a new cache directory')
        for folder in ('compile','includes'):
            shutil.copytree(source/folder,dest/folder)
        for relative in NATIVE_BASELINES:
            shutil.copy2(native_sources_root()/relative,dest/relative)
        marker.write_text(json.dumps({'fingerprint':key}))
        return dest

def load_native(builder, options):
    import deepspeed
    source=Path(deepspeed.__file__).parent/'ops/csrc'
    key=source_fingerprint(source)
    if key in _LOADED:
        return _LOADED[key]
    base=Path(os.environ.get('XDG_CACHE_HOME',Path.home()/'.cache'))/'turbo_physai/alpamayo2/deepcompile'
    root=base/key
    prepared=prepare_sources(source,root)
    name = 'turbo_alpamayo2_dc_' + key[:16]
    binary = root/'extensions'/name/(name+'.so')
    with (root/'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if key in _LOADED:
            return _LOADED[key]
        if binary.is_file():
            spec = importlib.util.spec_from_file_location(name, binary)
            native = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(native)
            mode = 'cached'
        else:
            instance = copy.copy(builder)
            instance.name = name
            original_path = instance.deepspeed_src_path
            instance.deepspeed_src_path = lambda path: str(prepared/Path(path).relative_to('csrc')) if str(path).startswith('csrc/') else original_path(path)
            names = ('TORCH_EXTENSIONS_DIR', 'MAX_JOBS', 'PYTORCH_ROCM_ARCH')
            previous = {name: os.environ.get(name) for name in names}
            try:
                os.environ['TORCH_EXTENSIONS_DIR'] = str(root/'extensions')
                os.environ.setdefault('MAX_JOBS', '2')
                native = instance.jit_load(verbose=True)
            finally:
                for variable, value in previous.items():
                    if value is None:
                        os.environ.pop(variable, None)
                    else:
                        os.environ[variable] = value
            mode = 'built'
        if not callable(getattr(native, 'set_reduce_process_group', None)):
            raise RuntimeError('Turbo native extension lacks the independent gradient communicator')
        _LOADED[key] = native
    print(f'ALPA_NATIVE mode={mode} loaded={native.__file__} source=turbo_physai fingerprint={key}', flush=True)
    return native


def native_load_wrapper(original, options):
    @wraps(original)
    def wrapped(builder, verbose=True):
        from .zero3_module_regions import disabled
        if disabled():
            return original(builder,verbose=verbose)
        return load_native(builder,options)
    return wrapped
