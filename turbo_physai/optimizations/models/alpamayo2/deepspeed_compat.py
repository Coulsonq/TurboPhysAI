# Copyright (c) Microsoft Corporation.
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Adapted from HYGON-AI/Deepspeed-das @ 0b8d6afe36f9d1e1c045449a3d16121aea4f69b3:
# deepspeed/compile/{patch_fake_tensor.py,list_schedule.py}.
"""Declared DeepCompile compatibility targets; no hidden global replacements."""
from functools import wraps
from importlib import import_module

_ACTIVE = False

def _active():
    from .zero3_module_regions import disabled
    return _ACTIVE and not disabled()

def activate_fake_compat_wrapper(original, options):
    @wraps(original)
    def wrapped():
        from .zero3_module_regions import disabled
        if disabled():
            return original()
        global _ACTIVE
        _ACTIVE = True
        # Torch targets are installed and restored by the Engine, not here.
    return wrapped

def fake_record_wrapper(original, options):
    @wraps(original)
    def wrapped(t, *args, **kwargs):
        if not _active():
            return original(t, *args, **kwargs)
        import torch
        ds = import_module('deepspeed.compile.patch_fake_tensor')
        dummy = ds.wrap_if_ds_param(t)
        result = original(dummy, *args, **kwargs)
        if tracing := torch._guards.TracingContext.try_get():
            tracing.tensor_to_context[t] = tracing.tensor_to_context.pop(dummy)
        source = kwargs.get('source')
        if source is not None and hasattr(t, 'ds_id'):
            tx = kwargs.get('tx') if 'tx' in kwargs else args[0]
            size, stride = ds._get_guard_sizes_strides(t)
            tx.output.input_source_to_sizes_strides[source] = {'size':size, 'stride':stride}
        return result
    return wrapped

def tensor_match_wrapper(original, options):
    @wraps(original)
    def wrapped(self, guard, value=None):
        if _active():
            ds = import_module('deepspeed.compile.patch_fake_tensor')
            param = ds._resolve_zero3_guarded_value(self, guard, value)
            if hasattr(param, 'ds_id') and hasattr(param, 'ds_shape'):
                expected = ds._zero3_parameter_guard_metadata(param)
                def matches(p):
                    return ds._matches_zero3_parameter_metadata(p, expected)
                code = f'DeepCompile ZeRO-3 semantic parameter ds_id={expected[-2]} shape={expected[-1]}'
                manager = self.get_guard_manager(guard)
                verbose = ds.get_verbose_code_parts(code, guard)
                try:
                    manager.add_lambda_guard(matches, verbose, guard.user_stack)
                except TypeError:
                    manager.add_lambda_guard(matches, verbose)
                return
        return original(self, guard, value)
    return wrapped

def fake_from_tensor_wrapper(original, options):
    @wraps(original)
    def wrapped(self, t, *args, **kwargs):
        if not _active():
            return original(self,t,*args,**kwargs)
        ds = import_module('deepspeed.compile.patch_fake_tensor')
        with ds.unset_fake_temporarily():
            return original(self,ds.wrap_if_ds_param(t),*args,**kwargs)
    return wrapped

def node_requirements(target_node, scheduled):
    from torch.fx.node import map_arg
    scheduled,visited,result=set(scheduled),set(),[]
    def visit(node):
        if node in scheduled or node in visited:
            return
        visited.add(node)
        args=[]
        map_arg((node.args,node.kwargs),args.append)
        for arg in args:
            visit(arg)
        result.append(node)
    visit(target_node)
    return result

def node_requirements_wrapper(original, options):
    @wraps(original)
    def wrapped(target_node, scheduled):
        from .zero3_module_regions import disabled
        if disabled():
            return original(target_node, scheduled)
        return node_requirements(target_node,scheduled)
    return wrapped
