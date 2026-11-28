# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental, strict module-level DeepCompile integration for Alpamayo Stage 1.

Native ZeRO-3 owns gather/release/reduce; do not manually gather entire modules,
remove ZeRO hooks, disable checkpointing, or catch compiler failures here.
"""

import copy
import json
import os
import re
from collections import Counter
from functools import wraps


def disabled():
    return os.environ.get("TURBO_PHYSAI_DISABLE_TORCH_COMPILE") == "1"


def single_stage_reductions(source):
    """Change only Inductor's literal loop pipeline directive, preserving math."""
    return re.sub(r"(?m)^([ \t]*for \w+ in tl\.range\([^\n]*, num_stages = )2(\):)$",
                  r"\g<1>1\2", source)


def reduction_codegen_wrapper(original, options):
    @wraps(original)
    def wrapped(kernel, *args, **kwargs):
        source = original(kernel, *args, **kwargs)
        if disabled():
            return source
        import torch
        from torch._inductor.virtualized import V
        device = V.graph.get_current_device_or_throw()
        if torch.version.hip and device.type == "cuda":
            arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
            if arch == "gfx938":
                return single_stage_reductions(source)
        return source
    return wrapped


def validate_checkpointing(blocks):
    for block in blocks:
        fn = getattr(block, "_gradient_checkpointing_func", None)
        keywords = getattr(fn, "keywords", {}) or {}
        if not getattr(block, "gradient_checkpointing", False) or keywords.get("use_reentrant") is not False:
            raise ValueError("module compile requires checkpointing with use_reentrant=False in every block")
        if keywords.get("preserve_rng_state", True) is not True:
            raise ValueError("module compile requires preserve_rng_state=True")


def validate_engine(engine):
    if engine.zero_optimization_stage() != 3:
        raise ValueError("module compile requires ZeRO-3")
    if engine.zero_use_cpu_optimizer() or getattr(engine.zero_offload_param(), "device", "none") not in ("none", None):
        raise ValueError("this experimental compile path requires no optimizer/parameter CPU offload")
    if not engine.is_deepcompile_enabled():
        raise RuntimeError("native DeepCompile must be enabled; ordinary torch.compile is insufficient for ZeRO-3")
    if engine.is_compiled:
        raise ValueError("do not also enable Trainer/Accelerate torch_compile")
    model = engine.module
    if getattr(model, "training_stage", None) != "vlm":
        raise ValueError("module compile currently supports Alpamayo Stage-1 VLM training only")
    visual, text = model.vlm.model.visual, model.vlm.model.language_model
    if any(m.config._attn_implementation != "flash_attention_2" for m in (visual, text)):
        raise ValueError("module compile requires alpamayo2.flash_attention")
    validate_checkpointing([*visual.blocks, *text.layers])
    import torch
    if torch._dynamo.config.suppress_errors:
        raise ValueError("compiler error suppression is incompatible with strict module compile")


class CompileState:
    def __init__(self, engine, options):
        self.engine = engine
        self.options = options
        self.expected = {p.ds_id for p in engine.module.parameters() if p.requires_grad}
        self.captured = set()
        self.graph_labels = {}
        self.capture_count = Counter()
        self.executions = Counter()
        self.pass_count = Counter()
        self.coverage_checked = False

    def emit(self, event, **values):
        rank = self.engine.global_rank
        print("ALPA_COMPILE " + json.dumps({"rank": rank, "event": event, **values}, sort_keys=True), flush=True)

    def backend(self, name, native_backend):
        def audited(gm, inputs):
            self.graph_labels[id(gm.graph)] = name
            nodes = list(gm.graph.nodes)
            checkpoint_count = sum("tag_activation_checkpoint" in str(n.target) for n in nodes)
            if name in ("encoder", "qwen") and checkpoint_count == 0:
                raise RuntimeError(f"{name} did not capture its non-reentrant checkpoint blocks")
            params = {x.ds_id for x in inputs if getattr(x, "requires_grad", False) and hasattr(x, "ds_id")}
            if not params:
                raise RuntimeError(f"{name} captured no trainable ZeRO-3 parameters")
            import torch
            # Preserve explicit BF16 rounding points across fused pointwise ops.
            # Inductor captures this setting for its forward/backward compilers.
            with torch._inductor.config.patch(emulate_precision_casts=True):
                result = native_backend(gm, inputs)
            self.captured.update(params)
            self.capture_count[name] += 1
            self.emit("captured", region=name, nodes=len(nodes), checkpoint_blocks=checkpoint_count,
                      parameter_count=len(params), captures=self.capture_count[name], fullgraph=True)

            @wraps(result)
            def execute(*args, **kwargs):
                import torch
                self.executions[name] += 1
                with torch.profiler.record_function("ALPA_GRAPH/" + name):
                    return result(*args, **kwargs)
            return execute
        return audited

    def audit_pass(self, gm, graph_id, graph_order, profiling_results,
                   create_inputs_fn, mem_budget, param_manager, bwd):
        name = self.graph_labels.get(graph_id)
        if name is None:
            raise RuntimeError("native DeepCompile produced an unlabelled graph")
        operations = Counter(str(n.target) for n in gm.graph.nodes if n.op == "call_function")
        comm = {k: v for k, v in operations.items() if k.startswith("dc.")}
        # Embedding backward only needs token indices and incoming gradients;
        # it does not read the weight and legitimately has no backward gather.
        if not bwd and not any("allgather_param" in k for k in comm):
            raise RuntimeError(f"{name} has no native ZeRO-3 gather nodes")
        if bwd and not any("reduce_grad" in k for k in comm):
            raise RuntimeError(f"{name} backward has no native ZeRO-3 gradient reduction")
        phase = "backward" if bwd else "forward"
        self.pass_count[name, phase] += 1
        self.emit("native_graph", region=name, phase=phase, operations=len(list(gm.graph.nodes)), collectives=comm)
        # None deliberately avoids asking DeepCompile to run an extra profiling pass.

    def check_forward_coverage(self):
        missing = self.expected - self.captured
        if missing:
            names = [n for n, p in self.engine.module.named_parameters() if getattr(p, "ds_id", None) in missing]
            raise RuntimeError("trainable parameters bypassed compiled gradient reduction: " + ", ".join(names))
        if not all(self.executions[x] > 0 for x in ("encoder", "qwen", "embedding", "lm_head")):
            raise RuntimeError("one of the required compiled regions was not executed")
        if not self.coverage_checked:
            self.coverage_checked = True
            self.emit("all_trainable_parameters_covered", count=len(self.expected))


def deepspeed_config_wrapper(original, options):
    @wraps(original)
    def wrapped(obj, config, *args, **kwargs):
        if disabled():
            return original(obj, config, *args, **kwargs)
        if not isinstance(config, dict):
            raise ValueError("module compile expects the Trainer-provided DeepSpeed config dictionary")
        config = copy.deepcopy(config)
        compile_config = config.setdefault("compile", {})
        for key in ("offload_activation", "offload_parameters", "offload_opt_states", "free_activation", "symmetric_memory"):
            if compile_config.get(key):
                raise ValueError(f"compile.{key} is outside this candidate's training contract")
        if compile_config.get("passes") not in (None, ["z3"]):
            raise ValueError("module compile requires only the native z3 pass")
        compile_config.update(deepcompile=True, passes=["z3"])
        return original(obj, config, *args, **kwargs)
    return wrapped


def native_init_wrapper(original, options):
    @wraps(original)
    def wrapped(engine, compile_config):
        if disabled():
            return original(engine, compile_config)
        native = original(engine, compile_config)
        group = getattr(engine.optimizer, "_alpamayo_gradient_reduce_group", None)
        if group is not None:
            setter = getattr(native, "set_reduce_process_group", None)
            if setter is None:
                raise RuntimeError("DeepCompile with zero3_comm_overlap requires the Turbo-managed native extension")
            setter(group)
            if native.get_reduce_group_name() != group.group_name:
                raise RuntimeError("native gradient communicator did not adopt the existing gradient group")
            state = getattr(engine.module, "_alpamayo_compile_state", None)
            if state is not None:
                state.emit("gradient_group_connected", group=group.group_name)
        return native
    return wrapped


def static_parameter_policy_wrapper(original, options):
    @wraps(original)
    def wrapped(compile_kwargs):
        release = original(compile_kwargs)
        if not disabled():
            import torch
            # Parameter geometry is invariant. Dynamic parameter symbols would
            # make SHAPE_ENV guards index physical empty(0), instead of ds_shape.
            # Native semantic guards still validate ds_id/ds_shape/dtype/device.
            # Tensor inputs (including valid-token counts) remain dynamic.
            torch._dynamo.config.force_parameter_static_shapes = True
            torch._dynamo.config.force_nn_module_property_static_shapes = True
        return release
    return wrapped


def root_compile_wrapper(original, options):
    @wraps(original)
    def wrapped(model, *args, backend=None, **kwargs):
        if disabled():
            if backend is not None:
                kwargs["backend"] = backend
            return original(model, *args, **kwargs)
        state = getattr(model, "_alpamayo_compile_state", None)
        if state is None or not callable(backend) or not state.engine.is_deepcompile_active():
            raise RuntimeError("module regions must be initialized by native DeepCompile engine.compile()")
        import torch
        from .qwen_regions import vision_region, text_region
        from transformers.modeling_flash_attention_utils import lazy_import_flash_attention

        # Transformers requires preloading for fullgraph: imports and signature
        # inspection must finish before entering the Encoder/Qwen graphs.
        lazy_import_flash_attention("flash_attention_2")

        visual, text = model.vlm.model.visual, model.vlm.model.language_model
        # Each complete network is one fullgraph region. Embedding and LM head
        # are separate because upstream invokes them outside the text network.
        embedding, head = text.embed_tokens, model.vlm.lm_head
        regions = [
            ("encoder", visual, vision_region), ("qwen", text, text_region),
            ("embedding", embedding, torch.nn.Embedding.forward.__wrapped__),
            ("lm_head", head, torch.nn.Linear.forward.__wrapped__),
        ]
        owners = {}
        for name, module, _ in regions:
            # text.embed_tokens is deliberately assigned to its own region.
            for pname, p in module.named_parameters():
                if name == "qwen" and pname.startswith("embed_tokens."):
                    continue
                if p.requires_grad and p.ds_id in owners:
                    raise ValueError("tied parameters across compile regions are not yet supported")
                if p.requires_grad:
                    owners[p.ds_id] = name
        if set(owners) != state.expected:
            raise ValueError("Stage-1 trainable parameters exist outside the declared compile regions")
        dynamic = options.get("dynamic", True)
        if not isinstance(dynamic, bool):
            raise ValueError("module compile dynamic must be boolean")
        for name, module, fn in regions:
            compiled = torch.compile(fn, backend=state.backend(name, backend), fullgraph=True, dynamic=dynamic)
            if name in ("encoder", "qwen"):
                module._alpamayo_compiled_region = compiled
                module._alpamayo_region_dtype = next(module.parameters()).dtype
            else:
                module._alpamayo_compiled_parameter_region = compiled
        state.emit("armed", regions=[name for name, _, _ in regions], fullgraph=True,
                   dynamic=dynamic, checkpoint_use_reentrant=False, native_zero_stage=3)
    return wrapped


def accelerator_prepare_wrapper(original, options):
    @wraps(original)
    def wrapped(accelerator, *args, **kwargs):
        if disabled():
            return original(accelerator, *args, **kwargs)
        # DeepSpeed can be prepared more than once (e.g. just a DataLoader).
        result = original(accelerator, *args, **kwargs)
        candidates = result if isinstance(result, (tuple, list)) else (result,)
        for engine in candidates:
            if not hasattr(engine, "zero_optimization_stage") or not hasattr(engine, "module"):
                continue
            if getattr(engine.module, "_alpamayo_compile_state", None) is not None:
                continue
            validate_engine(engine)
            from deepspeed.compile.passes.zero3_compile import add_z3_gather_release
            state = CompileState(engine, options)
            engine.module._alpamayo_compile_state = state
            engine.compile(backend="inductor", compile_kwargs={"fullgraph": True, "dynamic": options.get("dynamic", True)},
                schedule=[(0, [add_z3_gather_release, state.audit_pass])])
            if not engine.is_deepcompile_active() or not getattr(engine, "_deepcompile_native_initialized", False):
                raise RuntimeError("native DeepCompile initialization fell back to ordinary torch.compile")
        return result
    return wrapped


def engine_forward_wrapper(original, options):
    @wraps(original)
    def wrapped(engine, *args, **kwargs):
        result = original(engine, *args, **kwargs)
        state = getattr(engine.module, "_alpamayo_compile_state", None)
        if state is not None:
            state.check_forward_coverage()
        return result
    return wrapped
