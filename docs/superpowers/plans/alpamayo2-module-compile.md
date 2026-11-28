# Alpamayo2 module compile implementation plan

Goal: compile the complete visual Encoder and Qwen text network with non-reentrant
activation checkpointing and the installed Hygon DeepSpeed ZeRO-3 compiler.

Scope: an isolated candidate based on TurboPhysAI 9746369. Keep the original model,
loss, optimizer, precision and seven-group installation intact until validation.

- [ ] Add failing numerical/gradient tests for vision metadata, padded FA and
  checkpointed region compilation in `test/optimizations/test_alpamayo2_qwen_regions.py`.
- [ ] Implement tensor-only Encoder and Qwen regions in
  `turbo_physai/optimizations/models/alpamayo2/qwen_regions.py`. Prepare grid,
  padding and visual indices outside the graph; never cache learned embeddings.
- [ ] Add failing tests, then the ZeRO-3 lifecycle adapter in
  `zero3_module_regions.py`: require native DeepCompile, `fullgraph=True`,
  `use_reentrant=False`, and no error suppression. Compile embedding and LM head
  too, so native gradient reduction covers every trainable parameter.
- [ ] Declare all actual replacement targets in `catalog.py`; add an experimental
  default-off group to `configs/recipe.yaml`, regenerate the checked config and
  generation record. Keep every existing group enabled. Connect DeepCompile's
  native reduce communicator to the existing independent gradient group using
  `turbo_physai/optimizations/models/alpamayo2/native/csrc/` (current source implementation; the original patch was removed).
- [ ] Run focused tests, including real CPU Inductor forward/backward, and the
  existing Alpamayo tests. Document exact upstream provenance and unsupported use.
- [ ] After tests pass, stop the requested memory holder, verify eight free cards,
  run a short README-based experiment and export a trace. Check both compile
  regions, backward graphs, native gather/reduce and actual Triton GPU kernels.
  Restart the holder after our training exits. Never stop other users' jobs.

Validation is staged: CPU correctness and fullgraph capture are not HCU ZeRO-3
validation. A registered/applied group is not proof of runtime compilation.
