# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import ast


def test_reduction_codegen_changes_only_loop_pipeline_annotation():
    from turbo_physai.optimizations.models.alpamayo2.zero3_module_regions import single_stage_reductions
    source = '''
def kernel(x):
    for r0_offset in tl.range(0, n, R0_BLOCK, num_stages = 2):
        value = tl.load(x + r0_offset).to(tl.bfloat16).to(tl.float32)
        tl.store(x + r0_offset, value)
'''
    actual = single_stage_reductions(source)
    expected = source.replace("num_stages = 2", "num_stages = 1")
    assert actual == expected
    assert ast.dump(ast.parse(actual)) == ast.dump(ast.parse(expected))
    unrelated = "num_stages = 2\n# for r in tl.range(0, n, B, num_stages = 2):\n"
    assert single_stage_reductions(unrelated) == unrelated
