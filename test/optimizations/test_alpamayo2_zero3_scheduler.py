# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch


def _add_buffer(x, *, buffers):
    return x + buffers["dq"][0]


@pytest.mark.parametrize("nested", [False, True])
def test_native_zero3_scheduler_preserves_keyword_tensor_dependencies(nested):
    from deepspeed.compile.list_schedule import get_node_requirements, make_graph_from_schedule
    from turbo_physai.optimizations.models.alpamayo2.deepspeed_compat import node_requirements_wrapper
    get_node_requirements = node_requirements_wrapper(get_node_requirements,{})
    graph = torch.fx.Graph()
    x, y = graph.placeholder("x"), graph.placeholder("y")
    view = graph.call_function(torch.reshape, (y, (-1,)), name="view_181")
    if nested:
        result = graph.call_function(_add_buffer, (x,), {"buffers": {"dq": [view]}})
    else:
        result = graph.call_function(torch.add, (x,), {"other": view})
    output = graph.output(result)
    required = get_node_requirements(result, [x, y])
    rebuilt = make_graph_from_schedule([x, y] + required + [output])
    rebuilt.lint()
    assert required.index(view) < required.index(result)
    module = torch.fx.GraphModule({}, rebuilt)
    a, b = torch.randn(4), torch.randn(2, 2)
    torch.testing.assert_close(module(a, b), a + b.flatten())
