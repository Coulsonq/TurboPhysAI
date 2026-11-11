# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified by Hygon.
"""Reference ACT FFN compilation."""

import torch


def select_ffn(original, options):
    """Compile the reference FFN; reject removed implementations explicitly."""
    if options.get('implementation', 'reference_compile') != 'reference_compile':
        raise ValueError('FFN implementation must be reference_compile')
    return torch.compile(original, mode='max-autotune-no-cudagraphs', dynamic=False)
