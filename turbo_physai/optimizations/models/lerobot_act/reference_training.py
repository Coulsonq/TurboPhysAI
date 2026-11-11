# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
# ACT policy loss adapted from https://github.com/huggingface/lerobot,
# src/lerobot/policies/act/modeling_act.py at 3f2179f3b69708b6ad009b2e7685dd9d05269ee1.

"""Traceable replacements for the reference ACT training pipeline."""

from __future__ import annotations

import copy
import functools
import inspect
import torch.nn.functional as F


def ddp_kwargs_wrapper(original, options):
    """Match reference DDP and apply explicitly selected ACT DDP options."""
    signature = inspect.signature(original)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        bound.arguments['find_unused_parameters'] = False
        if 'static_graph' in options:
            bound.arguments['static_graph'] = options['static_graph']
        return original(*bound.args, **bound.kwargs)

    return wrapped


def accelerator_wrapper(original, options):
    """Enable Accelerate's non-blocking H2D without discarding its options."""
    del options

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        from accelerate.utils import DataLoaderConfiguration

        current = kwargs.get('dataloader_config')
        config = copy.copy(current) if current is not None else DataLoaderConfiguration()
        config.non_blocking = True
        kwargs['dataloader_config'] = config
        return original(*args, **kwargs)

    return wrapped


def act_policy_forward(self, batch):
    """Retain the ACT loss calculation but defer scalar reads to logging."""
    from lerobot.utils.constants import ACTION, OBS_IMAGES

    if self.config.image_features:
        batch = dict(batch)
        batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

    actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)
    abs_err = F.l1_loss(batch[ACTION], actions_hat, reduction='none')
    valid_mask = ~batch['action_is_pad'].unsqueeze(-1)
    num_valid = valid_mask.sum() * abs_err.shape[-1]
    l1_loss = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)
    loss_dict = {'l1_loss': l1_loss.detach()}
    if self.config.use_vae and log_sigma_x2_hat is not None:
        mean_kld = (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp())).sum(-1).mean()
        loss_dict['kld_loss'] = mean_kld.detach()
        loss = l1_loss + mean_kld * self.config.kl_weight
    else:
        loss = l1_loss
    return loss, loss_dict

