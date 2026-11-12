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
# Forward adapted from https://github.com/huggingface/lerobot,
# src/lerobot/policies/act/modeling_act.py at 3f2179f3b69708b6ad009b2e7685dd9d05269ee1.
"""ACT.forward for the verified LeRobot revision without the fixed VAE buffer clone."""

import functools
import hashlib
import inspect
from pathlib import Path


EXPECTED_SOURCE_SHA256 = 'a6c1a49c5daafa513b9abee9b986205d327e25c70ed13623ab2e6c0ab6b1f577'


def make_vae_no_clone_forward(original):
    # Reject unknown source or an already-wrapped function, whose semantics
    # cannot safely be reproduced by this fixed-source candidate.
    if inspect.unwrap(original) is not original:
        raise ValueError('ACT.forward is already wrapped')
    path = inspect.getsourcefile(original)
    if path is None or hashlib.sha256(Path(path).read_bytes()).hexdigest() != EXPECTED_SOURCE_SHA256:
        raise ValueError('ACT source does not match the tested revision')
    scope = original.__globals__
    torch = scope['torch']
    einops = scope['einops']
    ACTION = scope['ACTION']
    OBS_IMAGES = scope['OBS_IMAGES']
    OBS_ENV_STATE = scope['OBS_ENV_STATE']
    OBS_STATE = scope['OBS_STATE']

    @functools.wraps(original)
    def forward(self, batch):
        batch_size = batch[OBS_IMAGES][0].shape[0]
        if self.config.use_vae and self.training:
            if ACTION not in batch:
                raise ValueError('VAE training requires actions in the batch.')
            cls_embed = einops.repeat(self.vae_encoder_cls_embed.weight, '1 d -> b 1 d', b=batch_size)
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE])
                robot_state_embed = robot_state_embed.unsqueeze(1)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])
            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)
            pos_embed = self.vae_encoder_pos_enc.detach()
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False, device=batch[OBS_STATE].device,
            )
            key_padding_mask = torch.cat([cls_joint_is_pad, batch['action_is_pad']], axis=1)
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, :self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim:]
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch[OBS_STATE].device
            )

        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))
        for img in batch[OBS_IMAGES]:
            cam_features = self.backbone(img)['feature_map']
            cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
            cam_features = self.encoder_img_feat_input_proj(cam_features)
            cam_features = einops.rearrange(cam_features, 'b c h w -> (h w) b c')
            cam_pos_embed = einops.rearrange(cam_pos_embed, 'b c h w -> (h w) b c')
            encoder_in_tokens.extend(list(cam_features))
            encoder_in_pos_embed.extend(list(cam_pos_embed))
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype, device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in, encoder_out, encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )
        decoder_out = decoder_out.transpose(0, 1)
        actions = self.action_head(decoder_out)
        return actions, (mu, log_sigma_x2)

    return forward


def vae_no_clone_wrapper(original, options):
    del options
    return make_vae_no_clone_forward(original)
