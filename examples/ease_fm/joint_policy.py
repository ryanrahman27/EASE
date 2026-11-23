"""Variable-joint flow matching: shared scalar action head, no robot-specific heads."""
import math

import torch
from torch import nn


def positions(count, width, device):
    position = torch.arange(count, device=device)[:, None]
    rate = torch.exp(torch.arange(0, width, 2, device=device)*(-math.log(10000)/width))
    return torch.stack([(position*rate).sin(), (position*rate).cos()], dim=-1).flatten(1)


def transformer(width, layers=1):
    return nn.TransformerEncoder(nn.TransformerEncoderLayer(width, 4, width*2,
                                 dropout=0., batch_first=True), layers, enable_nested_tensor=False)


class JointFlowPolicy(nn.Module):
    def __init__(self, history=False, width=64, horizon=16, auxiliary=False):
        super().__init__()
        self.has_history, self.width, self.horizon = history, width, horizon
        self.input = nn.Linear(3, width)  # noisy scalar command + q + dq
        self.task = nn.Sequential(nn.Linear(7, width), nn.SiLU(), nn.Linear(width, width))
        self.spatial = nn.ModuleList([transformer(width) for _ in range(2)])
        self.temporal = nn.ModuleList([transformer(width) for _ in range(2)])
        self.output = nn.Linear(width, 1)
        self.register_buffer("joint_mean", torch.zeros(2))
        self.register_buffer("joint_std", torch.ones(2))
        self.register_buffer("task_mean", torch.zeros(6))
        self.register_buffer("task_std", torch.ones(6))
        if auxiliary:
            if not history:
                raise ValueError("Response supervision requires history")
            self.response_input = nn.Linear(3, width)
            self.response_spatial = transformer(width)
            self.response_output = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 3))
            self.register_buffer("response_scale", torch.ones(3))
        if history:
            self.history_input = nn.Linear(11, width)
            self.history_spatial = transformer(width)
            self.history_temporal = transformer(width, 2)
            self.empty = nn.Parameter(torch.zeros(1, 1, width))
            self.history_output = nn.Linear(width, width)
            self.register_buffer("history_mean", torch.zeros(11))
            self.register_buffer("history_std", torch.ones(11))

    def response(self, joints, command, joint_mask, latent):
        """Predict normalized measured delta EE from present state and executed command.

        No goal, morphology, simulator Jacobian, or future observation is an input.
        This training-only head does not supply actions to the controller.
        """
        normalized = (joints-self.joint_mean)/self.joint_std
        values = torch.cat([normalized, command[..., None]], dim=-1)
        values = values.masked_fill(~joint_mask[..., None], 0)
        h = self.response_input(values) + latent
        h = h + positions(joints.shape[1], self.width, joints.device)[None]
        h = self.response_spatial(h, src_key_padding_mask=~joint_mask)
        h = h.masked_fill(~joint_mask[..., None], 0).sum(1)/joint_mask.sum(1, keepdim=True)
        return self.response_output(h)

    def encode(self, history, lengths, joint_mask):
        batch, steps, joints, _ = history.shape
        h = ((history-self.history_mean)/self.history_std).masked_fill(~joint_mask[:, None, :, None], 0)
        h = self.history_input(h) + positions(joints, self.width, h.device)[None, None]
        padding = (~joint_mask[:, None].expand(-1, steps, -1)).reshape(batch*steps, joints)
        h = self.history_spatial(h.reshape(batch*steps, joints, self.width), src_key_padding_mask=padding)
        h = h.reshape(batch, steps, joints, self.width).permute(0, 2, 1, 3).reshape(batch*joints, steps, self.width)
        h = torch.cat([self.empty.expand(batch*joints, -1, -1), h], dim=1)
        h = h + positions(steps+1, self.width, h.device)[None]
        causal = torch.triu(torch.ones(steps+1, steps+1, device=h.device, dtype=torch.bool), diagonal=1)
        h = self.history_temporal(h, mask=causal)
        indices = lengths[:, None].expand(-1, joints).reshape(-1)
        z = self.history_output(h[torch.arange(batch*joints, device=h.device), indices])
        return z.reshape(batch, joints, self.width).masked_fill(~joint_mask[:, :, None], 0)

    def forward(self, action, tau, joints, task, joint_mask, latent=None):
        batch, horizon, count = action.shape
        normalized = ((joints-self.joint_mean)/self.joint_std).masked_fill(~joint_mask[:, :, None], 0)
        values = torch.cat([action[..., None], normalized[:, None].expand(-1, horizon, -1, -1)], dim=-1)
        values = values.masked_fill(~joint_mask[:, None, :, None], 0)
        h = self.input(values)
        h = h + positions(count, self.width, h.device)[None, None]
        h = h + positions(horizon, self.width, h.device)[None, :, None]
        context = self.task(torch.cat([(task-self.task_mean)/self.task_std, tau[:, None]], dim=-1))
        h = h + context[:, None, None]
        if latent is not None:
            h = h + latent[:, None]
        padding = (~joint_mask[:, None].expand(-1, horizon, -1)).reshape(batch*horizon, count)
        for spatial, temporal in zip(self.spatial, self.temporal):
            h = spatial(h.reshape(batch*horizon, count, self.width), src_key_padding_mask=padding)
            h = h.reshape(batch, horizon, count, self.width).permute(0, 2, 1, 3)
            h = temporal(h.reshape(batch*count, horizon, self.width))
            h = h.reshape(batch, count, horizon, self.width).permute(0, 2, 1, 3)
        return self.output(h).squeeze(-1).masked_fill(~joint_mask[:, None], 0)

    @torch.no_grad()
    def sample(self, joints, task, joint_mask, noise, latent=None, steps=10):
        action = noise.clone().masked_fill(~joint_mask[:, None], 0)
        for index in range(steps):
            tau = torch.full((len(joints),), index/steps, device=joints.device)
            action += self(action, tau, joints, task, joint_mask, latent)/steps
        return action.clamp(-1, 1)*.04
