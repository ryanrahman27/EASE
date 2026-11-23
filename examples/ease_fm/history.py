"""Causal transition encoder. The student receives no embodiment identifiers."""
import numpy as np
import torch
from torch import nn

from .baseline import FlowPolicy
from .benchmark import HISTORY_LENGTH, transition_token

TOKEN_DIM = 41


class HistoryFlowPolicy(FlowPolicy):
    def __init__(self):
        super().__init__()
        self.history_input = nn.Linear(TOKEN_DIM, 128)
        self.history_position = nn.Parameter(torch.randn(1, HISTORY_LENGTH + 1, 128) * .02)
        self.empty_history = nn.Parameter(torch.zeros(1, 1, 128))
        layer = nn.TransformerEncoderLayer(128, 4, 256, dropout=0., batch_first=True)
        self.history_encoder = nn.TransformerEncoder(layer, 3)
        self.history_output = nn.Linear(128, 128)
        self.register_buffer("history_mean", torch.zeros(TOKEN_DIM))
        self.register_buffer("history_std", torch.ones(TOKEN_DIM))

    def encode(self, history, lengths):
        if history.shape[1:] != (HISTORY_LENGTH, TOKEN_DIM):
            raise ValueError("Expected a batch of 32 x 41 history tokens")
        normalized = (history - self.history_mean) / self.history_std
        tokens = torch.cat([self.empty_history.expand(len(history), -1, -1),
                            self.history_input(normalized)], dim=1)
        tokens = tokens + self.history_position
        causal = torch.triu(torch.ones(HISTORY_LENGTH+1, HISTORY_LENGTH+1,
                                       device=history.device, dtype=torch.bool), diagonal=1)
        encoded = self.history_encoder(tokens, mask=causal)
        # Index zero is an always-valid empty token. Future/padded tokens cannot
        # affect this position or the last valid transition under causal attention.
        return self.history_output(encoded[torch.arange(len(history), device=history.device), lengths])


class HistoryBuffer:
    def __init__(self, tokens=()):
        self.tokens = [np.asarray(token, dtype=np.float32).copy() for token in tokens][-HISTORY_LENGTH:]

    def append(self, observation, action, next_observation):
        self.tokens.append(transition_token(observation, action, next_observation))
        self.tokens = self.tokens[-HISTORY_LENGTH:]

    def array(self):
        array = np.zeros((HISTORY_LENGTH, TOKEN_DIM), dtype=np.float32)
        if self.tokens:
            array[:len(self.tokens)] = self.tokens
        return array, len(self.tokens)
