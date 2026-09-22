from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    """Simple DQN for a 28-dimensional local observation and 5 actions."""

    def __init__(self, input_dim: int = 28, hidden_dim: int = 128, action_dim: int = 5) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(x, dtype=torch.float32, device=next(self.parameters()).device)
        return self.network(tensor)

    def act(self, observation: Sequence[float] | np.ndarray, action_mask: Sequence[bool] | None = None) -> int:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=next(self.parameters()).device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.forward(tensor)[0]
        if action_mask is not None:
            mask = np.asarray(action_mask, dtype=bool)
            q_values = q_values.detach().cpu().numpy().copy()
            q_values[~mask] = -np.inf
            return int(np.argmax(q_values))
        return int(torch.argmax(q_values).item())
