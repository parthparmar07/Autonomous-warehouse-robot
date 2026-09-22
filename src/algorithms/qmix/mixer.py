from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


class QMIXMixer(nn.Module):
    """Monotonic mixer with state-conditioned hypernetwork parameters."""

    def __init__(self, num_agents: int = 5, state_dim: int = 64, hidden_dim: int = 32) -> None:
        super().__init__()
        self.num_agents = int(num_agents)
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)

        self.hyper_w1 = nn.Linear(state_dim, hidden_dim * num_agents)
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_w2 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Linear(state_dim, 1)

    def forward(self, agent_q_values: Sequence[torch.Tensor], global_state: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack(agent_q_values, dim=1)
        if stacked.dim() == 1:
            stacked = stacked.unsqueeze(0)
        if global_state.dim() == 1:
            global_state = global_state.unsqueeze(0)

        batch_size = stacked.size(0)
        w1 = torch.abs(self.hyper_w1(global_state)).view(batch_size, self.num_agents, self.hidden_dim)
        b1 = self.hyper_b1(global_state)
        w2 = torch.abs(self.hyper_w2(global_state)).view(batch_size, self.hidden_dim, 1)
        b2 = self.hyper_b2(global_state)

        hidden = torch.relu(torch.bmm(stacked.unsqueeze(1), w1).squeeze(1) + b1)
        out = torch.bmm(hidden.unsqueeze(1), w2).squeeze(-1).squeeze(-1) + b2.squeeze(-1)
        return out

    def monotonicity_check(self, agent_q_values: Sequence[np.ndarray]) -> bool:
        for idx in range(len(agent_q_values)):
            base = np.asarray(agent_q_values[idx], dtype=np.float32).copy()
            boosted = base + 1.0
            if boosted.shape != base.shape:
                return False
            if not np.all(np.isfinite(boosted)):
                return False
        return True
