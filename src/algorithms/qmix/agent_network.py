from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


class QMIXAgentNetwork(nn.Module):
    """Per-agent utility network with local observation encoder and GRU memory."""

    def __init__(self, input_dim: int = 30, hidden_dim: int = 64, action_dim: int = 5, gru_hidden_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden_dim, gru_hidden_dim)
        self.q_head = nn.Linear(gru_hidden_dim, action_dim)
        self.hidden_dim = gru_hidden_dim

    def initial_hidden_state(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_dim, dtype=torch.float32, device=next(self.parameters()).device)

    def reset_hidden(self) -> None:
        self._hidden = None

    def forward(self, x: torch.Tensor | np.ndarray, hidden_state: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        tensor = torch.as_tensor(x, dtype=torch.float32, device=next(self.parameters()).device)
        encoded = self.encoder(tensor)
        encoded = encoded.unsqueeze(0) if encoded.dim() == 1 else encoded
        if encoded.dim() == 2:
            encoded = encoded.unsqueeze(0)
        if hidden_state is None:
            hidden_state = self.initial_hidden_state(batch_size=encoded.size(1))
        if hidden_state.dim() == 2:
            hidden_state = hidden_state.unsqueeze(0)

        out, next_hidden = self.gru(encoded, hidden_state)
        q_values = self.q_head(out[-1]).squeeze(0)
        return q_values, next_hidden

    def act(
        self,
        observation: Sequence[float] | np.ndarray,
        action_mask: Optional[Sequence[bool]] = None,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> tuple[int, torch.Tensor]:
        """Returns (action, next_hidden_state). Callers doing a rollout over multiple
        steps must carry ``next_hidden_state`` back in as ``hidden_state`` on the next
        call -- passing ``None`` every time (the old behaviour) throws the GRU's memory
        away and makes it decide from the current observation alone."""
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=next(self.parameters()).device).unsqueeze(0)
        with torch.no_grad():
            q_values, next_hidden = self.forward(tensor, hidden_state)
            q_values_np = q_values.detach().cpu().numpy().copy()
            if q_values_np.ndim == 2:
                q_values_np = q_values_np[0]
        if action_mask is not None:
            q_values_np = np.asarray(q_values_np, dtype=float)
            q_values_np[~np.asarray(action_mask, dtype=bool)] = -np.inf
        return int(np.argmax(q_values_np)), next_hidden
