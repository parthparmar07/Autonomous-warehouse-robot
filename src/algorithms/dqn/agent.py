from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.algorithms.dqn.network import DQNNetwork


class DQNAgent:
    """A standard DQN agent with epsilon-greedy policy and target network."""

    def __init__(
        self,
        input_dim: int = 28,
        action_dim: int = 5,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
    ) -> None:
        self.online_network = DQNNetwork(input_dim=input_dim, action_dim=action_dim)
        self.target_network = DQNNetwork(input_dim=input_dim, action_dim=action_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.online_network.to(self.device)
        self.target_network.to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def act(self, observation: Sequence[float] | np.ndarray, action_mask: Sequence[bool] | None = None) -> int:
        if np.random.rand() < self.epsilon:
            valid_actions = list(range(5)) if action_mask is None else [idx for idx, allowed in enumerate(action_mask) if allowed]
            if not valid_actions:
                return 0
            return int(np.random.choice(valid_actions))
        return self.online_network.act(observation, action_mask)

    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.online_network.state_dict())

    def train_step(self, batch: List[dict]) -> float:
        states = torch.stack([torch.as_tensor(item["state"], dtype=torch.float32, device=self.device) for item in batch])
        actions = torch.tensor([item["action"] for item in batch], dtype=torch.int64, device=self.device)
        rewards = torch.tensor([item["reward"] for item in batch], dtype=torch.float32, device=self.device)
        next_states = torch.stack([torch.as_tensor(item["next_state"], dtype=torch.float32, device=self.device) for item in batch])
        dones = torch.tensor([float(item["done"]) for item in batch], dtype=torch.float32, device=self.device)

        current_q = self.online_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_network(next_states)
            next_q_masked = next_q.clone()
            for idx, item in enumerate(batch):
                next_mask = item.get("next_action_mask")
                if next_mask is not None:
                    next_q_masked[idx, ~np.asarray(next_mask, dtype=bool)] = -torch.inf
            target_q = rewards + self.gamma * (1.0 - dones) * next_q_masked.max(dim=1).values

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return float(loss.item())
