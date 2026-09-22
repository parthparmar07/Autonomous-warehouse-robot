from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from src.algorithms.dqn.replay_buffer import ReplayBuffer
from src.algorithms.iql.agent import IndependentQLearning


class IQLTrainer:
    """Minimal trainer for the independent Q-learning baseline."""

    def __init__(self, num_agents: int = 5, replay_capacity: int = 100_000, batch_size: int = 64, learning_rate: float = 1e-3, gamma: float = 0.99) -> None:
        self.model = IndependentQLearning(num_agents=num_agents, input_dim=28, action_dim=5, learning_rate=learning_rate, gamma=gamma)
        self.replay_buffers = [ReplayBuffer(capacity=replay_capacity, batch_size=batch_size) for _ in range(num_agents)]

    def add_transition(self, agent_id: int, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, action_mask: Sequence[bool] | None = None, next_action_mask: Sequence[bool] | None = None) -> None:
        self.replay_buffers[agent_id].add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            action_mask=action_mask,
            next_action_mask=next_action_mask,
        )

    def train_step(self) -> List[float]:
        losses: List[float] = []
        for agent_id, agent in enumerate(self.model.agents):
            samples = self.replay_buffers[agent_id].sample()
            if not samples:
                losses.append(0.0)
                continue
            losses.append(agent.train_step(samples))
        return losses
