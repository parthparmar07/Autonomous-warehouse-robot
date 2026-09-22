from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from src.algorithms.dqn.agent import DQNAgent
from src.algorithms.dqn.replay_buffer import ReplayBuffer


class DQNTrainer:
    """Minimal trainer for the DQN base implementation."""

    def __init__(
        self,
        replay_capacity: int = 100_000,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
    ) -> None:
        self.agent = DQNAgent(
            input_dim=28,
            action_dim=5,
            learning_rate=learning_rate,
            gamma=gamma,
        )
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity, batch_size=batch_size)
        self.batch_size = batch_size

    def add_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        action_mask: List[bool] | None = None,
        next_action_mask: List[bool] | None = None,
    ) -> None:
        self.replay_buffer.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            action_mask=action_mask,
            next_action_mask=next_action_mask,
        )

    def train_step(self) -> float:
        samples = self.replay_buffer.sample()
        if not samples:
            return 0.0
        return self.agent.train_step(samples)

    def act(self, observations: List[np.ndarray], action_masks: List[List[bool] | None] | None = None) -> List[int]:
        """Select one independently learned action for each local observation."""
        masks = action_masks or [None for _ in observations]
        return [self.agent.act(observation, mask) for observation, mask in zip(observations, masks)]

    def update_target(self) -> None:
        self.agent.update_target_network()
