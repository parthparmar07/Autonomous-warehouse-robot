from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from src.algorithms.dqn.agent import DQNAgent


class IndependentQLearning:
    """Per-agent DQN baseline: each robot learns independently from its local observation."""

    def __init__(self, num_agents: int = 5, input_dim: int = 28, action_dim: int = 5, learning_rate: float = 1e-3, gamma: float = 0.99) -> None:
        self.num_agents = int(num_agents)
        self.input_dim = int(input_dim)
        self.action_dim = int(action_dim)
        self.agents = [
            DQNAgent(input_dim=input_dim, action_dim=action_dim, learning_rate=learning_rate, gamma=gamma)
            for _ in range(num_agents)
        ]

    def act(self, observations: Sequence[np.ndarray], action_masks: Sequence[Sequence[bool] | None] | None = None) -> List[int]:
        action_masks = action_masks or [None for _ in observations]
        actions: List[int] = []
        for idx, observation in enumerate(observations):
            mask = action_masks[idx]
            actions.append(self.agents[idx].act(observation, mask))
        return actions

    def sample_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, action_mask: Sequence[bool] | None = None, next_action_mask: Sequence[bool] | None = None) -> Dict[str, Any]:
        return {
            "state": np.asarray(state, dtype=np.float32),
            "action": int(action),
            "reward": float(reward),
            "next_state": np.asarray(next_state, dtype=np.float32),
            "done": bool(done),
            "action_mask": list(action_mask) if action_mask is not None else None,
            "next_action_mask": list(next_action_mask) if next_action_mask is not None else None,
        }
