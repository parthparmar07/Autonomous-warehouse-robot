from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RewardConfig:
    delivery_reward: float = 10.0
    collision_penalty: float = -5.0
    step_penalty: float = -0.1


DEFAULT_REWARD_CONFIG = RewardConfig()


def compute_reward(event: str, config: RewardConfig | None = None) -> float:
    """Return the configured scalar reward for a discrete event."""
    cfg = config or DEFAULT_REWARD_CONFIG
    if event == "delivery":
        return cfg.delivery_reward
    if event == "collision":
        return cfg.collision_penalty
    if event == "step":
        return cfg.step_penalty
    if event == "invalid_move":
        return cfg.collision_penalty
    raise ValueError(f"Unsupported event: {event}")
