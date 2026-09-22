from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional

import numpy as np


class QMIXReplayBuffer:
    """Replay buffer for centralized QMIX training."""

    def __init__(self, capacity: int = 100_000, batch_size: int = 64) -> None:
        self.capacity = int(capacity)
        self.batch_size = int(batch_size)
        self.buffer: Deque[Dict[str, object]] = deque(maxlen=self.capacity)

    def add(
        self,
        state: np.ndarray,
        actions: List[int],
        reward: float,
        next_state: np.ndarray,
        done: bool,
        action_masks: Optional[List[List[bool]]] = None,
        next_action_masks: Optional[List[List[bool]]] = None,
        observations: Optional[List[np.ndarray]] = None,
        next_observations: Optional[List[np.ndarray]] = None,
    ) -> None:
        self.buffer.append(
            {
                "state": np.asarray(state, dtype=np.float32),
                "actions": list(actions),
                "reward": float(reward),
                "next_state": np.asarray(next_state, dtype=np.float32),
                "done": bool(done),
                "action_masks": [list(mask) for mask in action_masks] if action_masks is not None else None,
                "next_action_masks": [list(mask) for mask in next_action_masks] if next_action_masks is not None else None,
                "observations": [np.asarray(obs, dtype=np.float32) for obs in observations] if observations is not None else None,
                "next_observations": [np.asarray(obs, dtype=np.float32) for obs in next_observations] if next_observations is not None else None,
            }
        )

    def sample(self) -> List[Dict[str, object]]:
        if len(self.buffer) < self.batch_size:
            return list(self.buffer)
        indices = np.random.choice(len(self.buffer), size=self.batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]

    def __len__(self) -> int:
        return len(self.buffer)
