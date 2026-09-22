from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


class ReplayBuffer:
    """Simple replay buffer for DQN training."""

    def __init__(self, capacity: int = 100_000, batch_size: int = 64) -> None:
        self.capacity = int(capacity)
        self.batch_size = int(batch_size)
        self.buffer: Deque[Dict[str, object]] = deque(maxlen=self.capacity)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        action_mask: Optional[List[bool]] = None,
        next_action_mask: Optional[List[bool]] = None,
    ) -> None:
        self.buffer.append(
            {
                "state": np.asarray(state, dtype=np.float32),
                "action": int(action),
                "reward": float(reward),
                "next_state": np.asarray(next_state, dtype=np.float32),
                "done": bool(done),
                "action_mask": list(action_mask) if action_mask is not None else None,
                "next_action_mask": list(next_action_mask) if next_action_mask is not None else None,
            }
        )

    def sample(self) -> List[Dict[str, object]]:
        if len(self.buffer) < self.batch_size:
            return list(self.buffer)
        indices = np.random.choice(len(self.buffer), size=self.batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]

    def __len__(self) -> int:
        return len(self.buffer)
