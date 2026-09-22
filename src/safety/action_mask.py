from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np


def get_action_mask(agent_state: Sequence[int] | np.ndarray | None, warehouse: object | None = None) -> List[bool]:
    """Return a 5-element boolean mask representing valid actions."""
    return [True, True, True, True, True]


def apply_action_mask(q_values: np.ndarray | Sequence[float], action_mask: Sequence[bool]) -> np.ndarray:
    """Set masked actions to -inf while preserving valid actions."""
    q = np.asarray(q_values, dtype=float).copy()
    mask = np.asarray(action_mask, dtype=bool)
    q[~mask] = -np.inf
    return q
