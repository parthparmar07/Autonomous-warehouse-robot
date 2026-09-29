from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

ACTION_FORWARD = 0
ACTION_BACKWARD = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_STAY = 4

# Grid cell codes a robot may never enter. Kept as plain ints (rather than importing
# warehouse_env's enum) so this module has no dependency on the environment package --
# WarehouseEnv imports FROM here, not the other way around.
GRID_SHELF = 1
GRID_SAFETY_CORRIDOR = 2
BLOCKED_CELL_VALUES = (GRID_SHELF, GRID_SAFETY_CORRIDOR)


def orientation_delta(orientation: int, action: int) -> Tuple[int, int]:
    """(dx, dy) for FORWARD/BACKWARD given a heading; LEFT/RIGHT/STAY never move."""
    if action == ACTION_FORWARD:
        return {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}[orientation % 4]
    if action == ACTION_BACKWARD:
        return {0: (0, 1), 1: (-1, 0), 2: (0, -1), 3: (1, 0)}[orientation % 4]
    return 0, 0


def compute_passable_mask(
    x: int,
    y: int,
    orientation: int,
    grid: np.ndarray,
    occupied_cells: Iterable[Tuple[int, int]] | None = None,
) -> List[bool]:
    """The single source of truth for which of the 5 actions are safe from (x, y, orientation).

    Blocks a move that would leave the grid, enter a shelf or human-worker safety-corridor
    cell, or step onto a cell another active robot currently occupies. LEFT, RIGHT and STAY
    never change position, so they are always legal -- this is the invariant ADR-003 requires
    (a robot always has at least one legal action).

    Shared by ``WarehouseEnv.get_action_mask`` (layer 2: masking before the policy's argmax)
    and the environment's own physical step logic (layer 3: the redundant post-hoc check), so
    the two layers read the same grid and can never drift apart.
    """
    grid = np.asarray(grid)
    height, width = grid.shape[0], grid.shape[1]
    occupied = set(occupied_cells or ())
    mask = [True, True, True, True, True]

    for action in (ACTION_FORWARD, ACTION_BACKWARD):
        dx, dy = orientation_delta(orientation, action)
        new_x, new_y = x + dx, y + dy
        if not (0 <= new_x < width and 0 <= new_y < height):
            mask[action] = False
            continue
        if int(grid[new_y, new_x]) in BLOCKED_CELL_VALUES:
            mask[action] = False
            continue
        if (new_x, new_y) in occupied:
            mask[action] = False

    return mask


def get_action_mask(agent_state: Sequence[int] | np.ndarray | None, warehouse: object | None = None) -> List[bool]:
    """Return a 5-element boolean mask for a robot's ``(x, y[, orientation])`` state.

    ``warehouse`` is any object exposing a ``grid`` attribute (a WarehouseEnv, or anything
    duck-typed the same way). Without one to check cells against, every action is reported
    legal -- callers that need a real safety guarantee must supply the warehouse.
    """
    if warehouse is None or not hasattr(warehouse, "grid"):
        return [True, True, True, True, True]

    state = list(agent_state) if agent_state is not None else [0, 0, 1]
    x = int(state[0]) if len(state) > 0 else 0
    y = int(state[1]) if len(state) > 1 else 0
    orientation = int(state[2]) if len(state) > 2 else 1
    return compute_passable_mask(x, y, orientation, np.asarray(warehouse.grid))


def apply_action_mask(q_values: np.ndarray | Sequence[float], action_mask: Sequence[bool]) -> np.ndarray:
    """Set masked actions to -inf while preserving valid actions."""
    q = np.asarray(q_values, dtype=float).copy()
    mask = np.asarray(action_mask, dtype=bool)
    q[~mask] = -np.inf
    return q
