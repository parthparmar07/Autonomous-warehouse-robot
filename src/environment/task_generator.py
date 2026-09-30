from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def generate_tasks(warehouse: Any, num_tasks: int, seed: int | None = None) -> List[Dict[str, Any]]:
    """Generate a deterministic list of pickup/delivery pairs.

    Pickup and delivery cells are drawn only from free floor cells -- never a shelf or a
    safety-corridor cell -- so every generated task is actually reachable under the
    action mask (ADR-003). Without a ``grid`` on ``warehouse`` (a bare duck-typed stub),
    falls back to sampling the full grid.
    """
    if warehouse is None:
        raise ValueError("warehouse must not be None")

    rng = np.random.default_rng(seed)
    tasks: List[Dict[str, Any]] = []
    width, height = warehouse.grid_size
    grid = getattr(warehouse, "grid", None)
    if grid is not None:
        free_cells = [(x, y) for y in range(height) for x in range(width) if int(grid[y, x]) == 0]
    else:
        free_cells = [(x, y) for y in range(height) for x in range(width)]
    if not free_cells:
        raise ValueError("warehouse grid has no free cells to place tasks on")

    for idx in range(num_tasks):
        pickup = free_cells[int(rng.integers(0, len(free_cells)))]
        delivery = free_cells[int(rng.integers(0, len(free_cells)))]

        tasks.append(
            {
                "task_id": idx,
                "pickup": pickup,
                "delivery": delivery,
                "creation_timestep": 0,
                "status": "pending",
            }
        )

    return tasks
