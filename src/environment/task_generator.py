from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def generate_tasks(warehouse: Any, num_tasks: int, seed: int | None = None) -> List[Dict[str, Any]]:
    """Generate a deterministic list of pickup/delivery pairs."""
    if warehouse is None:
        raise ValueError("warehouse must not be None")

    rng = np.random.default_rng(seed)
    tasks: List[Dict[str, Any]] = []
    width, height = warehouse.grid_size

    for idx in range(num_tasks):
        pickup = (
            int(rng.integers(0, width)),
            int(rng.integers(0, height)),
        )
        delivery = (
            int(rng.integers(0, width)),
            int(rng.integers(0, height)),
        )

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
