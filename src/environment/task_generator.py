from __future__ import annotations

from typing import Any
import math

import numpy as np


def schedule_order(order: dict[str, Any], release: int) -> dict[str, Any]:
    """Move an unreleased order while preserving its relative deadline slack."""
    from copy import deepcopy
    if isinstance(release,bool) or not isinstance(release,int) or release<0:
        raise ValueError('Order release must be a nonnegative integer')
    scheduled=deepcopy(order)
    offset=release-int(scheduled.get('creation_timestep',0))
    if 'deadline' in scheduled:scheduled['deadline']+=offset
    scheduled['creation_timestep']=release
    return scheduled


def generate_tasks(warehouse: Any, num_tasks: int, seed: int | None = None,
                   start_task_id: int = 0) -> list[dict[str, Any]]:
    """Generate orders from fixed shelf-side pick bays to outbound drop bays.

    Randomness selects demand at real stations, never arbitrary aisle cells.
    Each task reserves one package on a shelf. Inventory and replenishment are
    outside this order-batch model.
    """
    if warehouse is None:
        raise ValueError("warehouse must not be None")
    if isinstance(num_tasks,bool) or not isinstance(num_tasks,int) or isinstance(start_task_id,bool) or not isinstance(start_task_id,int) or num_tasks < 0 or start_task_id < 0:
        raise ValueError("task count and starting ID must be nonnegative")
    if num_tasks == 0:
        return []
    pickups = getattr(warehouse, "pickup_bays", [])
    drops = getattr(warehouse, "drop_bays", [])
    if not pickups or not drops:
        raise ValueError("order generation requires shelf pickup bays and outbound drop bays")
    rng = np.random.default_rng(seed)
    tasks = []
    created_at = int(warehouse.episode_metrics["step_count"])
    for index in range(num_tasks):
        pickup_index = int(rng.integers(len(pickups)))
        drop_index = int(rng.integers(len(drops)))
        pickup, delivery = pickups[pickup_index], drops[drop_index]
        shelf = next((pickup[0] + dx, pickup[1] + dy)
                     for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                     if (pickup[0] + dx, pickup[1] + dy) in warehouse.obstacles)
        route=warehouse.path(pickup,delivery)
        minimum_service=(len(route)-1 if route else sum(abs(a-b) for a,b in zip(pickup,delivery)))+2*warehouse.config.service_ticks
        tasks.append({
            "task_id": start_task_id + index,
            "pickup": pickup,
            "delivery": delivery,
            "pickup_bay": f"P{pickup_index + 1}",
            "drop_bay": f"D{drop_index + 1}",
            "shelf": shelf,
            "creation_timestep": created_at,
            "status": "pending",
            "priority": int(rng.choice([1,2,3],p=[.7,.2,.1])),
            "deadline": created_at+math.ceil(rng.uniform(2,4)*(minimum_service+10)),
        })
    return tasks
