from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple


class NearestTaskDispatcher:
    """Simple deterministic baseline that assigns each robot to the nearest available target."""

    def __init__(self, warehouse_env: Any) -> None:
        self.warehouse_env = warehouse_env

    def assign_tasks(self, tasks: Sequence[Dict[str, Any]]) -> None:
        for robot in self.warehouse_env.robots:
            robot["current_task"] = None
            robot["target"] = None
        for robot in self.warehouse_env.robots:
            if not robot["active"]:
                continue
            if not tasks:
                break
            nearest_task = min(
                tasks,
                key=lambda task: self._distance(robot["x"], robot["y"], task["pickup"]),
            )
            robot["current_task"] = nearest_task["task_id"]
            robot["target"] = nearest_task["pickup"]
            tasks = [task for task in tasks if task["task_id"] != nearest_task["task_id"]]

    def assign_pending_tasks(self, tasks: Sequence[Dict[str, Any]]) -> None:
        """Fill idle robots without interrupting tasks already in progress."""
        available_tasks = [task for task in tasks if task.get("status") == "pending"]
        for robot in self.warehouse_env.robots:
            if not robot["active"] or robot["current_task"] is not None or not available_tasks:
                continue
            nearest_task = min(
                available_tasks,
                key=lambda task: self._distance(robot["x"], robot["y"], task["pickup"]),
            )
            robot["current_task"] = nearest_task["task_id"]
            robot["target"] = nearest_task["pickup"]
            available_tasks = [
                task for task in available_tasks if task["task_id"] != nearest_task["task_id"]
            ]

    def choose_action(self, robot: Dict[str, Any]) -> int:
        target = robot["target"]
        if target is None:
            return 4
        dx = target[0] - robot["x"]
        dy = target[1] - robot["y"]
        if dx == 0 and dy == 0:
            return 4

        if abs(dx) >= abs(dy):
            desired_orientation = 1 if dx > 0 else 3
        else:
            desired_orientation = 2 if dy > 0 else 0

        orientation = int(robot["orientation"])
        if orientation == desired_orientation:
            return 0
        if (orientation - desired_orientation) % 4 == 2:
            return 1
        if (desired_orientation - orientation) % 4 == 1:
            return 3
        return 2

    def act(self, env: Any) -> List[int]:
        actions: List[int] = []
        for robot in env.robots:
            actions.append(self.choose_action(robot))
        return actions

    @staticmethod
    def _distance(x: int, y: int, target: Tuple[int, int]) -> float:
        tx, ty = target
        return ((x - tx) ** 2 + (y - ty) ** 2) ** 0.5
