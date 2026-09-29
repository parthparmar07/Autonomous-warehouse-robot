from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Sequence, Tuple

_BLOCKED_CELL_VALUES = (1, 2)  # GRID_SHELF, GRID_SAFETY_CORRIDOR


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

    def _next_step_towards(self, start: Tuple[int, int], target: Tuple[int, int]) -> Tuple[int, int]:
        """BFS shortest path over the floor plan (shelves and safety corridors are
        impassable) and return the first cell to move into. Without this, a robot
        whose target sits on the far side of the shelf block would just push into it
        forever -- the mask blocks the illegal move but never routes around it."""
        if start == target:
            return start
        grid = self.warehouse_env.grid
        width, height = self.warehouse_env.grid_size

        def passable(cell: Tuple[int, int]) -> bool:
            x, y = cell
            return 0 <= x < width and 0 <= y < height and int(grid[y, x]) not in _BLOCKED_CELL_VALUES

        if not passable(target):
            return start

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        visited = {start}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            if current == target:
                break
            cx, cy = current
            for neighbor in ((cx, cy - 1), (cx, cy + 1), (cx - 1, cy), (cx + 1, cy)):
                if neighbor in visited or not passable(neighbor):
                    continue
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)

        if target not in visited:
            return start  # no path found (shouldn't happen for reachable floor cells)

        path = [target]
        while path[-1] != start:
            path.append(came_from[path[-1]])
        path.reverse()
        return path[1] if len(path) > 1 else start

    def choose_action(self, robot: Dict[str, Any]) -> int:
        target = robot["target"]
        if target is None:
            return 4
        next_cell = self._next_step_towards((robot["x"], robot["y"]), tuple(target))
        dx = next_cell[0] - robot["x"]
        dy = next_cell[1] - robot["y"]
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
