from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from src.environment.rewards import compute_reward
from src.safety.action_mask import compute_passable_mask, orientation_delta

ACTION_FORWARD = 0
ACTION_BACKWARD = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_STAY = 4

GRID_FREE = 0
GRID_SHELF = 1
GRID_SAFETY_CORRIDOR = 2
GRID_PICKUP = 3
GRID_DELIVERY = 4


class WarehouseEnv:
    """Minimal 2D warehouse simulator used for training and evaluation."""

    def __init__(
        self,
        grid_size: Tuple[int, int] = (20, 20),
        num_agents: int = 5,
        seed: int = 0,
        obstacles: Sequence[Tuple[int, int]] | None = None,
        safety_corridors: Sequence[Tuple[int, int]] | None = None,
    ) -> None:
        self.grid_size = tuple(grid_size)
        self.num_agents = int(num_agents)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.grid = np.zeros(self.grid_size, dtype=np.int8)
        self.obstacles = set(obstacles or [])
        self.safety_corridors = set(safety_corridors or [])
        self.robots: List[Dict[str, Any]] = []
        self.task_queue: List[Dict[str, Any]] = []
        self.completed_tasks = 0
        self.episode_metrics: Dict[str, float] = {
            "collision_count": 0.0,
            "step_count": 0.0,
            "reward": 0.0,
            "safety_overrides": 0.0,
        }

        self._apply_static_layout()

    def _apply_static_layout(self) -> None:
        for x, y in self.obstacles:
            if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                self.grid[y, x] = GRID_SHELF
        for x, y in self.safety_corridors:
            if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                self.grid[y, x] = GRID_SAFETY_CORRIDOR

    def reset(self, seed: int | None = None) -> Dict[str, Any]:
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.robots = []
        self.task_queue = []
        self.completed_tasks = 0
        self.episode_metrics = {
            "collision_count": 0.0,
            "step_count": 0.0,
            "reward": 0.0,
            "safety_overrides": 0.0,
        }

        for robot_id in range(self.num_agents):
            x = 1 + (robot_id % max(1, self.grid_size[0] - 2))
            y = 1 + (robot_id // max(1, self.grid_size[0] - 2))
            if x >= self.grid_size[0]:
                x = self.grid_size[0] - 2
            if y >= self.grid_size[1]:
                y = self.grid_size[1] - 2
            x, y = self._nearest_free_cell(x, y)
            self.robots.append(
                {
                    "id": robot_id,
                    "x": int(x),
                    "y": int(y),
                    "orientation": 1,
                    "current_task": None,
                    "target": None,
                    "load_status": 0,
                    "active": True,
                }
            )

        state = self.get_global_state()
        return state

    def step(self, actions: Sequence[int]) -> Dict[str, Any]:
        actions = list(actions)
        if len(actions) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} actions, got {len(actions)}")

        collision_count_before = self.episode_metrics["collision_count"]
        proposed_positions: List[Tuple[int, int]] = []
        for robot_id, action in enumerate(actions):
            robot = self.robots[robot_id]
            dx, dy = self._action_delta(robot["orientation"], action)
            new_x = robot["x"] + dx
            new_y = robot["y"] + dy
            if action in (ACTION_LEFT, ACTION_RIGHT):
                new_x = robot["x"]
                new_y = robot["y"]
            if (
                action in (ACTION_FORWARD, ACTION_BACKWARD)
                and self._is_in_bounds(new_x, new_y)
                and self.grid[new_y, new_x] == GRID_SAFETY_CORRIDOR
            ):
                # Layer-3 diagnostic (ADR-003): counts even if a caller ignored the
                # action mask -- the physical block below is what actually stops it.
                self.episode_metrics["safety_overrides"] += 1.0
            proposed_positions.append((new_x, new_y))

        occupied: Dict[Tuple[int, int], int] = {}
        for robot_id, (new_x, new_y) in enumerate(proposed_positions):
            robot = self.robots[robot_id]
            if not self._is_in_bounds(new_x, new_y):
                continue
            if self.grid[new_y, new_x] in (GRID_SHELF, GRID_SAFETY_CORRIDOR):
                continue
            occupied.setdefault((new_x, new_y), robot_id)
            if robot_id != occupied[(new_x, new_y)] and (new_x, new_y) != (robot["x"], robot["y"]):
                self.episode_metrics["collision_count"] += 1.0

        for robot_id, action in enumerate(actions):
            robot = self.robots[robot_id]
            if action == ACTION_LEFT:
                robot["orientation"] = (robot["orientation"] + 3) % 4
            elif action == ACTION_RIGHT:
                robot["orientation"] = (robot["orientation"] + 1) % 4

            dx, dy = self._action_delta(robot["orientation"], action)
            new_x = robot["x"] + dx
            new_y = robot["y"] + dy
            if action in (ACTION_LEFT, ACTION_RIGHT):
                new_x = robot["x"]
                new_y = robot["y"]

            if not self._is_in_bounds(new_x, new_y):
                continue
            if self.grid[new_y, new_x] in (GRID_SHELF, GRID_SAFETY_CORRIDOR):
                continue
            if (new_x, new_y) in occupied and occupied[(new_x, new_y)] == robot_id:
                robot["x"] = int(new_x)
                robot["y"] = int(new_y)

        delivery_events = self._update_tasks()
        collision_events = int(self.episode_metrics["collision_count"] - collision_count_before)
        reward = compute_reward("step")
        reward += delivery_events * compute_reward("delivery")
        reward += collision_events * compute_reward("collision")
        self.episode_metrics["step_count"] += 1.0
        self.episode_metrics["reward"] += reward
        state = self.get_global_state()
        state["reward"] = reward
        state["done"] = not self.task_queue
        return state

    def _update_tasks(self) -> int:
        deliveries = 0
        remaining_tasks: List[Dict[str, Any]] = []
        for task in self.task_queue:
            assigned_robot = next(
                (robot for robot in self.robots if robot["current_task"] == task["task_id"]),
                None,
            )
            if assigned_robot is None:
                remaining_tasks.append(task)
                continue

            position = (assigned_robot["x"], assigned_robot["y"])
            if task["status"] == "pending" and position == tuple(task["pickup"]):
                task["status"] = "picked"
                assigned_robot["load_status"] = 1
                assigned_robot["target"] = task["delivery"]
            if task["status"] == "picked" and position == tuple(task["delivery"]):
                task["status"] = "delivered"
                assigned_robot["load_status"] = 0
                assigned_robot["current_task"] = None
                assigned_robot["target"] = None
                self.completed_tasks += 1
                deliveries += 1
                continue
            remaining_tasks.append(task)
        self.task_queue = remaining_tasks
        return deliveries

    def _action_delta(self, orientation: int, action: int) -> Tuple[int, int]:
        return orientation_delta(orientation, action)

    def _is_in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]

    def _nearest_free_cell(self, x: int, y: int) -> Tuple[int, int]:
        """Guard against spawning a robot on a shelf or safety-corridor cell: search
        outward in expanding rings until a free cell is found."""
        if self._is_in_bounds(x, y) and self.grid[y, x] == GRID_FREE:
            return x, y
        max_radius = max(self.grid_size)
        for radius in range(1, max_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = x + dx, y + dy
                    if self._is_in_bounds(nx, ny) and self.grid[ny, nx] == GRID_FREE:
                        return nx, ny
        return x, y

    def get_global_state(self) -> Dict[str, Any]:
        return {
            "grid_size": self.grid_size,
            "grid": self.grid.copy().tolist(),
            "robots": [dict(robot) for robot in self.robots],
            "task_queue": [dict(task) for task in self.task_queue],
            "metrics": dict(self.episode_metrics),
            "completed_tasks": self.completed_tasks,
        }

    def get_local_observation(self, agent_id: int) -> np.ndarray:
        """28 local/task features plus a 2-feature encoding of the robot's own heading.

        The local window and target bearing are both in fixed world-axis coordinates,
        not rotated to face the robot's heading -- so without the heading itself as a
        feature, the *same* observation would have to map to different correct actions
        depending on an orientation the policy never sees (whether FORWARD moves it
        toward or away from the target, or whether a masked FORWARD needs a LEFT or a
        RIGHT to clear). That is a partial-observability bug, not a training difficulty:
        no amount of extra training data fixes an observation that is genuinely
        ambiguous. sin/cos of the heading (rather than a raw 0-3 integer or one-hot) is
        continuous and keeps the four headings equally far apart for the network.
        """
        robot = self.robots[agent_id]
        x, y = robot["x"], robot["y"]
        local_window = np.zeros((5, 5), dtype=np.float32)

        for dy in range(-2, 3):
            for dx in range(-2, 3):
                wx = x + dx
                wy = y + dy
                if 0 <= wx < self.grid_size[0] and 0 <= wy < self.grid_size[1]:
                    local_window[dy + 2, dx + 2] = float(self.grid[wy, wx])

        target = robot["target"]
        if target is None:
            target = (x, y)
        dx = target[0] - x
        dy = target[1] - y
        norm = max(1.0, np.hypot(float(dx), float(dy)))
        target_bearing = np.array([dx / norm, dy / norm], dtype=np.float32)
        load_value = float(robot["load_status"])
        heading_radians = float(robot["orientation"]) * (np.pi / 2.0)
        heading_encoding = np.array([np.sin(heading_radians), np.cos(heading_radians)], dtype=np.float32)

        observation = np.concatenate(
            [
                local_window.reshape(-1),
                target_bearing,
                np.array([load_value], dtype=np.float32),
                heading_encoding,
            ]
        )
        return observation.astype(np.float32)

    def get_action_mask(self, agent_id: int) -> List[bool]:
        """Layer 2 (ADR-003): actions that would leave the grid, enter a shelf or
        safety-corridor cell, or collide with another robot's current cell are masked
        out before the policy's argmax. STAY, LEFT and RIGHT never move, so they are
        always legal -- a robot always has at least one safe action available."""
        robot = self.robots[agent_id]
        occupied = {
            (other["x"], other["y"])
            for idx, other in enumerate(self.robots)
            if idx != agent_id and other.get("active", True)
        }
        return compute_passable_mask(robot["x"], robot["y"], robot["orientation"], self.grid, occupied)

    def render(self) -> str:
        grid = self.grid.copy()
        for robot in self.robots:
            x, y = robot["x"], robot["y"]
            if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                grid[y, x] = 5 + robot["id"]

        rows: List[str] = []
        for row in grid:
            rows.append(" ".join(str(int(cell)) for cell in row))
        return "\n".join(rows)
