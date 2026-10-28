"""Synthetic local range sensing, separate from the known static warehouse map."""
from __future__ import annotations
import math
import numpy as np
from src.environment.spec import SENSOR_CHANNELS


def ray_cells(dx: int, dy: int):
    """Supercover ray: corner contacts include both adjacent cells (conservative)."""
    x = y = 0
    sx, sy = (1 if dx > 0 else -1), (1 if dy > 0 else -1)
    ax, ay = abs(dx), abs(dy)
    tx, ty = (0.5 / ax if ax else math.inf), (0.5 / ay if ay else math.inf)
    while (x, y) != (dx, dy):
        if abs(tx - ty) < 1e-12:
            yield x + sx, y
            yield x, y + sy
            x, y = x + sx, y + sy
            tx, ty = tx + 1 / ax, ty + 1 / ay
        elif tx < ty:
            x, tx = x + sx, tx + 1 / ax
        else:
            y, ty = y + sy, ty + 1 / ay
        yield x, y


def sensor_frame(grid, robots, agent_id, config, seed, tick):
    robot = robots[agent_id]
    radius = config.sensor_radius
    result = np.zeros((2 * radius + 1, 2 * radius + 1, len(SENSOR_CHANNELS)), dtype=np.float32)
    # Right and back basis: row zero in the output is always the robot's front.
    forward = ((0, -1), (1, 0), (0, 1), (-1, 0))[robot["orientation"]]
    right = (-forward[1], forward[0])
    occupied = {(r["x"], r["y"]): r for i, r in enumerate(robots) if i != agent_id}
    height, width = grid.shape
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(tick), int(agent_id), 719]))
    for back in range(-radius, radius + 1):
        for lateral in range(-radius, radius + 1):
            if lateral or back:
                angle = abs(math.degrees(math.atan2(lateral, -back)))
                if not ((config.front_fov > 0 and angle <= config.front_fov / 2)
                        or (config.rear_fov > 0 and 180 - angle <= config.rear_fov / 2)):
                    continue
                if rng.random() < config.sensor_dropout:
                    continue
            dx = lateral * right[0] - back * forward[0]
            dy = lateral * right[1] - back * forward[1]
            target = (robot["x"] + dx, robot["y"] + dy)
            occluded = False
            for rx, ry in ray_cells(dx, dy):
                cell = (robot["x"] + rx, robot["y"] + ry)
                if cell == target:
                    break
                x, y = cell
                if not (0 <= x < width and 0 <= y < height) or int(grid[y, x]) in (1, 2) or cell in occupied:
                    occluded = True
                    break
            if occluded:
                continue
            channels = result[back + radius, lateral + radius]
            channels[0] = 1
            x, y = target
            if not (0 <= x < width and 0 <= y < height):
                channels[1] = 1
                continue
            code = int(grid[y, x])
            if code in (1, 2, 3, 4, 5):
                channels[code + 1] = 1
            elif code == 6:
                channels[5] = 1
            other = occupied.get(target)
            if other is not None:
                channels[7] = 1
                relative = (other["orientation"] - robot["orientation"]) * math.pi / 2
                channels[8:12] = (math.sin(relative), math.cos(relative), other["load_status"], float(not other.get("active", True)))
    return result
