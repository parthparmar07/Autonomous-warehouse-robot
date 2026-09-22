from __future__ import annotations

from typing import Iterable


class WarehouseRenderer:
    """Simple text-based warehouse renderer for debugging and episode replay."""

    def __init__(self, warehouse_env: object) -> None:
        self.warehouse_env = warehouse_env

    def render(self) -> str:
        grid = getattr(self.warehouse_env, "grid", None)
        if grid is None:
            return "No warehouse grid available"
        return "\n".join(" ".join(str(int(cell)) for cell in row) for row in grid)

    def render_trajectory(self, trajectory: Iterable[tuple[int, int]]) -> str:
        steps = list(trajectory)
        if not steps:
            return "No trajectory available"
        return " -> ".join(f"({x},{y})" for x, y in steps)
