from __future__ import annotations

from typing import Dict, List, Tuple

Cell = Tuple[int, int]


def default_perimeter_corridors(grid_size: Tuple[int, int]) -> List[Cell]:
    """The warehouse's outer ring is a human-worker walkway, off-limits to robots.

    This is the layout ADR-003 and the literature review describe: a fixed map of
    corridor cells shared by the action mask (layer 2) and the environment's own
    physical block (layer 3).
    """
    width, height = grid_size
    cells: set[Cell] = set()
    for x in range(width):
        cells.add((x, 0))
        cells.add((x, height - 1))
    for y in range(height):
        cells.add((0, y))
        cells.add((width - 1, y))
    return sorted(cells)


def default_shelf_block(grid_size: Tuple[int, int]) -> List[Cell]:
    """A single central two-column shelf block with a crossing gap.

    Deliberately narrow: wide enough to be a real obstacle the mask and physics must
    respect, narrow enough that the nearest-task baseline's greedy (non-pathfinding)
    movement can still route around it without deadlocking.
    """
    width, height = grid_size
    if width < 12 or height < 12:
        return []
    center_x = width // 2
    gap_lo = height // 2 - 1
    gap_hi = height // 2
    cells: List[Cell] = []
    for x in (center_x - 1, center_x):
        for y in range(3, height - 3):
            if gap_lo <= y <= gap_hi:
                continue
            cells.append((x, y))
    return cells


def default_layout(grid_size: Tuple[int, int] = (20, 20)) -> Dict[str, List[Cell]]:
    """The canonical warehouse floor plan used for training, evaluation and the live demo.

    Pass this as ``WarehouseEnv(**default_layout(grid_size), ...)`` so every consumer
    (training scripts, evaluator, benchmark, API) shares one floor plan.
    """
    return {
        "obstacles": default_shelf_block(grid_size),
        "safety_corridors": default_perimeter_corridors(grid_size),
    }
