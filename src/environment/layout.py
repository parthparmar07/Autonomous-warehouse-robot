from __future__ import annotations

from typing import Any, List, Tuple

Cell = Tuple[int, int]
from src.environment.spec import ENV_VERSION
LAYOUT_VERSION = ENV_VERSION


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
    """Compatibility helper returning the structured layout's rack cells."""
    return default_layout(grid_size)["obstacles"]


def default_layout(grid_size: Tuple[int, int] = (20, 20)) -> dict[str, Any]:
    """Fixed racks with adjacent pick bays, cross aisles and outbound drop bays.

    Robots transport orders, not entire racks. The first interior rows are staging
    space; the last three provide outbound circulation and bays. Maps below 7x7
    support movement unit tests only, with no order stations.
    """
    width, height = grid_size
    if width < 3 or height < 3:
        raise ValueError("warehouse dimensions must be at least 3x3")
    shelves: list[Cell] = []
    pickups: list[Cell] = []
    drops: list[Cell] = []
    if width >= 7 and height >= 7:
        rack_top = 4 if height >= 12 else 3
        rack_rows = [y for y in range(rack_top, height - 4)
                     if height < 12 or y not in (height // 2 - 1, height // 2)]
        if not rack_rows:
            rack_rows = [3]
        segments: list[list[int]] = []
        for y in rack_rows:
            if not segments or y != segments[-1][-1] + 1:
                segments.append([])
            segments[-1].append(y)
        for left in range(3, width - 3, 5):
            shelves.extend((x, y) for x in (left, left + 1) for y in rack_rows)
            for segment in segments:
                middle = segment[len(segment) // 2]
                pickups.extend(((left - 1, middle), (left + 2, middle)))
        drops = [(x, height - 2) for x in range(2, width - 2, 4)]
    staging = [(x, y) for y in range(1, min(4 if height >= 12 else 3, height - 1))
               for x in range(1, width - 1)]
    chargers = [(width - 3, 2), (width - 3, height - 3)] if width >= 12 and height >= 12 else []
    transfers = [(width - 2, height // 2)] if width >= 12 and height >= 12 else []
    queues = [(x - 1, y) for x, y in chargers]
    staging = [cell for cell in staging if cell not in chargers + transfers + queues]
    return {
        "obstacles": shelves,
        "safety_corridors": default_perimeter_corridors(grid_size),
        "pickup_bays": pickups,
        "drop_bays": drops,
        "charger_bays": chargers,
        "transfer_bays": transfers,
        "staging_cells": staging,
        "layout_version": LAYOUT_VERSION,
    }
