from __future__ import annotations

import random
from typing import Sequence


def bootstrap_mean(values: Sequence[float], samples: int = 200) -> tuple[float, float]:
    """Return (mean, std) from a simple bootstrap-like resampling approximation."""
    if not values:
        return 0.0, 0.0
    draws = [random.choice(list(values)) for _ in range(samples)]
    mean_value = sum(draws) / len(draws)
    std_value = (sum((x - mean_value) ** 2 for x in draws) / len(draws)) ** 0.5
    return mean_value, std_value
