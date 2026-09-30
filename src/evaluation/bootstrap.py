from __future__ import annotations

import random
from typing import Sequence, Tuple

import numpy as np


def bootstrap_mean(values: Sequence[float], samples: int = 200) -> tuple[float, float]:
    """Return (mean, std) from a simple bootstrap-like resampling approximation.

    Note: this draws ``samples`` individual values from ``values`` with replacement and
    reports their mean/std, which estimates the population mean and the standard
    deviation of individual observations -- it is not the standard error of the mean.
    Kept as-is for backward compatibility; use ``bootstrap_confidence_interval`` below
    for an actual confidence interval on the mean.
    """
    if not values:
        return 0.0, 0.0
    draws = [random.choice(list(values)) for _ in range(samples)]
    mean_value = sum(draws) / len(draws)
    std_value = (sum((x - mean_value) ** 2 for x in draws) / len(draws)) ** 0.5
    return mean_value, std_value


def bootstrap_confidence_interval(
    values: Sequence[float],
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> Tuple[float, Tuple[float, float]]:
    """A real percentile bootstrap CI on the mean: resample ``len(values)`` observations
    with replacement ``samples`` times, take each resample's mean, and report the
    ``confidence`` percentile interval of that distribution of means.

    With very few underlying observations (e.g. the 3 seeds per fleet size the sealed
    evaluation manifests currently provide), the interval this produces is correspondingly
    wide and should be read as "we don't have much statistical power yet," not as a
    precise estimate -- that is an honest property of having 3 samples, not a bug here.
    """
    if not values:
        return 0.0, (0.0, 0.0)
    arr = np.asarray(values, dtype=float)
    point_estimate = float(arr.mean())
    if len(arr) < 2:
        return point_estimate, (point_estimate, point_estimate)
    rng = np.random.default_rng(seed)
    resample_means = np.empty(samples, dtype=float)
    for i in range(samples):
        resample_means[i] = rng.choice(arr, size=len(arr), replace=True).mean()
    alpha = (1.0 - confidence) / 2.0
    lower = float(np.quantile(resample_means, alpha))
    upper = float(np.quantile(resample_means, 1.0 - alpha))
    return point_estimate, (lower, upper)
