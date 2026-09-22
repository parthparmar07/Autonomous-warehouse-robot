from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MetricSummary:
    completion_rate: float = 0.0
    average_delivery_time: float = 0.0
    collision_rate: float = 0.0
    deadlocks: int = 0
    throughput: float = 0.0
    safety_violations: int = 0
    episode_reward: float = 0.0
    inference_latency_ms: float = 0.0
    safety_overrides: int = 0


def summarize_metrics(results: list[dict[str, Any]]) -> MetricSummary:
    """Aggregate a list of per-episode dictionaries into summary metrics."""
    if not results:
        return MetricSummary()

    completion_values = [float(item.get("completion_rate", 0.0)) for item in results]
    delivery_values = [float(item.get("delivery_time", 0.0)) for item in results]
    collisions = [float(item.get("collision_count", 0.0)) for item in results]
    rewards = [float(item.get("episode_reward", 0.0)) for item in results]

    return MetricSummary(
        completion_rate=sum(completion_values) / len(completion_values),
        average_delivery_time=sum(delivery_values) / len(delivery_values),
        collision_rate=sum(collisions) / len(collisions),
        deadlocks=int(sum(item.get("deadlocks", 0) for item in results)),
        throughput=sum(float(item.get("throughput", 0.0)) for item in results) / len(results),
        safety_violations=int(sum(item.get("safety_violations", 0) for item in results)),
        episode_reward=sum(rewards) / len(rewards),
        inference_latency_ms=float(sum(item.get("inference_latency_ms", 0.0) for item in results)) / len(results),
        safety_overrides=int(sum(item.get("safety_overrides", 0) for item in results)),
    )
