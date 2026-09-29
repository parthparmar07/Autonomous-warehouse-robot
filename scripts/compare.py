from __future__ import annotations

from src.evaluation.metrics import summarize_metrics
from src.evaluation.evaluator import evaluate_policy
from src.baselines.nearest_task import NearestTaskDispatcher
from src.environment.layout import default_layout
from src.environment.warehouse_env import WarehouseEnv


if __name__ == "__main__":
    grid_size = (20, 20)
    baseline = NearestTaskDispatcher(WarehouseEnv(grid_size=grid_size, num_agents=5, **default_layout(grid_size)))
    results = evaluate_policy(baseline, num_agents=5, seeds=(7,), episodes=2)
    summary = summarize_metrics(results)
    print({
        "completion_rate": summary.completion_rate,
        "average_delivery_time": summary.average_delivery_time,
        "collision_rate": summary.collision_rate,
        "throughput": summary.throughput,
    })
