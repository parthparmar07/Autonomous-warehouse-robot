from src.evaluation.metrics import MetricSummary, summarize_metrics
from src.tracking.mlflow_logger import MLFlowLogger
from src.visualization.warehouse_renderer import WarehouseRenderer
from src.environment.warehouse_env import WarehouseEnv


def test_metrics_summary_and_tracking_are_usable():
    results = [
        {"completion_rate": 0.8, "delivery_time": 3.5, "collision_count": 1.0, "episode_reward": 10.0, "deadlocks": 0, "throughput": 2.0, "safety_violations": 0, "inference_latency_ms": 20.0, "safety_overrides": 1},
        {"completion_rate": 0.9, "delivery_time": 4.0, "collision_count": 2.0, "episode_reward": 12.0, "deadlocks": 1, "throughput": 2.5, "safety_violations": 0, "inference_latency_ms": 22.0, "safety_overrides": 2},
    ]
    summary = summarize_metrics(results)
    assert isinstance(summary, MetricSummary)
    assert summary.completion_rate > 0.0

    logger = MLFlowLogger(experiment_name="test-experiment")
    logger.start_run("test-run")
    logger.log_metrics({"completion_rate": summary.completion_rate})
    logger.end_run()

    env = WarehouseEnv(grid_size=(6, 6), num_agents=2, seed=5)
    env.reset(seed=5)
    renderer = WarehouseRenderer(env)
    assert "No warehouse grid available" not in renderer.render()
