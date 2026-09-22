from src.evaluation.bootstrap import bootstrap_mean
from src.evaluation.evaluator import evaluate_policy
from src.evaluation.manifests import generate_manifest_set
from src.baselines.nearest_task import NearestTaskDispatcher
from src.environment.warehouse_env import WarehouseEnv


def test_evaluation_pipeline_generates_manifests_and_metrics():
    manifest_paths = generate_manifest_set("evaluation/manifests")
    assert len(manifest_paths) >= 3

    results = evaluate_policy(None, num_agents=2, seeds=(7,), episodes=1)
    assert len(results) == 1
    assert results[0]["completion_rate"] >= 0.0

    baseline = NearestTaskDispatcher(WarehouseEnv(grid_size=(20, 20), num_agents=2))
    baseline_results = evaluate_policy(baseline, num_agents=2, seeds=(7,), episodes=1)
    assert baseline_results[0]["completion_rate"] > 0.0

    mean_value, std_value = bootstrap_mean([1.0, 2.0, 3.0, 4.0])
    assert mean_value > 0.0
    assert std_value >= 0.0
