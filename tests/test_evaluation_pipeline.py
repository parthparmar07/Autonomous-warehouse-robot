from src.evaluation.evaluator import evaluate_policy
from src.evaluation.manifests import generate_manifest_set
from src.baselines.nearest_task import NearestTaskDispatcher
from src.environment.warehouse_env import WarehouseEnv


def test_evaluation_pipeline_generates_manifests_and_metrics(tmp_path):
    manifest_paths = generate_manifest_set(tmp_path)
    assert len(manifest_paths) >= 3

    import pytest
    with pytest.raises(ValueError,match='Explicit policy'):
        evaluate_policy(None, num_agents=2, seeds=(7,), episodes=1)

    baseline = NearestTaskDispatcher(WarehouseEnv(grid_size=(20, 20), num_agents=2))
    baseline_results = evaluate_policy(baseline, num_agents=2, seeds=(7,), episodes=1)
    assert baseline_results[0]["completion_rate"] > 0.0

