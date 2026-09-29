import numpy as np

from src.algorithms.iql.agent import IndependentQLearning


def test_iql_runs_multiple_agents_and_respects_local_observations():
    iql = IndependentQLearning(num_agents=3, input_dim=30, action_dim=5)
    obs = [np.random.randn(30).astype(np.float32) for _ in range(3)]
    actions = iql.act(obs, [None, None, None])
    assert len(actions) == 3
    assert all(0 <= action <= 4 for action in actions)

    sample = iql.sample_transition(obs[0], 0, 1.0, obs[1], False)
    assert sample["action"] == 0
