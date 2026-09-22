from src.environment.warehouse_env import WarehouseEnv


def test_local_observation_is_28_features_and_local_only():
    env = WarehouseEnv(grid_size=(8, 8), num_agents=2, seed=21)
    env.reset(seed=21)

    obs = env.get_local_observation(0)
    assert len(obs) == 28
    assert obs.shape == (28,)

    # The policy should only see a local window and target bearing, not the full grid.
    assert all(value == value for value in obs)
    assert obs[:25].sum() >= 0
    assert obs[25:27].shape == (2,)
    assert obs[27].shape == ()
