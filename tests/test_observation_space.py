from src.environment.warehouse_env import WarehouseEnv


def test_local_observation_is_30_features_and_local_only():
    env = WarehouseEnv(grid_size=(8, 8), num_agents=2, seed=21)
    env.reset(seed=21)

    obs = env.get_local_observation(0)
    assert len(obs) == 30
    assert obs.shape == (30,)

    # The policy should only see a local window, target bearing, load and heading --
    # not the full grid.
    assert all(value == value for value in obs)
    assert obs[:25].sum() >= 0
    assert obs[25:27].shape == (2,)
    assert obs[27].shape == ()
    # Heading is sin/cos of the current orientation -- always on the unit circle.
    import numpy as np

    assert np.isclose(obs[28] ** 2 + obs[29] ** 2, 1.0)


def test_local_observation_heading_reflects_orientation():
    """The same world position/target must look different to the policy depending on
    which way the robot is facing -- otherwise FORWARD's meaning is ambiguous from the
    observation alone."""
    env = WarehouseEnv(grid_size=(8, 8), num_agents=1, seed=3)
    env.reset(seed=3)

    env.robots[0]["orientation"] = 0
    obs_north = env.get_local_observation(0)
    env.robots[0]["orientation"] = 2
    obs_south = env.get_local_observation(0)

    assert not (obs_north[28:30] == obs_south[28:30]).all()
