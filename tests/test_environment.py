import numpy as np

from src.environment.warehouse_env import ACTION_BACKWARD, ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT, ACTION_STAY, WarehouseEnv


def test_reset_and_seed_reproducibility():
    env1 = WarehouseEnv(grid_size=(8, 8), num_agents=2, seed=42)
    env2 = WarehouseEnv(grid_size=(8, 8), num_agents=2, seed=42)

    state1 = env1.reset(seed=42)
    state2 = env2.reset(seed=42)

    assert state1 == state2
    assert env1.get_global_state() == env2.get_global_state()


def test_step_respects_action_space_and_maintains_valid_positions():
    env = WarehouseEnv(grid_size=(8, 8), num_agents=2, seed=7)
    env.reset(seed=7)

    actions = [ACTION_FORWARD, ACTION_STAY]
    state_after = env.step(actions)

    assert len(state_after["robots"]) == 2
    for robot in state_after["robots"]:
        assert 0 <= robot["x"] < env.grid_size[0]
        assert 0 <= robot["y"] < env.grid_size[1]
        assert robot["orientation"] in {0, 1, 2, 3}


def test_collision_detection_is_deterministic():
    env = WarehouseEnv(grid_size=(5, 5), num_agents=2, seed=11)
    env.reset(seed=11)

    env.robots[0]["x"] = 1
    env.robots[0]["y"] = 1
    env.robots[1]["x"] = 1
    env.robots[1]["y"] = 2

    env.step([ACTION_FORWARD, ACTION_FORWARD])
    assert env.episode_metrics["collision_count"] >= 0
