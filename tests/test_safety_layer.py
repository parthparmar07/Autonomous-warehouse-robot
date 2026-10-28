import numpy as np

from src.environment.layout import default_layout
from src.environment.warehouse_env import ACTION_BACKWARD, ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT, ACTION_STAY, WarehouseEnv
from src.safety.action_mask import apply_action_mask, compute_passable_mask, get_action_mask


def test_safety_mask_discards_unsafe_actions_even_when_q_values_are_highest():
    q_values = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=float)
    mask = [True, True, True, False, True]

    masked = apply_action_mask(q_values, mask)
    assert np.isfinite(masked[0])
    assert np.isfinite(masked[1])
    assert np.isfinite(masked[2])
    assert np.isneginf(masked[3])
    assert np.isfinite(masked[4])

    safe_action = int(np.argmax(masked))
    assert mask[safe_action] is True

    # mask function should also behave as expected.
    action_mask = get_action_mask(np.array([0, 0]), None)
    assert len(action_mask) == 5


def test_get_action_mask_without_a_warehouse_allows_everything():
    assert get_action_mask([3, 3, 1], None) == [True, True, True, True, True]


def test_stay_left_right_are_always_legal_from_any_cell():
    # An explicit env invariant (ADR-003): a robot must never be left with zero legal
    # actions, so a no-op must always be available even when boxed in on every side.
    grid = np.zeros((3, 3), dtype=np.int8)
    grid[:, :] = 1  # every cell a shelf
    mask = compute_passable_mask(x=1, y=1, orientation=1, grid=grid)
    assert mask[ACTION_STAY] is True
    assert mask[ACTION_LEFT] is True
    assert mask[ACTION_RIGHT] is True
    assert mask[ACTION_FORWARD] is False
    assert mask[ACTION_BACKWARD] is False


def test_env_action_mask_blocks_corridor_entry():
    grid_size = (20, 20)
    env = WarehouseEnv(grid_size=grid_size, num_agents=1, seed=1, **default_layout(grid_size))
    env.reset(seed=1)
    env.robots[0]["x"], env.robots[0]["y"], env.robots[0]["orientation"] = 1, 1, 0  # facing the top corridor row
    mask = env.get_action_mask(0)
    assert mask[ACTION_FORWARD] is False
    assert mask[ACTION_STAY] is True


def test_env_action_mask_blocks_shelf_entry():
    env = WarehouseEnv(grid_size=(5, 5), num_agents=1, seed=1, obstacles=[(2, 1)])
    env.reset(seed=1)
    env.robots[0]["x"], env.robots[0]["y"], env.robots[0]["orientation"] = 1, 1, 1  # facing the shelf
    mask = env.get_action_mask(0)
    assert mask[ACTION_FORWARD] is False


def test_joint_resolver_blocks_stationary_robot_without_mask_leaking_occupancy():
    env = WarehouseEnv(grid_size=(5, 5), num_agents=2, seed=1)
    env.reset(seed=1)
    env.robots[0]["x"], env.robots[0]["y"], env.robots[0]["orientation"] = 1, 1, 1  # facing east
    env.robots[1]["x"], env.robots[1]["y"] = 2, 1
    mask = env.get_action_mask(0)
    assert mask[ACTION_FORWARD] is True
    env.step([ACTION_FORWARD,ACTION_STAY])
    assert (env.robots[0]['x'],env.robots[0]['y'])==(1,1)
    assert env.episode_metrics['collision_count']==1


def test_env_action_mask_out_of_bounds_is_masked():
    env = WarehouseEnv(grid_size=(5, 5), num_agents=1, seed=1)
    env.reset(seed=1)
    env.robots[0]["x"], env.robots[0]["y"], env.robots[0]["orientation"] = 0, 0, 0  # facing off the top edge
    mask = env.get_action_mask(0)
    assert mask[ACTION_FORWARD] is False


def test_step_physically_refuses_corridor_entry_even_if_commanded():
    # Layer 3 (ADR-003): the environment itself is the redundant, non-ML backstop --
    # even a policy that ignores the mask cannot actually enter a corridor cell, and the
    # attempt is logged as a safety override.
    grid_size = (20, 20)
    env = WarehouseEnv(grid_size=grid_size, num_agents=1, seed=1, **default_layout(grid_size))
    env.reset(seed=1)
    env.robots[0]["x"], env.robots[0]["y"], env.robots[0]["orientation"] = 1, 1, 0
    state = env.step([ACTION_FORWARD])
    robot = state["robots"][0]
    assert (robot["x"], robot["y"]) == (1, 1)
    assert state["metrics"]["safety_overrides"] == 1.0


def test_adversarial_policy_never_achieves_a_corridor_intrusion():
    # Mirrors the project's CI safety gate: an adversarial policy that always tries to
    # drive into the corridor must record zero actual intrusions across many episodes.
    grid_size = (20, 20)
    for seed in range(20):
        env = WarehouseEnv(grid_size=grid_size, num_agents=3, seed=seed, **default_layout(grid_size))
        env.reset(seed=seed)
        for _ in range(30):
            actions = [ACTION_FORWARD if robot["orientation"] in (0, 1) else ACTION_BACKWARD for robot in env.robots]
            env.step(actions)
            for robot in env.robots:
                assert env.grid[robot["y"], robot["x"]] != 2  # never actually inside a corridor cell
