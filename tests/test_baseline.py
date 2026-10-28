from src.baselines.nearest_task import NearestTaskDispatcher
from src.environment.warehouse_env import WarehouseEnv


def test_nearest_task_dispatcher_assigns_and_selects_action():
    env = WarehouseEnv(grid_size=(8, 8), num_agents=2, seed=9, obstacles=[], safety_corridors=[])
    env.reset(seed=9)
    env.task_queue = [
        {"task_id": 0, "pickup": (1, 1), "delivery": (6, 6), "creation_timestep": 0, "status": "pending"},
        {"task_id": 1, "pickup": (3, 3), "delivery": (6, 2), "creation_timestep": 0, "status": "pending"},
    ]

    baseline = NearestTaskDispatcher(env)
    baseline.assign_tasks(env.task_queue)

    assert env.robots[0]["current_task"] is not None
    assert env.robots[0]["target"] is not None
    action = baseline.choose_action(env.robots[0])
    assert 0 <= action <= 6
    assert env.get_action_mask(0)[action]
