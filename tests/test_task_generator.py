from src.environment.task_generator import generate_tasks
from src.environment.warehouse_env import WarehouseEnv


def test_task_generation_is_reproducible_and_counted():
    env = WarehouseEnv(grid_size=(8, 8), num_agents=2, seed=13)
    env.reset(seed=13)

    tasks_a = generate_tasks(env, num_tasks=6, seed=13)
    tasks_b = generate_tasks(env, num_tasks=6, seed=13)

    assert len(tasks_a) == len(tasks_b) == 6
    assert tasks_a == tasks_b
    for task in tasks_a:
        assert task["pickup"] is not None
        assert task["delivery"] is not None
        assert task["task_id"] >= 0
