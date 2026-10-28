import json
from collections import deque

import pytest

from src.baselines.nearest_task import NearestTaskDispatcher
from src.environment.task_generator import generate_tasks
from src.environment.warehouse_env import WarehouseEnv
from src.evaluation.manifests import environment_from_manifest, generate_evaluation_manifest
from src.utils.hashing import sha256_digest


@pytest.mark.parametrize("size", [(7, 7), (8, 8), (20, 20), (24, 16), (16, 24)])
def test_all_stations_reachable_and_orders_stay_at_stations(size):
    env = WarehouseEnv(grid_size=size, num_agents=1)
    env.reset()
    assert env.grid.shape == (size[1], size[0])
    start = (env.robots[0]["x"], env.robots[0]["y"])
    seen, queue = {start}, deque([start])
    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if (0 <= nx < size[0] and 0 <= ny < size[1] and env.grid[ny, nx] not in (1, 2)
                    and (nx, ny) not in seen):
                seen.add((nx, ny))
                queue.append((nx, ny))
    assert set(env.pickup_bays + env.drop_bays) <= seen
    for task in generate_tasks(env, 100, seed=4):
        assert task["pickup"] in env.pickup_bays
        assert task["delivery"] in env.drop_bays
        assert task["shelf"] in env.obstacles
        assert env.grid[task["pickup"][1], task["pickup"][0]] == 3
        assert env.grid[task["delivery"][1], task["delivery"][0]] == 4
        assert task["pickup"] != task["delivery"]


def test_staging_is_unique_and_clear_of_service_bays():
    env = WarehouseEnv(num_agents=40)
    env.reset()
    positions = {(r["x"], r["y"]) for r in env.robots}
    assert len(positions) == 40
    assert positions <= set(env.staging_cells)
    assert all(env.grid[y, x] == 0 for x, y in positions)


def test_single_robot_completes_shelf_to_drop_orders_without_entering_racks():
    env = WarehouseEnv(num_agents=1)
    env.reset()
    env.task_queue = generate_tasks(env, 6, seed=19)
    dispatcher = NearestTaskDispatcher(env)
    for _ in range(1000):
        dispatcher.assign_pending_tasks(env.task_queue)
        previous = env.get_global_state()
        result = env.step(dispatcher.act(env))
        robot = result["robots"][0]
        position = (robot["x"], robot["y"])
        assert env.grid[position[1], position[0]] not in (1, 2)
        if previous["robots"][0]["load_status"] == 0 and robot["load_status"] == 1:
            assert position in env.pickup_bays
        if result["completed_tasks"] > previous["completed_tasks"]:
            assert position in env.drop_bays
        if result["done"]:
            break
    assert env.completed_tasks == 6
    assert env.robots[0]["load_status"] == 0


def test_new_batch_uses_same_stations_and_fresh_order_ids():
    env = WarehouseEnv(num_agents=1)
    env.reset()
    first = generate_tasks(env, 10, seed=1)
    env.episode_metrics["step_count"] = 100
    second = generate_tasks(env, 10, seed=2, start_task_id=10)
    assert not {t["task_id"] for t in first} & {t["task_id"] for t in second}
    assert all(t["creation_timestep"] == 100 for t in second)
    assert all(t["pickup"] in env.pickup_bays and t["delivery"] in env.drop_bays for t in second)


def test_manifest_round_trip_preserves_geometry_and_rejects_legacy():
    manifest = json.loads(json.dumps(generate_evaluation_manifest(5, 101)))
    env = environment_from_manifest(manifest)
    assert env.get_layout_spec() == manifest["layout"]
    assert json.loads(json.dumps(env.task_queue)) == manifest["tasks"]
    del manifest["layout"]
    manifest["manifest_hash"] = ""
    manifest["manifest_hash"] = sha256_digest(manifest)
    with pytest.raises(ValueError, match="Legacy"):
        environment_from_manifest(manifest)


def test_missing_bays_fail_instead_of_generating_random_floor_orders():
    env = WarehouseEnv(grid_size=(5, 5))
    with pytest.raises(ValueError, match="requires"):
        generate_tasks(env, 1)
