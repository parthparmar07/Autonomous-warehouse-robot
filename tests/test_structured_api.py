from fastapi.testclient import TestClient

from src.deployment.api import app


def test_live_orders_use_fixed_bays_and_refill_with_consistent_state():
    client = TestClient(app)
    session = "structured-api-test"
    try:
        response = client.post(f"/reset?session_id={session}", json={
            "algorithm": "nearest_task", "num_agents": 1, "num_tasks": 1, "seed": 42,
        })
        assert response.status_code == 200
        initial = response.json()
        layout = initial["layout"]
        assert len(layout["pickup_bays"]) == 12
        assert len(layout["drop_bays"]) == 4
        original_id = initial["task_queue"][0]["task_id"]
        for _ in range(200):
            response = client.post(f"/tick?session_id={session}")
            assert response.status_code == 200
            state = response.json()
            assert state["layout"] == layout
            assert all(task["pickup"] in layout["pickup_bays"] and task["delivery"] in layout["drop_bays"]
                       for task in state["task_queue"])
            if state["completed_tasks"]:
                task = state["task_queue"][0]
                assert task["task_id"] != original_id
                assert state["robots"][0]["current_task"] == task["task_id"]
                assert state["robots"][0]["target"] == task["pickup"]
                assert task["creation_timestep"] == state["metrics"]["step_count"]
                assert state["done"] is False
                break
        else:
            raise AssertionError("Single robot failed to complete and replenish an order")
    finally:
        client.delete(f"/session?session_id={session}")


def test_api_rejects_empty_or_unbounded_order_batches():
    client = TestClient(app)
    for count in (0, -1, 100000):
        response = client.post("/reset", json={"num_tasks": count})
        assert response.status_code == 422
