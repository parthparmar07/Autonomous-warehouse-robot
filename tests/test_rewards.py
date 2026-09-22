from src.environment.rewards import compute_reward


def test_compute_reward_uses_project_defaults():
    delivery = compute_reward("delivery")
    collision = compute_reward("collision")
    step = compute_reward("step")

    assert delivery == 10
    assert collision == -5
    assert step == -0.1
