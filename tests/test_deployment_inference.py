import numpy as np

from src.deployment.inference import predict_actions


def test_inference_returns_valid_masked_actions():
    observations = [np.zeros(30, dtype=np.float32), np.ones(30, dtype=np.float32)]
    actions = predict_actions(observations, [[True, False, False, False, False], [False, True, True, True, True]])
    assert actions[0] == 0
    assert actions[1] in {1, 2, 3, 4}