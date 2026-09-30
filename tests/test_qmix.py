import numpy as np

from src.algorithms.qmix.agent_network import QMIXAgentNetwork
from src.algorithms.qmix.mixer import QMIXMixer


def test_qmix_shapes_and_monotonic_condition_hold():
    agent = QMIXAgentNetwork(input_dim=30, action_dim=5)
    obs = np.random.randn(30).astype(np.float32)
    q_values, hidden = agent.forward(np.asarray(obs, dtype=np.float32))
    assert q_values.shape == (5,)

    mixer = QMIXMixer(num_agents=2, state_dim=16, hidden_dim=8)
    q1 = np.random.randn(5).astype(np.float32)
    q2 = np.random.randn(5).astype(np.float32)
    state = np.random.randn(16).astype(np.float32)
    result = mixer.monotonicity_check([q1, q2])
    assert result is True
