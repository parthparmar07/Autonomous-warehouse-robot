import numpy as np

from src.algorithms.dqn.agent import DQNAgent
from src.algorithms.dqn.network import DQNNetwork
from src.algorithms.dqn.replay_buffer import ReplayBuffer


def test_dqn_output_shape_and_replay_buffer_basic_behavior():
    network = DQNNetwork(input_dim=30, action_dim=5)
    obs = np.random.randn(30).astype(np.float32)
    q_values = network(obs)
    assert q_values.shape == (5,)

    buffer = ReplayBuffer(capacity=128, batch_size=4)
    buffer.add(np.zeros(30, dtype=np.float32), 1, 1.0, np.ones(30, dtype=np.float32), False)
    assert len(buffer) == 1

    agent = DQNAgent(input_dim=30, action_dim=5, learning_rate=1e-3)
    action = agent.act(obs, [True, True, True, True, True])
    assert 0 <= action <= 4
