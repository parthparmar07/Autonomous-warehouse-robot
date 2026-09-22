from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from src.algorithms.qmix.agent_network import QMIXAgentNetwork
from src.algorithms.qmix.mixer import QMIXMixer
from src.algorithms.qmix.replay_buffer import QMIXReplayBuffer


class QMIXTrainer:
    """Minimal QMIX trainer scaffold for centralized training with local execution."""

    def __init__(self, num_agents: int = 5, input_dim: int = 28, action_dim: int = 5, replay_capacity: int = 100_000, batch_size: int = 64) -> None:
        self.num_agents = int(num_agents)
        self.agent_networks = [QMIXAgentNetwork(input_dim=input_dim, action_dim=action_dim) for _ in range(num_agents)]
        self.target_agent_networks = [QMIXAgentNetwork(input_dim=input_dim, action_dim=action_dim) for _ in range(num_agents)]
        self.mixer = QMIXMixer(num_agents=num_agents, state_dim=64, hidden_dim=32)
        self.target_mixer = QMIXMixer(num_agents=num_agents, state_dim=64, hidden_dim=32)
        for target, source in zip(self.target_agent_networks, self.agent_networks):
            target.load_state_dict(source.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for network in self.agent_networks + self.target_agent_networks:
            network.to(self.device)
        self.mixer.to(self.device)
        self.target_mixer.to(self.device)
        parameters = list(self.mixer.parameters())
        for network in self.agent_networks:
            parameters.extend(network.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=1e-3)
        self.buffer = QMIXReplayBuffer(capacity=replay_capacity, batch_size=batch_size)

    def act(self, observations: List[np.ndarray], action_masks: List[List[bool] | None] | None = None) -> List[int]:
        action_masks = action_masks or [None for _ in observations]
        return [
            self.agent_networks[idx].act(obs, action_masks[idx])
            for idx, obs in enumerate(observations)
        ]

    def add_transition(self, state: np.ndarray, actions: List[int], reward: float, next_state: np.ndarray, done: bool, action_masks: List[List[bool]] | None = None, next_action_masks: List[List[bool]] | None = None, observations: List[np.ndarray] | None = None, next_observations: List[np.ndarray] | None = None) -> None:
        self.buffer.add(state, actions, reward, next_state, done, action_masks, next_action_masks, observations, next_observations)

    def train_step(self) -> float:
        sample = self.buffer.sample()
        if not sample or sample[0].get("observations") is None:
            return 0.0
        states = torch.as_tensor(np.stack([item["state"] for item in sample]), dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([item["next_state"] for item in sample]), dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor([item["reward"] for item in sample], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([float(item["done"]) for item in sample], dtype=torch.float32, device=self.device)
        chosen_values = []
        next_values = []
        for agent_id in range(self.num_agents):
            observations = torch.as_tensor(np.stack([item["observations"][agent_id] for item in sample]), dtype=torch.float32, device=self.device)
            next_observations = torch.as_tensor(np.stack([item["next_observations"][agent_id] for item in sample]), dtype=torch.float32, device=self.device)
            q_values, _ = self.agent_networks[agent_id].forward(observations)
            if q_values.dim() == 1:
                q_values = q_values.unsqueeze(0)
            chosen_values.append(q_values.gather(1, torch.as_tensor([item["actions"][agent_id] for item in sample], device=self.device).unsqueeze(1)).squeeze(1))
            with torch.no_grad():
                target_q_values = self.target_agent_networks[agent_id].forward(next_observations)[0]
                if target_q_values.dim() == 1:
                    target_q_values = target_q_values.unsqueeze(0)
                next_values.append(target_q_values.max(dim=1).values)
        joint_values = self.mixer(chosen_values, states)
        with torch.no_grad():
            target_values = self.target_mixer(next_values, next_states)
            target = rewards + 0.99 * (1.0 - dones) * target_values
        loss = F.mse_loss(joint_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], 10.0)
        self.optimizer.step()
        return float(loss.item())

    def update_targets(self) -> None:
        for target, source in zip(self.target_agent_networks, self.agent_networks):
            target.load_state_dict(source.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
