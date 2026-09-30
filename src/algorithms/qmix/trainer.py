from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.algorithms.qmix.agent_network import QMIXAgentNetwork
from src.algorithms.qmix.mixer import QMIXMixer
from src.algorithms.qmix.replay_buffer import QMIXReplayBuffer


class QMIXTrainer:
    """Minimal QMIX trainer scaffold for centralized training with local execution."""

    def __init__(
        self,
        num_agents: int = 5,
        input_dim: int = 30,
        action_dim: int = 5,
        replay_capacity: int = 100_000,
        batch_size: int = 64,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
    ) -> None:
        self.num_agents = int(num_agents)
        self.action_dim = int(action_dim)
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
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
        # Per-agent GRU hidden state, carried across successive act() calls within an
        # episode. None means "episode just started" (the network zero-initialises it).
        # Without this, every act() call fed the network a from-scratch zero hidden
        # state and the GRU's recurrent memory did nothing at rollout time.
        self.hidden_states: List[Optional[torch.Tensor]] = [None] * self.num_agents

    def reset_hidden_states(self) -> None:
        """Call at the start of every episode -- the recurrent memory must not leak
        across episode boundaries."""
        self.hidden_states = [None] * self.num_agents

    def act(self, observations: List[np.ndarray], action_masks: List[List[bool] | None] | None = None) -> List[int]:
        """Epsilon-greedy over the masked action set, advancing each agent's GRU hidden
        state from its own observation regardless of whether the action taken was
        greedy or random -- the memory summarises what was *observed*, not which action
        was picked. Without exploration noise here, each agent's network would
        deterministically repeat whatever its (initially near-random) weights pick, and
        QMIX would never see enough of the state space to learn from -- CTDE still
        applies at training time via the mixer in train_step."""
        action_masks = action_masks or [None for _ in observations]
        actions: List[int] = []
        for idx, obs in enumerate(observations):
            mask = action_masks[idx]
            network = self.agent_networks[idx]
            if np.random.rand() < self.epsilon:
                valid_actions = list(range(self.action_dim)) if mask is None else [a for a, allowed in enumerate(mask) if allowed]
                action = int(np.random.choice(valid_actions)) if valid_actions else 0
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    _, next_hidden = network.forward(obs_tensor, self.hidden_states[idx])
            else:
                action, next_hidden = network.act(obs, mask, self.hidden_states[idx])
            self.hidden_states[idx] = next_hidden
            actions.append(action)
        return actions

    def add_transition(
        self,
        state: np.ndarray,
        actions: List[int],
        reward: float,
        next_state: np.ndarray,
        done: bool,
        action_masks: List[List[bool]] | None = None,
        next_action_masks: List[List[bool]] | None = None,
        observations: List[np.ndarray] | None = None,
        next_observations: List[np.ndarray] | None = None,
        hidden_ins: List[Optional[torch.Tensor]] | None = None,
        next_hidden_ins: List[Optional[torch.Tensor]] | None = None,
    ) -> None:
        to_numpy = lambda hiddens: None if hiddens is None else [None if h is None else h.detach().cpu().numpy() for h in hiddens]
        self.buffer.add(
            state, actions, reward, next_state, done, action_masks, next_action_masks,
            observations, next_observations, to_numpy(hidden_ins), to_numpy(next_hidden_ins),
        )

    def _stack_hidden(self, hidden_list: List, gru_hidden_dim: int) -> torch.Tensor:
        """Batch a list of per-sample hidden states (each a (1,1,H) array, or None at an
        episode's first step) into the (1, batch, H) tensor forward() expects. A None
        entry -- the recurrent equivalent of a zero-initialised hidden state -- becomes
        an all-zero row rather than silently reusing whatever the batch happens to hold."""
        rows = [np.zeros(gru_hidden_dim, dtype=np.float32) if h is None else np.asarray(h, dtype=np.float32).reshape(-1) for h in hidden_list]
        return torch.as_tensor(np.stack(rows), dtype=torch.float32, device=self.device).unsqueeze(0)

    def train_step(self) -> float:
        sample = self.buffer.sample()
        if not sample or sample[0].get("observations") is None:
            return 0.0
        states = torch.as_tensor(np.stack([item["state"] for item in sample]), dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([item["next_state"] for item in sample]), dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor([item["reward"] for item in sample], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([float(item["done"]) for item in sample], dtype=torch.float32, device=self.device)
        has_hidden = sample[0].get("hidden_ins") is not None
        chosen_values = []
        next_values = []
        for agent_id in range(self.num_agents):
            network = self.agent_networks[agent_id]
            observations = torch.as_tensor(np.stack([item["observations"][agent_id] for item in sample]), dtype=torch.float32, device=self.device)
            next_observations = torch.as_tensor(np.stack([item["next_observations"][agent_id] for item in sample]), dtype=torch.float32, device=self.device)
            if has_hidden:
                hidden_in = self._stack_hidden([item["hidden_ins"][agent_id] for item in sample], network.hidden_dim)
                next_hidden_in = self._stack_hidden([item["next_hidden_ins"][agent_id] for item in sample], network.hidden_dim)
            else:
                hidden_in = next_hidden_in = None
            q_values, _ = network.forward(observations, hidden_in)
            if q_values.dim() == 1:
                q_values = q_values.unsqueeze(0)
            chosen_values.append(q_values.gather(1, torch.as_tensor([item["actions"][agent_id] for item in sample], device=self.device).unsqueeze(1)).squeeze(1))
            with torch.no_grad():
                target_q_values = self.target_agent_networks[agent_id].forward(next_observations, next_hidden_in)[0]
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
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return float(loss.item())

    def update_targets(self) -> None:
        for target, source in zip(self.target_agent_networks, self.agent_networks):
            target.load_state_dict(source.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
