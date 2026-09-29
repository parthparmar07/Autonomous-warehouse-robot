from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.algorithms.dqn.trainer import DQNTrainer
from src.algorithms.iql.agent import IndependentQLearning
from src.algorithms.qmix.trainer import QMIXTrainer
from src.baselines.nearest_task import NearestTaskDispatcher
from src.environment.layout import default_layout
from src.environment.warehouse_env import WarehouseEnv

ALGORITHM_DESCRIPTIONS = {
    "nearest_task": "Rule-based greedy dispatcher with BFS pathfinding around shelves/corridors. Not a learned policy.",
    "dqn": "Single shared network, parameter sharing across the fleet.",
    "iql": "Independent Q-learning: one DQN per robot, no coordination signal.",
    "qmix": "Centralized-training/decentralized-execution value decomposition across the fleet.",
}

_CHECKPOINTS = {
    "dqn": Path("artifacts/dqn_policy.pt"),
    "iql": Path("artifacts/iql_policy.pt"),
    "qmix": Path("artifacts/qmix_policy.pt"),
}


def checkpoint_exists(name: str) -> bool:
    path = _CHECKPOINTS.get(name)
    return path is not None and path.exists()


def load_policy(name: str, num_agents: int) -> Any:
    """Build (and load a checkpoint into, if one exists) the named policy.

    Every learned policy is returned in deterministic (epsilon=0) mode: exploration noise
    belongs to training, never to evaluation or a live demo (ADR-003's train/execute
    consistency argument applies just as much to epsilon as it does to the action mask).
    """
    if name == "nearest_task":
        grid_size = (20, 20)
        return NearestTaskDispatcher(WarehouseEnv(grid_size=grid_size, num_agents=num_agents, **default_layout(grid_size)))

    if name == "iql":
        policy = IndependentQLearning(num_agents=num_agents)
        checkpoint = _CHECKPOINTS["iql"]
        if checkpoint.exists():
            state_dicts = torch.load(checkpoint, map_location="cpu", weights_only=True)
            # Each agent's network is architecturally independent of fleet size -- only
            # the *count* of robots differs, not what a single robot's policy looks like
            # -- so a checkpoint trained at one N applies just as well at another by
            # cycling through the trained networks, rather than only ever loading when
            # the fleet size matches exactly and silently evaluating random weights
            # otherwise.
            if state_dicts:
                for idx, agent in enumerate(policy.agents):
                    agent.online_network.load_state_dict(state_dicts[idx % len(state_dicts)])
        for agent in policy.agents:
            agent.epsilon = 0.0
        return policy

    if name == "qmix":
        policy = QMIXTrainer(num_agents=num_agents)
        checkpoint = _CHECKPOINTS["qmix"]
        if checkpoint.exists():
            artifact = torch.load(checkpoint, map_location=policy.device, weights_only=True)
            agent_state_dicts = artifact["agent_networks"]
            if agent_state_dicts:
                for idx, network in enumerate(policy.agent_networks):
                    network.load_state_dict(agent_state_dicts[idx % len(agent_state_dicts)])
            # Unlike the agent networks, the mixer's hypernetwork output size is a
            # function of num_agents -- it genuinely is fleet-size-specific (ADR/README
            # already note this) -- so it only loads on an exact match. That's fine for
            # inference/evaluation: act() never touches the mixer, only train_step does.
            if len(agent_state_dicts) == num_agents:
                policy.mixer.load_state_dict(artifact["mixer"])
        policy.epsilon = 0.0
        return policy

    if name == "dqn":
        trainer = DQNTrainer()
        checkpoint = _CHECKPOINTS["dqn"]
        if checkpoint.exists():
            state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
            trainer.agent.online_network.load_state_dict(state_dict)
        trainer.agent.epsilon = 0.0
        return trainer

    raise ValueError(f"Unknown algorithm: {name}")


def describe_algorithms() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "trained": True if name == "nearest_task" else checkpoint_exists(name),
            "description": ALGORITHM_DESCRIPTIONS[name],
        }
        for name in ("nearest_task", "dqn", "iql", "qmix")
    ]
