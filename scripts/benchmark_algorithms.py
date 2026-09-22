from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.algorithms.dqn.trainer import DQNTrainer
from src.algorithms.iql.agent import IndependentQLearning
from src.algorithms.qmix.trainer import QMIXTrainer
from src.baselines.nearest_task import NearestTaskDispatcher
from src.environment.warehouse_env import WarehouseEnv
from src.evaluation.evaluator import evaluate_policy
from src.evaluation.metrics import summarize_metrics


def make_policy(name: str, num_agents: int) -> Any:
    if name == "nearest_task":
        return NearestTaskDispatcher(WarehouseEnv(grid_size=(20, 20), num_agents=num_agents))
    if name == "iql":
        policy = IndependentQLearning(num_agents=num_agents)
        checkpoint = Path("artifacts/iql_policy.pt")
        if checkpoint.exists():
            state_dicts = torch.load(checkpoint, map_location="cpu", weights_only=True)
            for agent, state_dict in zip(policy.agents, state_dicts):
                agent.online_network.load_state_dict(state_dict)
                agent.epsilon = 0.0
        return policy
    if name == "qmix":
        policy = QMIXTrainer(num_agents=num_agents)
        checkpoint = Path("artifacts/qmix_policy.pt")
        if checkpoint.exists():
            artifact = torch.load(checkpoint, map_location=policy.device, weights_only=True)
            checkpoint_agents = len(artifact["agent_networks"])
            if checkpoint_agents == num_agents:
                for network, state_dict in zip(policy.agent_networks, artifact["agent_networks"]):
                    network.load_state_dict(state_dict)
                policy.mixer.load_state_dict(artifact["mixer"])
        return policy
    if name == "dqn":
        trainer = DQNTrainer()
        checkpoint = Path("artifacts/dqn_policy.pt")
        if checkpoint.exists():
            state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
            trainer.agent.online_network.load_state_dict(state_dict)
            trainer.agent.epsilon = 0.0
        return trainer
    raise ValueError(f"Unknown algorithm: {name}")


def main() -> None:
    records: list[dict[str, Any]] = []
    for algorithm in ("nearest_task", "iql", "qmix", "dqn"):
        for num_agents in (5, 10, 20):
            policy = make_policy(algorithm, num_agents)
            results = evaluate_policy(policy, num_agents=num_agents, seeds=(100,), episodes=1)
            summary = summarize_metrics(results)
            records.append({
                "algorithm": algorithm,
                "num_agents": num_agents,
                "completion_rate": summary.completion_rate,
                "average_delivery_time": summary.average_delivery_time,
                "collision_rate": summary.collision_rate,
                "throughput": summary.throughput,
                "episode_reward": summary.episode_reward,
            })

    output = Path("evaluation/benchmark_results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    for record in records:
        print(record)
    print(f"Saved {len(records)} records to {output}")


if __name__ == "__main__":
    main()
