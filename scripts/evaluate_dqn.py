from __future__ import annotations

from pathlib import Path

import torch

from src.algorithms.dqn.trainer import DQNTrainer
from src.evaluation.evaluator import evaluate_policy
from src.evaluation.metrics import summarize_metrics


def main() -> None:
    checkpoint = Path("artifacts/dqn_policy.pt")
    if not checkpoint.exists():
        raise FileNotFoundError("Missing artifacts/dqn_policy.pt; run scripts/train_dqn.py first")

    results = []
    for num_agents in (5, 10, 20):
        trainer = DQNTrainer()
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        trainer.agent.online_network.load_state_dict(state_dict)
        trainer.agent.epsilon = 0.0
        episode_results = evaluate_policy(trainer, num_agents=num_agents, seeds=(100,), episodes=1)
        summary = summarize_metrics(episode_results)
        results.append({
            "num_agents": num_agents,
            "completion_rate": summary.completion_rate,
            "episode_reward": summary.episode_reward,
            "collision_rate": summary.collision_rate,
            "throughput": summary.throughput,
        })

    for result in results:
        print(result)


if __name__ == "__main__":
    main()
