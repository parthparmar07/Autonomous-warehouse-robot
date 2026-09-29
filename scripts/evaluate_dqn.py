from __future__ import annotations

from pathlib import Path

from src.deployment.policies import load_policy
from src.evaluation.evaluator import evaluate_policy
from src.evaluation.metrics import summarize_metrics


def main() -> None:
    checkpoint = Path("artifacts/dqn_policy.pt")
    if not checkpoint.exists():
        raise FileNotFoundError("Missing artifacts/dqn_policy.pt; run scripts/train_dqn.py or scripts/pretrain_bc.py first")

    results = []
    for num_agents in (5, 10, 20):
        policy = load_policy("dqn", num_agents)
        episode_results = evaluate_policy(policy, num_agents=num_agents, seeds=(100,), episodes=1)
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
