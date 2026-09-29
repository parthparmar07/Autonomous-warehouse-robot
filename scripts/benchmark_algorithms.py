from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.deployment.policies import load_policy
from src.evaluation.evaluator import evaluate_policy
from src.evaluation.metrics import summarize_metrics


def main() -> None:
    # 5 seeds x 2 episodes rather than one smoke episode: with only 20 tasks per
    # episode, a single episode makes completion_rate noisy in steps of 5 percentage
    # points and prone to misreading run-to-run variance as a real change. This is
    # still not the sealed-manifest evaluation the project's own integrity design calls
    # for (see "Evaluation integrity" below and Remaining Work) -- it's a steadier smoke
    # comparison, not a final result.
    seeds = (100, 101, 102, 103, 104)
    episodes_per_seed = 2
    records: list[dict[str, Any]] = []
    for algorithm in ("nearest_task", "iql", "qmix", "dqn"):
        for num_agents in (5, 10, 20):
            policy = load_policy(algorithm, num_agents)
            results = evaluate_policy(policy, num_agents=num_agents, seeds=seeds, episodes=episodes_per_seed)
            summary = summarize_metrics(results)
            records.append({
                "algorithm": algorithm,
                "num_agents": num_agents,
                "completion_rate": summary.completion_rate,
                "average_delivery_time": summary.average_delivery_time,
                "collision_rate": summary.collision_rate,
                "throughput": summary.throughput,
                "episode_reward": summary.episode_reward,
                "safety_overrides": summary.safety_overrides,
            })

    output = Path("evaluation/benchmark_results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    for record in records:
        print(record)
    print(f"Saved {len(records)} records to {output}")


if __name__ == "__main__":
    main()
