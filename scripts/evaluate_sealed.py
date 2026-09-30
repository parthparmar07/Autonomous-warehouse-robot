from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.baselines.nearest_task import NearestTaskDispatcher
from src.deployment.policies import load_policy
from src.environment.layout import default_layout
from src.environment.warehouse_env import WarehouseEnv
from src.evaluation.bootstrap import bootstrap_confidence_interval
from src.evaluation.manifests import verify_manifest_hash

MANIFEST_DIR = Path("evaluation/manifests")
ALGORITHMS = ("nearest_task", "dqn", "iql", "qmix")
MAX_STEPS = 200


def load_and_verify_manifest(path: Path) -> Dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not verify_manifest_hash(manifest):
        raise ValueError(f"Manifest hash mismatch for {path} -- this sealed manifest has been altered since it was generated, refusing to evaluate against it")
    return manifest


def run_manifest(policy_name: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Replay one sealed manifest exactly: same floor plan, same seed, same tasks."""
    num_agents = manifest["num_agents"]
    grid_size = tuple(manifest["grid_size"])
    env = WarehouseEnv(grid_size=grid_size, num_agents=num_agents, seed=manifest["seed"], **default_layout(grid_size))
    env.reset(seed=manifest["seed"])

    # Defensive integrity check: if a fresh reset no longer reproduces the sealed
    # robot_initial_state, the environment/layout has drifted since this manifest was
    # sealed, and the manifest is no longer a faithful replay -- fail loudly rather than
    # silently evaluating against conditions that don't match what was sealed.
    for robot, sealed in zip(env.robots, manifest["robot_initial_state"]):
        actual = (robot["x"], robot["y"], robot["orientation"])
        expected = (sealed["x"], sealed["y"], sealed["orientation"])
        if actual != expected:
            raise ValueError(f"Sealed robot_initial_state for robot {robot['id']} no longer matches a fresh reset ({actual} != {expected}) -- the environment has changed since this manifest was sealed; regenerate manifests before evaluating")

    env.task_queue = [dict(task) for task in manifest["tasks"]]
    policy = load_policy(policy_name, num_agents)
    if hasattr(policy, "reset_hidden_states"):
        policy.reset_hidden_states()
    dispatcher = NearestTaskDispatcher(env)
    dispatcher.assign_tasks(env.task_queue)

    total_reward = 0.0
    steps = 0
    collisions = 0
    safety_overrides = 0
    for _ in range(MAX_STEPS):
        dispatcher.assign_pending_tasks(env.task_queue)
        if policy_name == "nearest_task":
            actions = dispatcher.act(env)
        else:
            observations = [env.get_local_observation(i) for i in range(num_agents)]
            masks = [env.get_action_mask(i) for i in range(num_agents)]
            actions = policy.act(observations, masks)

        state = env.step(actions)
        total_reward += float(state.get("reward", 0.0))
        collisions = int(state["metrics"].get("collision_count", 0.0))
        safety_overrides = int(state["metrics"].get("safety_overrides", 0.0))
        steps += 1
        if not env.task_queue:
            break

    return {
        "seed": manifest["seed"],
        "num_tasks": manifest["num_tasks"],
        "completed_tasks": env.completed_tasks,
        "completion_rate": env.completed_tasks / manifest["num_tasks"],
        "steps": steps,
        "collisions": collisions,
        "safety_overrides": safety_overrides,
        "episode_reward": total_reward,
    }


def main() -> None:
    manifest_paths = sorted(MANIFEST_DIR.glob("eval_N*_seed*.json"))
    if not manifest_paths:
        raise FileNotFoundError(f"No sealed manifests found in {MANIFEST_DIR}; run scripts/generate_manifests.py first")

    manifests = [load_and_verify_manifest(p) for p in manifest_paths]
    print(f"Verified {len(manifests)} sealed manifest hashes -- none have been altered since generation.")

    by_config: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for manifest in manifests:
        for algorithm in ALGORITHMS:
            result = run_manifest(algorithm, manifest)
            by_config.setdefault((algorithm, manifest["num_agents"]), []).append(result)

    summary: List[Dict[str, Any]] = []
    for (algorithm, num_agents), results in sorted(by_config.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        completion_values = [r["completion_rate"] for r in results]
        collision_values = [float(r["collisions"]) for r in results]
        reward_values = [r["episode_reward"] for r in results]

        completion_mean, completion_ci = bootstrap_confidence_interval(completion_values, seed=0)
        collision_mean, collision_ci = bootstrap_confidence_interval(collision_values, seed=0)
        reward_mean, reward_ci = bootstrap_confidence_interval(reward_values, seed=0)

        summary.append({
            "algorithm": algorithm,
            "num_agents": num_agents,
            "manifests_evaluated": len(results),
            "completion_rate_mean": completion_mean,
            "completion_rate_ci95": list(completion_ci),
            "collision_count_mean": collision_mean,
            "collision_count_ci95": list(collision_ci),
            "episode_reward_mean": reward_mean,
            "episode_reward_ci95": list(reward_ci),
            "per_seed": results,
        })

    output = Path("evaluation/sealed_benchmark_results.json")
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'Algorithm':<14}{'N':>4}  {'Completion (95% CI)':<26}{'Collisions (95% CI)':<26}Reward (95% CI)")
    for row in summary:
        completion_str = f"{row['completion_rate_mean']:.2f} [{row['completion_rate_ci95'][0]:.2f}, {row['completion_rate_ci95'][1]:.2f}]"
        collision_str = f"{row['collision_count_mean']:.1f} [{row['collision_count_ci95'][0]:.1f}, {row['collision_count_ci95'][1]:.1f}]"
        reward_str = f"{row['episode_reward_mean']:.1f} [{row['episode_reward_ci95'][0]:.1f}, {row['episode_reward_ci95'][1]:.1f}]"
        print(f"{row['algorithm']:<14}{row['num_agents']:>4}  {completion_str:<26}{collision_str:<26}{reward_str}")

    print(f"\nSaved {len(summary)} rows to {output}")
    print("Note: only 3 sealed seeds per fleet size -> bootstrap CIs are wide by construction, not a bug.")


if __name__ == "__main__":
    main()
