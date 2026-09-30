from __future__ import annotations

import argparse

import torch

from src.algorithms.iql.trainer import IQLTrainer
from src.baselines.nearest_task import NearestTaskDispatcher
from src.deployment.policies import generic_checkpoint, resolve_checkpoint, specific_checkpoint
from src.environment.layout import default_layout
from src.environment.task_generator import generate_tasks
from src.environment.warehouse_env import WarehouseEnv
from src.utils.seeding import set_global_seed


def _distance_to_target(robot: dict) -> float:
    target = robot.get("target")
    if target is None:
        return 0.0
    return float(abs(target[0] - robot["x"]) + abs(target[1] - robot["y"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IQL on the canonical warehouse environment")
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--warm-start", dest="warm_start", action="store_true", default=True, help="Load a matching iql checkpoint before training if one exists (default: on).")
    parser.add_argument("--no-warm-start", dest="warm_start", action="store_false", help="Start from a random init even if a checkpoint exists.")
    args = parser.parse_args()

    seed = 123
    num_agents = args.num_agents
    episodes = args.episodes
    max_steps = args.max_steps
    set_global_seed(seed)
    trainer = IQLTrainer(num_agents=num_agents, batch_size=32)
    grid_size = (20, 20)
    env = WarehouseEnv(grid_size=grid_size, num_agents=num_agents, seed=seed, **default_layout(grid_size))
    losses: list[float] = []

    warm_start_checkpoint = resolve_checkpoint("iql", num_agents) if args.warm_start else None
    if warm_start_checkpoint is not None:
        state_dicts = torch.load(warm_start_checkpoint, map_location="cpu", weights_only=True)
        for idx, agent in enumerate(trainer.model.agents):
            state_dict = state_dicts[idx % len(state_dicts)]
            agent.online_network.load_state_dict(state_dict)
            agent.target_network.load_state_dict(state_dict)
        exact = "exact match" if len(state_dicts) == num_agents else f"cycled from {len(state_dicts)} agents"
        print(f"Warm-started from {warm_start_checkpoint} ({exact})")

    for episode in range(episodes):
        env.reset(seed=seed + episode)
        env.task_queue = generate_tasks(env, num_tasks=num_agents * 2, seed=seed + episode)
        dispatcher = NearestTaskDispatcher(env)
        dispatcher.assign_tasks(env.task_queue)

        for _ in range(max_steps):
            dispatcher.assign_pending_tasks(env.task_queue)
            observations = [env.get_local_observation(agent_id) for agent_id in range(num_agents)]
            masks = [env.get_action_mask(agent_id) for agent_id in range(num_agents)]
            previous_distances = [_distance_to_target(robot) for robot in env.robots]
            actions = trainer.model.act(observations, masks)
            state = env.step(actions)
            next_observations = [env.get_local_observation(agent_id) for agent_id in range(num_agents)]
            done = bool(state.get("done", False))

            for agent_id in range(num_agents):
                progress = previous_distances[agent_id] - _distance_to_target(env.robots[agent_id])
                shaped_reward = float(state.get("reward", 0.0)) + 0.05 * progress
                trainer.add_transition(
                    agent_id,
                    observations[agent_id],
                    actions[agent_id],
                    shaped_reward,
                    next_observations[agent_id],
                    done,
                    masks[agent_id],
                    masks[agent_id],
                )
            losses.extend(loss for loss in trainer.train_step() if loss)
            if done:
                break

    payload = [agent.online_network.state_dict() for agent in trainer.model.agents]
    generic_checkpoint("iql").parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, generic_checkpoint("iql"))
    torch.save(payload, specific_checkpoint("iql", num_agents))
    average_loss = sum(losses) / len(losses) if losses else 0.0
    print({"episodes": episodes, "num_agents": num_agents, "checkpoint": str(specific_checkpoint("iql", num_agents)), "average_loss": average_loss})


if __name__ == "__main__":
    main()
