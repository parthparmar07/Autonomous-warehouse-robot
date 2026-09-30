from __future__ import annotations

import argparse

import torch

from src.algorithms.dqn.trainer import DQNTrainer
from src.baselines.nearest_task import NearestTaskDispatcher
from src.deployment.policies import generic_checkpoint, resolve_checkpoint, specific_checkpoint
from src.environment.layout import default_layout
from src.environment.task_generator import generate_tasks
from src.environment.warehouse_env import WarehouseEnv
from src.utils.config import load_yaml_config
from src.utils.seeding import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DQN on the canonical warehouse environment")
    parser.add_argument("--warm-start", dest="warm_start", action="store_true", default=True, help="Load a matching dqn checkpoint before training if one exists (default: on).")
    parser.add_argument("--no-warm-start", dest="warm_start", action="store_false", help="Start from a random init even if a checkpoint exists.")
    args = parser.parse_args()

    config = load_yaml_config("configs/training.yaml")
    training_cfg = config["training"]
    seed = 42
    set_global_seed(seed)

    num_agents = int(training_cfg.get("num_agents", 5))
    episodes = int(training_cfg.get("episodes", 50))
    max_steps = int(training_cfg.get("max_steps", 100))
    grid_size = (20, 20)
    env = WarehouseEnv(grid_size=grid_size, num_agents=num_agents, seed=seed, **default_layout(grid_size))
    trainer = DQNTrainer(
        replay_capacity=training_cfg.get("replay_capacity", 100000),
        batch_size=training_cfg.get("batch_size", 64),
        learning_rate=training_cfg.get("learning_rate", 1e-3),
        gamma=training_cfg.get("gamma", 0.99),
    )

    warm_start_checkpoint = resolve_checkpoint("dqn", num_agents) if args.warm_start else None
    if warm_start_checkpoint is not None:
        state_dict = torch.load(warm_start_checkpoint, map_location=trainer.agent.device, weights_only=True)
        trainer.agent.online_network.load_state_dict(state_dict)
        trainer.agent.target_network.load_state_dict(state_dict)
        print(f"Warm-started from {warm_start_checkpoint}")

    losses = []
    for episode in range(episodes):
        env.reset(seed=seed + episode)
        env.task_queue = generate_tasks(env, num_tasks=num_agents * 2, seed=seed + episode)
        dispatcher = NearestTaskDispatcher(env)
        dispatcher.assign_tasks(env.task_queue)

        for _ in range(max_steps):
            dispatcher.assign_pending_tasks(env.task_queue)
            observations = [env.get_local_observation(agent_id) for agent_id in range(num_agents)]
            masks = [env.get_action_mask(agent_id) for agent_id in range(num_agents)]
            previous_distances = [
                _distance_to_target(robot) for robot in env.robots
            ]
            actions = trainer.act(observations, masks)
            state = env.step(actions)
            next_observations = [env.get_local_observation(agent_id) for agent_id in range(num_agents)]
            done = bool(state.get("done", False))
            for agent_id in range(num_agents):
                progress = previous_distances[agent_id] - _distance_to_target(env.robots[agent_id])
                shaped_reward = float(state.get("reward", 0.0)) + 0.05 * progress
                trainer.add_transition(
                    observations[agent_id], actions[agent_id], shaped_reward,
                    next_observations[agent_id], done, masks[agent_id], masks[agent_id],
                )
            loss = trainer.train_step()
            if loss:
                losses.append(loss)
            if done:
                break
        if episode % 10 == 0:
            trainer.update_target()

    generic_checkpoint("dqn").parent.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.agent.online_network.state_dict(), generic_checkpoint("dqn"))
    torch.save(trainer.agent.online_network.state_dict(), specific_checkpoint("dqn", num_agents))

    average_loss = sum(losses) / len(losses) if losses else 0.0
    print({"episodes": episodes, "num_agents": num_agents, "checkpoint": str(specific_checkpoint("dqn", num_agents)), "average_loss": average_loss})


def _distance_to_target(robot: dict) -> float:
    target = robot.get("target")
    if target is None:
        return 0.0
    return float(abs(target[0] - robot["x"]) + abs(target[1] - robot["y"]))


if __name__ == "__main__":
    main()
