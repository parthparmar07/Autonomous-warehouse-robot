from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.algorithms.qmix.trainer import QMIXTrainer
from src.baselines.nearest_task import NearestTaskDispatcher
from src.environment.task_generator import generate_tasks
from src.environment.warehouse_env import WarehouseEnv
from src.utils.seeding import set_global_seed


def global_state(env: WarehouseEnv) -> np.ndarray:
    values: list[float] = []
    for robot in env.robots:
        values.extend((robot["x"] / env.grid_size[0], robot["y"] / env.grid_size[1], robot["orientation"] / 3.0, float(robot["load_status"])))
    values.extend((env.episode_metrics["step_count"] / 200.0, len(env.task_queue) / 20.0))
    return np.pad(np.asarray(values, dtype=np.float32), (0, max(0, 64 - len(values))))[:64]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train QMIX on the canonical warehouse environment")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--train-every", type=int, default=4)
    args = parser.parse_args()

    seed = 321
    num_agents = 5
    set_global_seed(seed)
    trainer = QMIXTrainer(num_agents=num_agents, batch_size=32)
    env = WarehouseEnv(grid_size=(20, 20), num_agents=num_agents, seed=seed)
    losses: list[float] = []

    update_count = 0
    for episode in range(args.episodes):
        env.reset(seed=seed + episode)
        env.task_queue = generate_tasks(env, num_tasks=num_agents * 2, seed=seed + episode)
        dispatcher = NearestTaskDispatcher(env)
        dispatcher.assign_tasks(env.task_queue)
        for step in range(args.max_steps):
            dispatcher.assign_pending_tasks(env.task_queue)
            observations = [env.get_local_observation(agent_id) for agent_id in range(num_agents)]
            masks = [env.get_action_mask(agent_id) for agent_id in range(num_agents)]
            actions = trainer.act(observations, masks)
            state_before = global_state(env)
            state = env.step(actions)
            next_observations = [env.get_local_observation(agent_id) for agent_id in range(num_agents)]
            trainer.add_transition(
                state_before, actions, float(state.get("reward", 0.0)), global_state(env), bool(state.get("done", False)),
                masks, masks, observations, next_observations,
            )
            if step % max(1, args.train_every) == 0:
                loss = trainer.train_step()
                update_count += 1
                if loss:
                    losses.append(loss)
            if state.get("done", False):
                break
        if episode % 10 == 0:
            trainer.update_targets()

    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = artifact_dir / "qmix_policy.pt"
    torch.save({
        "agent_networks": [network.state_dict() for network in trainer.agent_networks],
        "mixer": trainer.mixer.state_dict(),
    }, checkpoint)
    print({"episodes": args.episodes, "updates": update_count, "checkpoint": str(checkpoint), "average_loss": sum(losses) / len(losses) if losses else 0.0})


if __name__ == "__main__":
    main()
