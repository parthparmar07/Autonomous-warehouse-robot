from __future__ import annotations

from pathlib import Path

import torch

from src.algorithms.iql.trainer import IQLTrainer
from src.baselines.nearest_task import NearestTaskDispatcher
from src.environment.task_generator import generate_tasks
from src.environment.warehouse_env import WarehouseEnv
from src.utils.seeding import set_global_seed


def main() -> None:
    seed = 123
    num_agents = 5
    episodes = 30
    max_steps = 100
    set_global_seed(seed)
    trainer = IQLTrainer(num_agents=num_agents, batch_size=32)
    env = WarehouseEnv(grid_size=(20, 20), num_agents=num_agents, seed=seed)
    losses: list[float] = []

    for episode in range(episodes):
        env.reset(seed=seed + episode)
        env.task_queue = generate_tasks(env, num_tasks=num_agents * 2, seed=seed + episode)
        dispatcher = NearestTaskDispatcher(env)
        dispatcher.assign_tasks(env.task_queue)

        for _ in range(max_steps):
            dispatcher.assign_pending_tasks(env.task_queue)
            observations = [env.get_local_observation(agent_id) for agent_id in range(num_agents)]
            masks = [env.get_action_mask(agent_id) for agent_id in range(num_agents)]
            actions = trainer.model.act(observations, masks)
            state = env.step(actions)
            next_observations = [env.get_local_observation(agent_id) for agent_id in range(num_agents)]
            done = bool(state.get("done", False))

            for agent_id in range(num_agents):
                trainer.add_transition(
                    agent_id,
                    observations[agent_id],
                    actions[agent_id],
                    float(state.get("reward", 0.0)),
                    next_observations[agent_id],
                    done,
                    masks[agent_id],
                    masks[agent_id],
                )
            losses.extend(loss for loss in trainer.train_step() if loss)
            if done:
                break

    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = artifact_dir / "iql_policy.pt"
    torch.save([agent.online_network.state_dict() for agent in trainer.model.agents], checkpoint)
    average_loss = sum(losses) / len(losses) if losses else 0.0
    print({"episodes": episodes, "checkpoint": str(checkpoint), "average_loss": average_loss})


if __name__ == "__main__":
    main()
