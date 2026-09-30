from __future__ import annotations

import argparse

import numpy as np
import torch

from src.algorithms.qmix.trainer import QMIXTrainer
from src.baselines.nearest_task import NearestTaskDispatcher
from src.deployment.policies import generic_checkpoint, resolve_checkpoint, specific_checkpoint
from src.environment.layout import default_layout
from src.environment.task_generator import generate_tasks
from src.environment.warehouse_env import WarehouseEnv
from src.utils.seeding import set_global_seed


def global_state(env: WarehouseEnv) -> np.ndarray:
    values: list[float] = []
    for robot in env.robots:
        values.extend((robot["x"] / env.grid_size[0], robot["y"] / env.grid_size[1], robot["orientation"] / 3.0, float(robot["load_status"])))
    values.extend((env.episode_metrics["step_count"] / 200.0, len(env.task_queue) / 20.0))
    return np.pad(np.asarray(values, dtype=np.float32), (0, max(0, 64 - len(values))))[:64]


def _distance_to_target(robot: dict) -> float:
    target = robot.get("target")
    if target is None:
        return 0.0
    return float(abs(target[0] - robot["x"]) + abs(target[1] - robot["y"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train QMIX on the canonical warehouse environment")
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--warm-start", dest="warm_start", action="store_true", default=True, help="Load a matching qmix checkpoint before training if one exists (default: on).")
    parser.add_argument("--no-warm-start", dest="warm_start", action="store_false", help="Start from a random init even if a checkpoint exists.")
    args = parser.parse_args()

    seed = 321
    num_agents = args.num_agents
    set_global_seed(seed)
    trainer = QMIXTrainer(num_agents=num_agents, batch_size=32)
    grid_size = (20, 20)
    env = WarehouseEnv(grid_size=grid_size, num_agents=num_agents, seed=seed, **default_layout(grid_size))
    losses: list[float] = []

    warm_start_checkpoint = resolve_checkpoint("qmix", num_agents) if args.warm_start else None
    if warm_start_checkpoint is not None:
        artifact = torch.load(warm_start_checkpoint, map_location=trainer.device, weights_only=True)
        agent_state_dicts = artifact["agent_networks"]
        if len(agent_state_dicts) == num_agents:
            for network, target_network, state_dict in zip(trainer.agent_networks, trainer.target_agent_networks, agent_state_dicts):
                network.load_state_dict(state_dict)
                target_network.load_state_dict(state_dict)
            trainer.mixer.load_state_dict(artifact["mixer"])
            trainer.target_mixer.load_state_dict(artifact["mixer"])
            print(f"Warm-started agent networks + mixer from {warm_start_checkpoint}")
        else:
            for network, target_network, idx in zip(trainer.agent_networks, trainer.target_agent_networks, range(num_agents)):
                state_dict = agent_state_dicts[idx % len(agent_state_dicts)]
                network.load_state_dict(state_dict)
                target_network.load_state_dict(state_dict)
            print(f"Warm-started agent networks only (cycled) from {warm_start_checkpoint}; mixer stays random-init since it has {len(agent_state_dicts)} agents, need {num_agents}")

    update_count = 0
    for episode in range(args.episodes):
        env.reset(seed=seed + episode)
        env.task_queue = generate_tasks(env, num_tasks=num_agents * 2, seed=seed + episode)
        dispatcher = NearestTaskDispatcher(env)
        dispatcher.assign_tasks(env.task_queue)
        trainer.reset_hidden_states()  # recurrent memory must not leak across episodes
        for step in range(args.max_steps):
            dispatcher.assign_pending_tasks(env.task_queue)
            observations = [env.get_local_observation(agent_id) for agent_id in range(num_agents)]
            masks = [env.get_action_mask(agent_id) for agent_id in range(num_agents)]
            previous_distances = [_distance_to_target(robot) for robot in env.robots]
            hidden_before = list(trainer.hidden_states)
            actions = trainer.act(observations, masks)
            hidden_after = list(trainer.hidden_states)
            state_before = global_state(env)
            state = env.step(actions)
            next_observations = [env.get_local_observation(agent_id) for agent_id in range(num_agents)]
            # Team-level potential shaping: average per-agent progress toward its current
            # target, added to the shared team reward. QMIX trains on one joint reward, so
            # this nudges every agent toward "get closer to your task" without breaking the
            # cooperative value-decomposition structure -- it is not itself the objective.
            team_progress = sum(previous_distances[i] - _distance_to_target(env.robots[i]) for i in range(num_agents)) / num_agents
            shaped_reward = float(state.get("reward", 0.0)) + 0.1 * team_progress
            trainer.add_transition(
                state_before, actions, shaped_reward, global_state(env), bool(state.get("done", False)),
                masks, masks, observations, next_observations, hidden_before, hidden_after,
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

    payload = {
        "agent_networks": [network.state_dict() for network in trainer.agent_networks],
        "mixer": trainer.mixer.state_dict(),
    }
    generic_checkpoint("qmix").parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, generic_checkpoint("qmix"))
    torch.save(payload, specific_checkpoint("qmix", num_agents))
    print({"episodes": args.episodes, "num_agents": num_agents, "updates": update_count, "checkpoint": str(specific_checkpoint("qmix", num_agents)), "average_loss": sum(losses) / len(losses) if losses else 0.0})


if __name__ == "__main__":
    main()
