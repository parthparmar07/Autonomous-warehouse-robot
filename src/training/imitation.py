from __future__ import annotations

from typing import Callable, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.baselines.nearest_task import NearestTaskDispatcher
from src.environment.layout import default_layout
from src.environment.task_generator import generate_tasks
from src.environment.warehouse_env import WarehouseEnv

Transition = Tuple[np.ndarray, int]


def collect_expert_transitions(
    num_agents: int = 5,
    episodes: int = 60,
    max_steps: int = 150,
    seed_start: int = 1000,
    grid_size: Tuple[int, int] = (20, 20),
) -> List[Transition]:
    """Roll out the nearest-task BFS dispatcher -- the project's rule-based baseline,
    which already gets ~100% completion -- on the real floor plan, and record every
    (local_observation, expert_action) pair it produces.

    This is the PRIMAL playbook (Sartoretti et al. 2019; cited in this project's own
    literature review, `docs/09_assignment1_paper_summaries.md`): imitate a working
    expert planner rather than relying on trial-and-error RL alone to discover a working
    policy in a sparse-reward, long-horizon task. The project's own lit review
    pre-registers exactly this fallback: "if our own method fails to reach the target
    completion rate through trial and error alone, initialising it from our rule-based
    planner is an approach with published support rather than an improvisation."

    Observations (not actions) are shared across every agent and algorithm here: the
    task is homogeneous per-robot (same local window, same bearing/load features), so
    one pooled dataset can pretrain a DQN's shared network, every IQL agent, or every
    QMIX agent network -- whichever is being warm-started.
    """
    transitions: List[Transition] = []
    for episode in range(episodes):
        seed = seed_start + episode
        env = WarehouseEnv(grid_size=grid_size, num_agents=num_agents, seed=seed, **default_layout(grid_size))
        env.reset(seed=seed)
        env.task_queue = generate_tasks(env, num_tasks=num_agents * 2, seed=seed)
        dispatcher = NearestTaskDispatcher(env)
        dispatcher.assign_tasks(env.task_queue)

        for _ in range(max_steps):
            dispatcher.assign_pending_tasks(env.task_queue)
            actions = []
            for agent_id, robot in enumerate(env.robots):
                obs = env.get_local_observation(agent_id)
                action = dispatcher.choose_action(robot)
                transitions.append((obs, action))
                actions.append(action)
            state = env.step(actions)
            if state.get("done", False):
                break

    return transitions


def class_weights_for(actions: np.ndarray, num_classes: int = 5) -> np.ndarray:
    """Inverse-frequency class weights. The BFS expert turns rarely (most steps are a
    straight FORWARD run or a STAY at a blocked cell) -- LEFT/RIGHT/BACKWARD make up a
    small fraction of the demonstrations, so unweighted cross-entropy lets a network
    reach low average loss by nearly never predicting them, which is exactly the
    behaviour observed in practice (a network that never turns cannot route around a
    shelf). Weighting by inverse frequency forces every action class to matter equally."""
    counts = np.bincount(actions, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    return counts.sum() / (num_classes * counts)


def behavior_clone(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    parameters: Iterable[torch.nn.Parameter],
    dataset: List[Transition],
    epochs: int = 8,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cpu",
    weight_classes: bool = True,
) -> float:
    """Supervised cross-entropy pretraining of a Q-network against expert actions.

    This does not fit the Bellman equation -- it just shapes the network's argmax to
    already match a policy that works, before RL (if any) fine-tunes the actual value
    estimates from there. Converges in a handful of epochs because it's ordinary
    classification, not sparse-reward exploration.
    """
    observations = np.stack([obs for obs, _ in dataset]).astype(np.float32)
    actions = np.array([action for _, action in dataset], dtype=np.int64)
    n = len(dataset)
    if n == 0:
        return 0.0

    weight = torch.as_tensor(class_weights_for(actions), dtype=torch.float32, device=device) if weight_classes else None
    optimizer = torch.optim.Adam(parameters, lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=weight)

    last_epoch_loss = 0.0
    for _ in range(epochs):
        permutation = np.random.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = permutation[start : start + batch_size]
            obs_batch = torch.as_tensor(observations[idx], dtype=torch.float32, device=device)
            act_batch = torch.as_tensor(actions[idx], dtype=torch.long, device=device)
            logits = forward_fn(obs_batch)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            loss = loss_fn(logits, act_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        last_epoch_loss = epoch_loss / n
    return last_epoch_loss


def collect_dagger_transitions(
    act_fn: Callable[[List[np.ndarray], List[List[bool]]], List[int]],
    num_agents: int = 5,
    episodes: int = 20,
    max_steps: int = 150,
    seed_start: int = 2000,
    grid_size: Tuple[int, int] = (20, 20),
) -> List[Transition]:
    """DAgger data aggregation (Ross, Gordon & Bagnell, 2011): let the *current* learned
    policy drive the rollout -- so it visits the states its own errors actually lead to,
    not just the expert's trajectory -- but label every visited state with what the
    expert would have done there. Plain behaviour cloning only ever sees on-expert-
    trajectory states, so once the learned policy drifts even slightly (as it always
    will, being imperfect), it finds itself in states it was never shown and the errors
    compound; querying the expert off-trajectory is what fixes that, since the BFS
    dispatcher can score any state on demand, not just ones it visited itself.
    """
    transitions: List[Transition] = []
    for episode in range(episodes):
        seed = seed_start + episode
        env = WarehouseEnv(grid_size=grid_size, num_agents=num_agents, seed=seed, **default_layout(grid_size))
        env.reset(seed=seed)
        env.task_queue = generate_tasks(env, num_tasks=num_agents * 2, seed=seed)
        dispatcher = NearestTaskDispatcher(env)
        dispatcher.assign_tasks(env.task_queue)

        for _ in range(max_steps):
            dispatcher.assign_pending_tasks(env.task_queue)
            observations = [env.get_local_observation(agent_id) for agent_id in range(num_agents)]
            masks = [env.get_action_mask(agent_id) for agent_id in range(num_agents)]
            expert_actions = [dispatcher.choose_action(robot) for robot in env.robots]
            transitions.extend(zip(observations, expert_actions))

            policy_actions = act_fn(observations, masks)
            state = env.step(policy_actions)
            if state.get("done", False):
                break

    return transitions
