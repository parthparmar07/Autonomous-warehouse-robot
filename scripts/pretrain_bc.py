from __future__ import annotations

import argparse
from typing import List

import torch

from src.algorithms.dqn.agent import DQNAgent
from src.algorithms.iql.agent import IndependentQLearning
from src.algorithms.qmix.agent_network import QMIXAgentNetwork
from src.algorithms.qmix.mixer import QMIXMixer
from src.deployment.policies import generic_checkpoint, specific_checkpoint
from src.training.imitation import behavior_clone, collect_dagger_transitions, collect_expert_transitions
from src.utils.seeding import set_global_seed


def save_checkpoint(name: str, num_agents: int, payload) -> None:
    """Write both the generic path (used by cycling at other fleet sizes) and the
    fleet-size-specific path (preferred exact match, see src/deployment/policies.py)."""
    generic_checkpoint(name).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, generic_checkpoint(name))
    torch.save(payload, specific_checkpoint(name, num_agents))


def _qmix_forward(network: QMIXAgentNetwork):
    def forward(batch: torch.Tensor) -> torch.Tensor:
        q_values, _ = network.forward(batch)
        return q_values if q_values.dim() == 2 else q_values.unsqueeze(0)

    return forward


def _train_and_report(name: str, forward_fn, parameters, dataset, epochs: int, device: str = "cpu") -> None:
    loss = behavior_clone(forward_fn, parameters, dataset, epochs=epochs, device=device)
    print(f"  {name}: loss={loss:.4f} (n={len(dataset)})")


def run_dagger(
    label: str,
    act_fn_factory,
    train_fn,
    num_agents: int,
    episodes: int,
    max_steps: int,
    dagger_rounds: int,
    dagger_episodes: int,
    seed: int,
) -> List:
    """Shared DAgger loop: collect expert data, train, then repeatedly let the *current*
    network drive rollouts (querying the expert for the correct label at every state it
    visits) and retrain on the growing aggregated dataset. `train_fn(dataset)` must
    (re)fit the network(s) in place; `act_fn_factory()` must return a fresh
    (observations, masks) -> actions callable reflecting the network's current weights.
    """
    print(f"[{label}] collecting initial expert demonstrations...")
    dataset = collect_expert_transitions(num_agents=num_agents, episodes=episodes, max_steps=max_steps, seed_start=seed)
    print(f"[{label}] round 0 (behaviour cloning only):")
    train_fn(dataset)

    for round_idx in range(1, dagger_rounds + 1):
        act_fn = act_fn_factory()
        new_data = collect_dagger_transitions(
            act_fn, num_agents=num_agents, episodes=dagger_episodes, max_steps=max_steps,
            seed_start=seed + 10_000 * round_idx,
        )
        dataset = dataset + new_data
        print(f"[{label}] round {round_idx} (+{len(new_data)} DAgger transitions, {len(dataset)} total):")
        train_fn(dataset)

    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Imitation-learning warm start from the nearest-task BFS dispatcher, refined with DAgger "
            "(PRIMAL-style expert demonstrations; see docs/09_assignment1_paper_summaries.md, Paper 5)."
        )
    )
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--dagger-rounds", type=int, default=3)
    parser.add_argument("--dagger-episodes", type=int, default=25)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--only", nargs="+", choices=["dqn", "iql", "qmix"], default=["dqn", "iql", "qmix"], help="Train only these algorithms (e.g. --only qmix to (re)train just QMIX for a specific fleet size without redoing DQN/IQL).")
    args = parser.parse_args()

    set_global_seed(args.seed)

    if "dqn" in args.only:
        # --- DQN: one shared network across the fleet. ---
        dqn_agent = DQNAgent(epsilon_start=0.0, epsilon_end=0.0)

        def dqn_train(dataset):
            _train_and_report("dqn", dqn_agent.online_network.forward, dqn_agent.online_network.parameters(), dataset, args.epochs, str(dqn_agent.device))
            dqn_agent.target_network.load_state_dict(dqn_agent.online_network.state_dict())
            save_checkpoint("dqn", args.num_agents, dqn_agent.online_network.state_dict())

        def dqn_act_fn_factory():
            def act(observations, masks):
                return [dqn_agent.online_network.act(obs, mask) for obs, mask in zip(observations, masks)]

            return act

        run_dagger("dqn", dqn_act_fn_factory, dqn_train, args.num_agents, args.episodes, args.max_steps, args.dagger_rounds, args.dagger_episodes, args.seed)

    if "iql" in args.only:
        # --- IQL: one network per robot, all cloned from the same pooled dataset. ---
        iql = IndependentQLearning(num_agents=args.num_agents)
        for agent in iql.agents:
            agent.epsilon = 0.0

        def iql_train(dataset):
            for idx, agent in enumerate(iql.agents):
                _train_and_report(f"iql[{idx}]", agent.online_network.forward, agent.online_network.parameters(), dataset, args.epochs, str(agent.device))
                agent.target_network.load_state_dict(agent.online_network.state_dict())
            save_checkpoint("iql", args.num_agents, [agent.online_network.state_dict() for agent in iql.agents])

        def iql_act_fn_factory():
            def act(observations, masks):
                return [agent.online_network.act(obs, mask) for agent, obs, mask in zip(iql.agents, observations, masks)]

            return act

        run_dagger("iql", iql_act_fn_factory, iql_train, args.num_agents, args.episodes, args.max_steps, args.dagger_rounds, args.dagger_episodes, args.seed + 1)

    if "qmix" in args.only:
        # --- QMIX: pretrain each agent's utility network the same way. The mixer has no
        # per-agent expert target of its own and is left at its random init -- RL
        # fine-tuning (scripts/train_qmix.py --warm-start) is what would shape it. ---
        qmix_networks = [QMIXAgentNetwork() for _ in range(args.num_agents)]

        def qmix_train(dataset):
            for idx, network in enumerate(qmix_networks):
                _train_and_report(f"qmix[{idx}]", _qmix_forward(network), network.parameters(), dataset, args.epochs)
            save_checkpoint(
                "qmix", args.num_agents,
                {"agent_networks": [network.state_dict() for network in qmix_networks], "mixer": QMIXMixer(num_agents=args.num_agents).state_dict()},
            )

        def qmix_act_fn_factory():
            def act(observations, masks):
                return [network.act(obs, mask)[0] for network, obs, mask in zip(qmix_networks, observations, masks)]

            return act

        run_dagger("qmix", qmix_act_fn_factory, qmix_train, args.num_agents, args.episodes, args.max_steps, args.dagger_rounds, args.dagger_episodes, args.seed + 2)

    print("Done. Run `python scripts/benchmark_algorithms.py` (or scripts/evaluate_sealed.py) to evaluate the imitation-pretrained checkpoints.")


if __name__ == "__main__":
    main()
