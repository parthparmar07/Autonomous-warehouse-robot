"""Record a trained-policy episode as a per-tick trajectory for the PyBullet renderer (scripts/demo_pybullet.py)."""
import argparse
import json
from pathlib import Path

import torch

from src.baselines.nearest_task import NearestTaskDispatcher
from src.environment.mixed_logistics import make_mixed_scenario
from src.training.learner import RecurrentLearner
from src.training.runner import collect_episode

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--algorithm', default='qmix', choices=('qmix', 'vdn', 'iql', 'planner'))
    p.add_argument('--num-agents', type=int, default=5)
    p.add_argument('--seed', type=int, default=200000)
    p.add_argument('--ticks', type=int, default=600)
    p.add_argument('--output', default=str(ROOT / 'outputs' / 'demo' / 'trajectory.json'))
    args = p.parse_args()
    env, manifest = make_mixed_scenario(args.seed, 20, args.ticks, args.num_agents, 'sealed_test')
    if args.algorithm == 'planner':
        policy = NearestTaskDispatcher(env)
    else:
        path = ROOT / 'artifacts' / 'warehouse_v6_research' / 'mixed_stock_v1' / f'{args.algorithm}_policy_N{args.num_agents}.pt'
        policy = RecurrentLearner.from_checkpoint(torch.load(path, map_location='cpu', weights_only=True), device='cpu')
        policy.epsilon = 0
    snap = lambda: [dict(x=int(r['x']), y=int(r['y']), heading=int(r['orientation']), loaded=bool(r['load_status']),
                         mode=r['mode'], soc=round(r['energy_eu'] / r['capacity_eu'], 3)) for r in env.robots]
    frames = [dict(tick=0, robots=snap(), shipped=0)]
    shipped = lambda: sum(s['state'] == 'shipped' for s in env.inventory.values())
    _, metrics = collect_episode(env, policy, manifest,
                                 train_callback=lambda: frames.append(dict(tick=env.tick, robots=snap(), shipped=shipped())))
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(algorithm=args.algorithm, num_agents=args.num_agents, seed=args.seed,
                                   grid=env.grid.tolist(), frames=frames,
                                   completion_rate=metrics['completion_rate'],
                                   executed_collisions=metrics['executed_collisions'])), encoding='utf-8')
    print(json.dumps(dict(output=str(out), frames=len(frames), completion_rate=metrics['completion_rate'],
                          executed_collisions=metrics['executed_collisions'])))


if __name__ == '__main__':
    main()
