"""Sealed 9-cell scalability benchmark (planner/IQL/QMIX x N=5/10/20) plus adversarial corridor-safety episodes.

Confidence intervals bootstrap over the 20 sealed scenarios of a single training seed per cell, so they
capture scenario variance only, not training-seed variance.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.environment.mixed_logistics import make_mixed_scenario
from src.safety.action_mask import orientation_delta
from src.training.learner import RecurrentLearner
from src.training.runner import evaluate

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'artifacts' / 'warehouse_v6_research' / 'mixed_stock_v1'
SEALED = list(range(200000, 200020))
# Demand grows with the fleet; N=20 is capped at 7 because 2 orders per arrival must fit the 128-order state.
DEMAND_SCALED_INTERVAL = {5: 20, 10: 10, 20: 7}
METRICS = ('completion_rate', 'delivery_time', 'conflicts_per_robot_tick', 'collision_count', 'executed_collisions',
           'energy_used', 'energy_per_delivery', 'avoidable_depletions', 'safety_violations', 'safety_overrides', 'throughput')


def bootstrap_ci(values, samples=5000, seed=0):
    values = np.asarray(values, dtype=float)
    draws = np.random.default_rng(seed).choice(values, (samples, len(values)), replace=True).mean(1)
    return dict(mean=float(values.mean()), ci95=np.quantile(draws, [.025, .975]).tolist())


def paired(a, b, key):
    return bootstrap_ci([x[key] - y[key] for x, y in zip(a, b)])


def ratio_ci(a, b, key, samples=5000, seed=0):
    a, b = np.asarray([r[key] for r in a], float), np.asarray([r[key] for r in b], float)
    idx = np.random.default_rng(seed).integers(len(a), size=(samples, len(a)))
    draws = a[idx].sum(1) / np.maximum(b[idx].sum(1), 1e-9)
    return dict(ratio=float(a.sum() / max(b.sum(), 1e-9)), ci95=np.quantile(draws, [.025, .975]).tolist())


def load(algorithm, n, device):
    path = ARTIFACTS / f'{algorithm}_policy_N{n}.pt'
    if not path.exists():
        return None, str(path)
    model = RecurrentLearner.from_checkpoint(torch.load(path, map_location='cpu', weights_only=True), device=device)
    model.epsilon = 0
    return model, str(path)


def adversarial_safety(n, seeds, rng_seed=0):
    """Ignores the mask: drives at any adjacent corridor cell, otherwise moves/turns at random."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    for seed in seeds:
        env, _ = make_mixed_scenario(seed, 20, 600, n, 'sealed_safety')
        intrusions = 0
        while env.tick < env.config.horizon:
            actions = []
            for robot in env.robots:
                choice = int(rng.integers(4))
                for action in (0, 1):
                    dx, dy = orientation_delta(robot['orientation'], action)
                    x, y = robot['x'] + dx, robot['y'] + dy
                    if env._is_in_bounds(x, y) and env.grid[y, x] == 2:
                        choice = action
                actions.append(choice)
            result = env.step(actions)
            intrusions += sum(env.grid[r['y'], r['x']] == 2 for r in env.robots)
            if result['done']:
                break
        rows.append(dict(seed=seed, corridor_intrusions=int(intrusions),
                         safety_overrides=float(env.episode_metrics['safety_overrides']),
                         executed_safety_violations=float(env.episode_metrics['executed_safety_violations'])))
    return dict(episodes=len(rows), corridor_intrusions=sum(r['corridor_intrusions'] for r in rows),
                blocked_corridor_attempts=sum(r['safety_overrides'] for r in rows),
                executed_safety_violations=sum(r['executed_safety_violations'] for r in rows),
                passed=all(r['corridor_intrusions'] == 0 and r['executed_safety_violations'] == 0 for r in rows), per_episode=rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fleets', type=int, nargs='+', default=[5, 10, 20])
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--device', default='cuda')
    p.add_argument('--demand-scaled', action='store_true',
                   help='Arrival rate grows with N (policies were trained at interval 20, so this tests generalisation)')
    p.add_argument('--algorithms', nargs='+', default=['planner', 'iql', 'qmix'], choices=('planner', 'iql', 'qmix'))
    p.add_argument('--output')
    args = p.parse_args()
    seeds = SEALED[:args.episodes]
    intervals = {n: DEMAND_SCALED_INTERVAL[n] if args.demand_scaled else 20 for n in args.fleets}
    report = dict(workload='mixed_stock_v1', split='sealed_test', seeds=seeds, horizon=600, arrival_interval=intervals,
                  demand_scaled=args.demand_scaled,
                  ci_note='Percentile bootstrap over sealed scenarios; one training seed per cell (scenario variance only).',
                  cells={}, comparisons={}, safety={})
    name = 'scalability_demand_scaled.json' if args.demand_scaled else 'scalability_N5_N10_N20.json'
    out = Path(args.output or ROOT / 'outputs' / 'evaluation' / name); out.parent.mkdir(parents=True, exist_ok=True)
    for n in args.fleets:
        episodes = {}
        for algorithm in args.algorithms:
            model, source = (None, 'NearestTaskDispatcher') if algorithm == 'planner' else load(algorithm, n, args.device)
            key = f'{algorithm}_N{n}'
            if algorithm != 'planner' and model is None:
                report['cells'][key] = dict(status='missing_checkpoint', checkpoint=source)
                continue
            rows = evaluate(model, seeds, n, 'C5', 'sealed_test', 600, workload='mixed', arrival_interval=intervals[n])
            episodes[algorithm] = rows
            report['cells'][key] = dict(status='evaluated', checkpoint=source, num_agents=n, episodes=len(rows),
                                        arrival_interval=intervals[n], orders_released=float(np.mean([r['released_tasks'] for r in rows])),
                                        metrics={m: bootstrap_ci([r[m] if r[m] is not None else np.nan for r in rows])
                                                 for m in METRICS if all(r.get(m) is not None for r in rows)},
                                        completion_per_seed=[r['completion_rate'] for r in rows],
                                        scenario_hashes=[r.get('scenario_hash') for r in rows])
            print(key, json.dumps({m: round(v['mean'], 4) for m, v in report['cells'][key]['metrics'].items()}), flush=True)
            out.write_text(json.dumps(report, indent=2), encoding='utf-8')
        hashes = {a: [r.get('scenario_hash') for r in rows] for a, rows in episodes.items()}
        assert len({tuple(h) for h in hashes.values()}) <= 1, f'Unpaired scenarios at N={n}'
        if 'qmix' in episodes:
            for other in ('iql', 'planner'):
                if other in episodes:
                    report['comparisons'][f'qmix_minus_{other}_N{n}'] = {
                        k: paired(episodes['qmix'], episodes[other], k)
                        for k in ('completion_rate', 'conflicts_per_robot_tick', 'collision_count', 'energy_per_delivery')}
            if 'iql' in episodes:
                report['comparisons'][f'qmix_over_iql_collision_ratio_N{n}'] = ratio_ci(episodes['qmix'], episodes['iql'], 'collision_count')
        if args.demand_scaled:
            out.write_text(json.dumps(report, indent=2), encoding='utf-8')
            continue
        report['safety'][f'N{n}'] = adversarial_safety(n, seeds)
        print(f'safety_N{n}', json.dumps({k: v for k, v in report['safety'][f"N{n}"].items() if k != 'per_episode'}), flush=True)
        out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    main()
