"""Task-queue drift monitor: Evidently PSI on order stream and live queue length; PSI > 0.25 triggers retraining."""
import argparse
import json
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.baselines.nearest_task import NearestTaskDispatcher
from src.environment.mixed_logistics import make_mixed_scenario
from src.training.runner import collect_episode

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 0.25
# Seed ranges disjoint from train (100000+), sealed (200000+), development (300000+) and rates (410000+).
REGIMES = {'reference': (520000, 20), 'control_normal': (530000, 20), 'high_demand_3x': (530000, 7)}


def order_features(manifest):
    rows, last = [], {}
    for event in sorted(manifest['events'], key=lambda e: e['tick']):
        order = event['order']
        (pr, pc), (dr, dc) = order['pickup'], order['delivery']
        flow = order['flow']
        rows.append(dict(inter_arrival=event['tick'] - last.get(flow, event['tick']),
                         route_distance=abs(pr - dr) + abs(pc - dc), outbound=int(flow == 'outbound')))
        last[flow] = event['tick']
    return rows


def queue_features(env, manifest):
    trace = []
    collect_episode(env, NearestTaskDispatcher(env), manifest,
                    train_callback=lambda: trace.append(dict(queue_length=len(env.task_queue))))
    return trace


def collect(seed0, interval, episodes, agents):
    orders, queue = [], []
    for seed in range(seed0, seed0 + episodes):
        env, manifest = make_mixed_scenario(seed, interval, 600, agents, 'drift_monitor')
        orders += order_features(manifest)
        queue += queue_features(env, manifest)
    return pd.DataFrame(orders), pd.DataFrame(queue)


def psi(reference, current, html):
    snapshot = Report([DataDriftPreset(method='psi', threshold=THRESHOLD)]).run(reference_data=reference, current_data=current)
    snapshot.save_html(str(html))
    return {m['metric_name'].split('column=')[1].split(',')[0]: float(m['value'])
            for m in snapshot.dict()['metrics'] if m['metric_name'].startswith('ValueDrift')}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--num-agents', type=int, default=5)
    p.add_argument('--output', default=str(ROOT / 'outputs' / 'monitoring'))
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    data = {name: collect(seed, interval, args.episodes, args.num_agents) for name, (seed, interval) in REGIMES.items()}
    ref_orders, ref_queue = data['reference']
    report = dict(threshold=THRESHOLD, method='population_stability_index (Evidently ValueDrift)',
                  episodes_per_regime=args.episodes, num_agents=args.num_agents,
                  regimes={k: dict(seed_start=s, arrival_interval=i) for k, (s, i) in REGIMES.items()}, comparisons={})
    for name in ('control_normal', 'high_demand_3x'):
        orders, queue = data[name]
        scores = {**psi(ref_orders, orders, out / f'drift_orders_{name}.html'),
                  **psi(ref_queue, queue, out / f'drift_queue_{name}.html')}
        report['comparisons'][name] = dict(psi=scores, drifted=[k for k, v in scores.items() if v > THRESHOLD],
                                           retrain_triggered=any(v > THRESHOLD for v in scores.values()),
                                           mean_queue_length=float(queue['queue_length'].mean()),
                                           reference_mean_queue_length=float(ref_queue['queue_length'].mean()))
        print(name, json.dumps(report['comparisons'][name]), flush=True)
    (out / 'drift_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    main()
