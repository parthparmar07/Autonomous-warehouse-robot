"""Optuna reward-shaping sensitivity study (OBJ-5 / NFR-4). Development seeds only; failed trials are retained.

Two objectives -> Pareto frontier: maximise completion, minimise collision rate (conflicts per robot-tick).
Every trial gets the same small BC warm start and RL budget, so only the sampled weights differ.
"""
import argparse
import json

import numpy as np
from scipy.stats import spearmanr

from src.training.runner import ROOT, parser, train


def main():
    import optuna
    p = argparse.ArgumentParser()
    p.add_argument('--algorithm', choices=('iql', 'vdn', 'qmix'), default='qmix')
    p.add_argument('--trials', type=int, default=16)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--num-agents', type=int, default=5)
    p.add_argument('--bc-episodes', type=int, default=10)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if min(args.trials, args.steps, args.num_agents, args.bc_episodes) < 1:
        raise ValueError('Positive budgets required')
    output = ROOT / 'outputs' / 'tuning'
    output.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        c = parser(args.algorithm).parse_args([])
        c.workload, c.num_agents, c.steps, c.horizon, c.device = 'mixed', args.num_agents, args.steps, 600, args.device
        c.bc_episodes, c.bc_epochs, c.dagger_rounds, c.bc_aux_weight = args.bc_episodes, 4, 0, 0.25
        c.eval_episodes, c.eval_every, c.checkpoint_every, c.publish_artifact = 5, 0, 0, False
        c.train_every, c.epsilon_steps, c.epsilon_start = 8, 30000, 0.3
        c.learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)
        c.conflict_cost = trial.suggest_float('conflict_cost', 0.01, 1.0, log=True)
        c.delivery_reward = trial.suggest_float('delivery_reward', 5.0, 20.0)
        c.potential_weight = trial.suggest_float('potential_weight', 0.0, 4.0)
        _, metadata, results = train(c)
        trial.set_user_attr('run_id', metadata['run_id'])
        trial.set_user_attr('executed_collisions', float(sum(r['executed_collisions'] for r in results)))
        return (float(np.mean([r['completion_rate'] for r in results])),
                float(np.mean([r['conflicts_per_robot_tick'] for r in results])))

    name = f'{args.algorithm}_N{args.num_agents}_steps{args.steps}_reward_shaping'
    study = optuna.create_study(study_name=name, storage=f'sqlite:///{(output / "study.sqlite3").as_posix()}',
                                load_if_exists=True, directions=['maximize', 'minimize'],
                                sampler=optuna.samplers.TPESampler(seed=11))
    study.optimize(objective, n_trials=args.trials, catch=(RuntimeError, ValueError, FloatingPointError))
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    rows = [dict(number=t.number, params=t.params, completion=t.values[0], collision_rate=t.values[1], **t.user_attrs) for t in done]
    pareto = sorted({t.number for t in study.best_trials})
    rho = spearmanr([r['params']['conflict_cost'] for r in rows], [r['collision_rate'] for r in rows]) if len(rows) > 2 else None
    report = dict(study=name, scope='development seeds 300000-300004 only; sealed test never used',
                  budget=dict(steps=args.steps, bc_episodes=args.bc_episodes, bc_epochs=4, num_agents=args.num_agents),
                  trials=rows, failed_trials=len(study.trials) - len(done), pareto_front=pareto,
                  conflict_cost_vs_collision_rate_spearman=None if rho is None else dict(rho=float(rho.statistic), p=float(rho.pvalue)))
    (output / f'{name}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(pareto_front=pareto, spearman=report['conflict_cost_vs_collision_rate_spearman'], completed=len(rows))))


if __name__ == '__main__':
    main()
