"""Paired, fixed-duration rate evaluation; no training or checkpoint selection."""
import argparse
import json
from pathlib import Path
from statistics import mean
import torch
from src.environment.mixed_logistics import make_mixed_scenario
from src.baselines.nearest_task import NearestTaskDispatcher
from src.training.runner import collect_episode
from src.training.learner import RecurrentLearner


def evaluate_rates(checkpoint, output, episodes=10, intervals=(80,40,20,12,8), policies=('qmix','planner')):
    torch.set_num_threads(1)
    model=None
    if 'qmix' in policies:
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if payload.get('metadata',{}).get('arguments',{}).get('workload')!='mixed':
            raise ValueError('Rate evaluation requires a mixed-stock checkpoint')
        model=RecurrentLearner.from_checkpoint(payload,device='cpu')
        model.epsilon=0
        probe,_=make_mixed_scenario(410000)
        if model.state_dim!=len(probe.get_training_state()):raise ValueError('Incompatible mixed inventory state')
    rows=[];summary=[]
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for interval in intervals:
        for name in policies:
            group=[]
            for seed in range(410000,410000+episodes):
                env,manifest=make_mixed_scenario(seed,interval,600,5,'heldout_rates')
                backlog=[]
                def trace():
                    backlog.append(dict(tick=env.tick,unfinished=len(env.task_queue),
                        stored=sum(s['state']=='shelf' for s in env.inventory.values()),
                        shipped=sum(s['state']=='shipped' for s in env.inventory.values())))
                _,metrics=collect_episode(env,model if name=='qmix' else NearestTaskDispatcher(env),manifest,train_callback=trace)
                metrics.update(policy=name,backlog_trace=backlog)
                group.append(metrics);rows.append(metrics)
                (output/'episodes.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
            row=dict(policy=name,arrival_interval=interval,external_packages_per_100_steps=100/interval,
                incoming=sum(m['incoming_packages'] for m in group),stored=sum(m['stored_packages'] for m in group),
                requested=sum(m['shipping_requests'] for m in group),shipped=sum(m['shipped_packages'] for m in group),
                ship_completion=sum(m['shipped_packages'] for m in group)/sum(m['shipping_requests'] for m in group),
                unfinished_mean=mean(len(m['unfinished_orders']) for m in group),
                backlog_at_arrival_cutoff_mean=mean(m['backlog_trace'][399]['unfinished'] for m in group),
                backlog_peak_mean=mean(max(t['unfinished'] for t in m['backlog_trace']) for m in group),
                throughput_mean=mean(m['throughput'] for m in group),
                avoidable_depletions=sum(m['avoidable_depletions'] for m in group),
                executed_collisions=sum(m['executed_collisions'] for m in group),
                completed_charges=sum(m['completed_charge_cycles'] for m in group),
                mean_completed_job_time=mean(m['delivery_time'] for m in group if m['delivery_time'] is not None),
                weighted_tardiness=sum(m['weighted_tardiness'] for m in group))
            summary.append(row)
            (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
            print(json.dumps(row),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--output',required=True);p.add_argument('--episodes',type=int,default=10)
    p.add_argument('--policy',choices=('qmix','planner','both'),default='both')
    args=p.parse_args()
    if args.policy!='planner' and not args.checkpoint:p.error('--checkpoint is required for QMIX')
    evaluate_rates(args.checkpoint,args.output,args.episodes,policies=('qmix','planner') if args.policy=='both' else (args.policy,))
