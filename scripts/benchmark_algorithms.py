from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from src.deployment.policies import load_policy
from src.evaluation.scenarios import make_scenario
from src.evaluation.metrics import summarize_metrics
from src.training.runner import collect_episode
from src.environment.spec import ENV_VERSION

def main(default_algorithms=None):
    p=argparse.ArgumentParser(description='Development comparison using paired scenarios and measured operational outcomes')
    p.add_argument('--algorithms',nargs='+',default=default_algorithms or ['nearest_task','iql','vdn','qmix'])
    p.add_argument('--fleets',nargs='+',type=int,default=[5,10,20])
    p.add_argument('--episodes',type=int,default=20);p.add_argument('--horizon',type=int,default=400)
    args=p.parse_args()
    if min(args.episodes,args.horizon)<1:raise ValueError('Positive counts required')
    rows=[]
    for n in args.fleets:
        for algorithm in args.algorithms:
            results=[]
            for seed in range(300000,300000+args.episodes):
                env,manifest=make_scenario(seed,n,horizon=args.horizon,split='development')
                policy=load_policy(algorithm,n,env=env)
                _,metrics=collect_episode(env,policy,manifest)
                results.append(metrics)
            rows.append(dict(algorithm=algorithm,num_agents=n,summary=asdict(summarize_metrics(results)),results=results))
            print(dict(algorithm=algorithm,num_agents=n,**rows[-1]['summary']),flush=True)
    target=Path('outputs/evaluation')/f'{ENV_VERSION}_development_{time.time_ns()}.json'
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(rows,indent=2),encoding='utf-8');print(target)
if __name__=='__main__':main()
