"""Explicit matched algorithm x fleet x training-seed matrix; no silent failed-run exclusion."""
import argparse
import json
import time
from src.training.runner import parser,train,ROOT

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--algorithms',nargs='+',choices=('iql','vdn','qmix'),default=['iql','vdn','qmix'])
    p.add_argument('--fleets',nargs='+',type=int,default=[5,10,20])
    p.add_argument('--seeds',nargs='+',type=int,default=[11,23,37,53,71])
    p.add_argument('--steps',type=int,default=200000)
    p.add_argument('--bc-episodes',type=int,default=0)
    p.add_argument('--bc-aux-weight',type=float,default=0)
    p.add_argument('--scaling',choices=('fixed_map','area_per_robot'),default='fixed_map')
    args=p.parse_args()
    directory=ROOT/'outputs'/'experiments'/str(time.time_ns());directory.mkdir(parents=True)
    (directory/'matrix.json').write_text(json.dumps(vars(args),indent=2),encoding='utf-8')
    with (directory/'runs.jsonl').open('w',encoding='utf-8') as log:
        for n in args.fleets:
            for seed in args.seeds:
                for algorithm in args.algorithms:
                    config=parser(algorithm).parse_args([])
                    config.num_agents=n;config.seed=seed;config.steps=args.steps;config.bc_episodes=args.bc_episodes
                    config.bc_aux_weight=args.bc_aux_weight;config.scaling=args.scaling
                    try:
                        _,metadata,_=train(config)
                        row=dict(algorithm=algorithm,num_agents=n,training_seed=seed,status='completed',run_id=metadata['run_id'])
                    except Exception as exc:
                        row=dict(algorithm=algorithm,num_agents=n,training_seed=seed,status='failed',error=f'{type(exc).__name__}: {exc}')
                    log.write(json.dumps(row)+'\n');log.flush()
    print(directory)

if __name__=='__main__':main()
