import argparse
import json
from pathlib import Path
import torch
from src.deployment.policies import load_policy
from src.evaluation.policy_cases import POLICY_CASES,materialize,skill_outcome
from src.training.runner import collect_episode

def main():
    p=argparse.ArgumentParser();p.add_argument('--algorithm',default='nearest_task')
    p.add_argument('--cases',nargs='+',choices=POLICY_CASES,default=list(POLICY_CASES))
    p.add_argument('--num-agents',type=int,default=5);p.add_argument('--episodes',type=int,default=20)
    p.add_argument('--horizon',type=int,default=600);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path,default=Path('outputs/evaluation/skill_evaluation.json'))
    args=p.parse_args();torch.set_num_threads(1)
    rows=[]
    for case in args.cases:
        for index in range(args.episodes):
            env,manifest=materialize(case,303000+index,args.num_agents,args.horizon,index)
            model=load_policy(args.algorithm,args.num_agents,env=env,checkpoint=args.checkpoint)
            _,metrics=collect_episode(env,model,dict(seed=manifest['seed'],scenario_hash=manifest['manifest_hash']))
            rows.append(dict(case=case,role=index,**metrics,**skill_outcome(case,env,metrics)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(scope='operational prerequisites; statistical catalog oracles remain explicit',results=rows),indent=2),encoding='utf-8')
    print(dict(episodes=len(rows),operational_prerequisites_passed=sum(r['operational_prerequisite_passed'] for r in rows),output=str(args.output)))
if __name__=='__main__':main()
