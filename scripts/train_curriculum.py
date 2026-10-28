"""GPU-capable staged training with periodic skill and retention checks."""
import argparse
import json
from pathlib import Path
from src.training.runner import parser,train,ROOT
from src.evaluation.curriculum import STAGES
from src.evaluation.curriculum_gate import evaluate_curriculum_gate

def run(args):
    base=parser(args.algorithm).parse_args([])
    for key in ('num_agents','seed','horizon','bc_episodes','bc_epochs','bc_aux_weight',
                'dagger_rounds','eval_episodes','device','train_every','eval_every',
                'potential_weight','epsilon_start','balanced_demonstrations'):
        setattr(base,key,getattr(args,key,getattr(base,key)))
    base.full_demonstration_coverage=base.balanced_demonstrations
    base.epsilon_steps=args.epsilon_steps or max(1,args.stage_steps//2)
    base.publish_artifact=False
    stages=[];checks=[];total=0
    output=ROOT/'outputs'/'curriculum'/f'{args.algorithm}_N{args.num_agents}_seed{args.seed}.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    for index,stage in enumerate(STAGES):
        base.stage=stage;base.steps=total+args.stage_steps
        latest=None
        def check(model,current_stage,steps):
            nonlocal latest
            latest=evaluate_curriculum_gate(model,STAGES[:index+1],args.num_agents,
                args.eval_episodes,args.skill_episodes,args.horizon)
            record=dict(stage=current_stage,environment_steps=steps,gate=latest)
            if checks and checks[-1]['stage']==current_stage and checks[-1]['environment_steps']==steps:
                checks[-1]=record
            else:checks.append(record)
            output.write_text(json.dumps(dict(status='training',stages=stages,checks=checks,
                environment_steps=steps),indent=2),encoding='utf-8')
            print(json.dumps(dict(kind='curriculum_prerequisites',stage=current_stage,
                environment_steps=steps,passed=latest['passed'])),flush=True)
        model,metadata,_=train(base,evaluation_callback=check)
        total=model.environment_steps
        stages.append(dict(stage=stage,run_id=metadata['run_id'],gate=latest))
        output.write_text(json.dumps(dict(status='stage_passed' if latest['passed'] else 'not_ready',
            stages=stages,checks=checks,environment_steps=total),indent=2),encoding='utf-8')
        if not latest['passed']:
            print(f'{stage} failed development prerequisites; stopped as not_ready. Details: {output}')
            return False
        base.resume=ROOT/'outputs'/'runs'/metadata['run_id']/'checkpoint.pt'
        base.bc_episodes=args.bc_episodes
    output.write_text(json.dumps(dict(status='development_passed_final_tests_pending',stages=stages,
        checks=checks,environment_steps=total),indent=2),encoding='utf-8')
    print('All development prerequisites passed; independent frozen test evaluation is still required.')
    return True

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--algorithm',choices=('iql','vdn','qmix'),default='qmix')
    p.add_argument('--num-agents',type=int,default=5);p.add_argument('--seed',type=int,default=11)
    p.add_argument('--stage-steps',type=int,default=100000);p.add_argument('--horizon',type=int,default=400)
    p.add_argument('--bc-episodes',type=int,default=0);p.add_argument('--bc-epochs',type=int,default=3)
    p.add_argument('--bc-aux-weight',type=float,default=0);p.add_argument('--dagger-rounds',type=int,default=0)
    p.add_argument('--potential-weight',type=float,default=2.0)
    p.add_argument('--epsilon-start',type=float,default=.3)
    p.add_argument('--balanced-demonstrations',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--eval-episodes',type=int,default=20)
    p.add_argument('--skill-episodes',type=int,default=None,help='Default max(20,4*N), covering each worker/heading')
    p.add_argument('--eval-every',type=int,default=25000)
    p.add_argument('--device',default='cuda')
    p.add_argument('--train-every',type=int,default=16)
    p.add_argument('--epsilon-steps',type=int,default=None,help='Exploration decay per stage, default half its budget')
    args=p.parse_args()
    if (args.stage_steps<1 or args.eval_episodes<20 or args.eval_every<1 or args.train_every<1
        or args.epsilon_steps is not None and args.epsilon_steps<1
        or args.skill_episodes is not None and args.skill_episodes<max(20,4*args.num_agents)):
        raise ValueError('Positive budgets and at least 20 cases / complete worker-heading roles required')
    run(args)

if __name__=='__main__':main()
