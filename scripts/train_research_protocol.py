"""Execute one matched staged run; source/config stay frozen for the process."""
import argparse
import json
from pathlib import Path
import torch
import yaml
import os
from src.training.runner import parser,train,collect_episode,runtime_settings,ROOT
from src.training.provenance import provenance
from src.evaluation.scenarios import make_scenario
from src.evaluation.policy_cases import materialize
from src.evaluation.curriculum_gate import evaluate_curriculum_gate
from src.evaluation.curriculum import eligible_cases
from src.evaluation.policy_cases import skill_outcome
from src.baselines.nearest_task import NearestTaskDispatcher


def stage_profile(config,stage):
    return config.get('stage_profiles',{}).get(stage,config.get('profile','robust'))


def atomic_json(path,payload):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    os.replace(temporary,path)


def resume_progress(report,checkpoint):
    """Only completed, promoted stages survive; unsaved gate streaks do not."""
    completed=[row for row in report['stages'] if row['promoted']]
    total=completed[-1]['steps'] if completed else 0
    return completed,total,checkpoint


def stage_demonstration_count(config,stage,resume_stage):
    """The saved same-stage checkpoint already contains the expert reservoir."""
    return 0 if stage==resume_stage else config['bc_episodes']


def screen(model,stages,args,index,stage_profiles=None):
    """Cheap development screen; it can reject, never authorize promotion."""
    from dataclasses import replace
    rows=[];epsilon=model.epsilon;model.epsilon=0
    try:
        for stage in stages:
            for sample in range(5 if stage==stages[-1] else 1):
                seed=306000+index*10+sample
                env,manifest=make_scenario(seed,args.num_agents,stage,'development',args.horizon,
                    profile=(stage_profiles or {}).get(stage,args.profile),runtime_overrides=runtime_settings(args))
                _,metrics=collect_episode(env,model,manifest)
                rows.append(dict(case=stage,passed=metrics['completion_rate']>=.95 and
                    metrics['executed_collisions']==0 and metrics['avoidable_depletions']==0,metrics=metrics))
        for case in sorted({c for stage in stages for c in eligible_cases(stage,args.num_agents)}):
            env,manifest=materialize(case,307000+index,args.num_agents,args.horizon,index%(args.num_agents*4))
            env.config=replace(env.config,**runtime_settings(args))
            from src.evaluation.manifests import capture_manifest
            manifest=capture_manifest(env,307000+index,'development',case)
            _,metrics=collect_episode(env,model,dict(seed=manifest['seed'],scenario_hash=manifest['manifest_hash']))
            rows.append(dict(case=case,passed=skill_outcome(case,env,metrics)['operational_prerequisite_passed'],metrics=metrics))
    finally:model.epsilon=epsilon
    return dict(passed=all(r['passed'] for r in rows),results=rows,
                scope='Development screen only; full statistical/role checks required for promotion')


def run(config,algorithm,seed,output,resume_protocol=False):
    args=parser(algorithm).parse_args([])
    for key,value in config.items():
        if hasattr(args,key):setattr(args,key,value)
    args.algorithm=algorithm;args.seed=seed;args.publish_artifact=False
    args.full_demonstration_coverage=True
    args.stage_profiles={stage:stage_profile(config,stage) for stage in config['stages']}
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads)
    args.checkpoint_pointer=output/'latest_checkpoint.json'
    previous=None;resume_path=None;resume_stage=None
    if resume_protocol:
        previous=json.loads((output/'result.json').read_text(encoding='utf-8'))
        if previous['config']!=config or previous['algorithm']!=algorithm or previous['seed']!=seed:
            raise ValueError('Protocol resume configuration, algorithm or seed differs')
        pointer=json.loads(args.checkpoint_pointer.read_text(encoding='utf-8'))
        resume_path=Path(pointer['checkpoint'])
        resume_stage=pointer['stage']
        if not resume_path.is_file():raise ValueError('Latest replay checkpoint is missing')
    elif (output/'result.json').exists():
        raise ValueError('Output contains an existing study; use --resume-protocol or a new directory')
    report=dict(algorithm=algorithm,seed=seed,condition=config['study'],config=config,
        provenance=provenance(ROOT),stages=[],status='training',research_ready=False)
    total=0
    if previous:
        report['stages'],total,resume_path=resume_progress(previous,resume_path)
        report['previous_attempts']=previous.get('previous_attempts',[])+[dict(
            status=previous['status'],stages=previous['stages'],current=previous.get('current'))]
        report['resume_note']='Gate streaks reset conservatively at process restart; saved learner/replay/cursor are restored.'
        report['restarts']=previous.get('restarts',0)+1
        args.resume=resume_path
    def save():atomic_json(output/'result.json',report)
    save();seen=[]
    for stage,cap in config['stages'].items():
        seen.append(stage)
        if any(row['stage']==stage for row in report['stages']):continue
        args.stage=stage;args.profile=stage_profile(config,stage);args.steps=total+cap
        args.bc_episodes=stage_demonstration_count(config,stage,resume_stage)
        args.epsilon_steps=config.get('stage_epsilon_steps',{}).get(stage,config.get('epsilon_steps',10000))
        stage_profiles={s:stage_profile(config,s) for s in seen}
        report['current']=dict(stage=stage,profile=args.profile,stage_start_steps=total,stage_cap=cap,status='initializing')
        save()
        checks=[];consecutive=0;last=-1;passed=False
        def check(model,current,steps):
            nonlocal consecutive,last,passed
            if steps==last:return passed
            preview=screen(model,seen,args,len(checks),stage_profiles)
            if preview['passed']:
                gate=evaluate_curriculum_gate(model,seen,args.num_agents,args.eval_episodes,
                    config['skill_episodes'],args.horizon,profile=args.profile,runtime_overrides=runtime_settings(args),stage_profiles=stage_profiles)
            else:
                gate=dict(passed=False,generic_gate=dict(families=[]),skill_gate=dict(families=[]),screen=preview)
            consecutive=consecutive+1 if gate['passed'] else 0
            passed=consecutive>=config['promotion']['consecutive_checks'] and steps-total>=config['promotion']['minimum_stage_rl_steps']
            record=dict(stage=current,environment_steps=steps,consecutive_passes=consecutive,
                promoted=passed,gate=gate,screen=preview)
            checks.append(record);last=steps
            with (output/'checks.jsonl').open('a',encoding='utf-8') as log:log.write(json.dumps(record)+'\n')
            report['current']=dict(stage=current,profile=args.profile,stage_start_steps=total,stage_cap=cap,environment_steps=steps,gate_passed=gate['passed'],consecutive=consecutive)
            save()
            print(json.dumps(dict(kind='research_gate',stage=current,steps=steps,passed=gate['passed'],promoted=passed,
                generic={f['family']:f['successes'] for f in gate['generic_gate']['families']},
                skills={f['case']:f['successes'] for f in gate['skill_gate']['families']})),flush=True)
            return passed
        model,metadata,rows=train(args,evaluation_callback=check)
        total=model.environment_steps
        report['stages'].append(dict(stage=stage,profile=args.profile,run_id=metadata['run_id'],steps=total,
            promoted=passed,final_completion=sum(r['completion_rate'] for r in rows)/len(rows),
            expert_data=metadata.get('demonstrations'),dagger=metadata.get('dagger'),
            source_unchanged=metadata['source_unchanged_during_run']))
        if not passed:
            report.update(status='stopped_failed_development_gate',environment_steps=total)
            save();return report
        args.resume=ROOT/'outputs/runs'/metadata['run_id']/'checkpoint.pt'
        resume_stage=stage
        # Deliberately matched stage-local demonstrations, not hidden pure RL.
        args.bc_episodes=config['bc_episodes']
    report.update(status='all_configured_development_stages_passed_final_tests_pending',environment_steps=total)
    save();return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/active_training.yaml')
    p.add_argument('--algorithm',choices=('qmix','iql'),required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume-protocol',action='store_true')
    a=p.parse_args();config=yaml.safe_load(a.config.read_text(encoding='utf-8'))
    report=run(config,a.algorithm,a.seed,a.output,a.resume_protocol)
    print(json.dumps(dict(status=report['status'],algorithm=a.algorithm,seed=a.seed,output=str(a.output))),flush=True)

if __name__=='__main__':main()
