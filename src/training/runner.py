from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict,replace
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
import os
import numpy as np
import torch
import yaml
from src.baselines.nearest_task import NearestTaskDispatcher
from src.environment.spec import ENV_VERSION, OBS_VERSION, ACTION_DIM
from src.evaluation.scenarios import make_scenario
from src.evaluation.operational import episode_metrics
from src.training.learner import RecurrentLearner, LearnerConfig
from src.utils.hashing import sha256_digest
from src.training.provenance import provenance
from src.evaluation.curriculum import STAGES,TRAINING_CASES,eligible_cases
from src.training.demonstrations import demonstration_case,demonstration_count

ROOT=Path(__file__).resolve().parents[2]
LOADED_SOURCE=provenance(ROOT)


def save_training_checkpoint(model,metadata,args,next_episode,path):
    """Atomic, episode-boundary recovery: never publish a partial replay file."""
    cursor=dict(stage=args.stage,next_episode=next_episode,profile=args.profile,
        runtime_settings=runtime_settings(args),epsilon_start=args.epsilon_start,
        **{key:getattr(args,key) for key in ('seed','scaling','tiny','horizon','scenario_mix','epsilon_steps','train_every')})
    saved=dict(metadata,training_cursor=cursor,environment_steps=model.environment_steps,
        updates=model.updates,checkpoint_kind='completed_episode_boundary',
        checkpoint_saved_at=datetime.now(timezone.utc).isoformat())
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    try:
        torch.save(model.checkpoint(saved,include_replay=True),temporary)
        os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()
    status=path.with_suffix('.json');temporary_status=status.with_name(status.name+'.tmp')
    temporary_status.write_text(json.dumps(dict(path=str(path),environment_steps=model.environment_steps,
        next_episode=next_episode,saved_at=saved['checkpoint_saved_at'])),encoding='utf-8')
    os.replace(temporary_status,status)
    pointer=getattr(args,'checkpoint_pointer',None)
    if pointer:
        pointer=Path(pointer);pointer.parent.mkdir(parents=True,exist_ok=True)
        temporary_pointer=pointer.with_name(pointer.name+'.tmp')
        temporary_pointer.write_text(json.dumps(dict(checkpoint=str(path.resolve()),
            run_id=metadata['run_id'],stage=args.stage,environment_steps=model.environment_steps,
            next_episode=next_episode,saved_at=saved['checkpoint_saved_at'])),encoding='utf-8')
        os.replace(temporary_pointer,pointer)
    return saved

def runtime_settings(args):
    return dict(potential_weight=args.potential_weight,potential_gamma=args.gamma,
        idle_bay_cost=args.idle_bay_cost,delivery_reward=args.delivery_reward,conflict_cost=args.conflict_cost)

def training_case(args,seed,index):
    if getattr(args,'workload','outbound')=='mixed':
        from src.environment.mixed_logistics import make_mixed_scenario
        intervals=(8,12,20,40,80)
        env,manifest=make_mixed_scenario(seed,intervals[index%len(intervals)],args.horizon,args.num_agents,'train',runtime_settings(args))
        manifest.update(case_id='mixed',role_index=index)
        return env,manifest
    cases=eligible_cases(args.stage,args.num_agents)
    if not args.tiny and args.num_agents>1 and args.scaling=='fixed_map' and args.scenario_mix and cases and index%5==4:
        from src.evaluation.policy_cases import materialize
        case=cases[(index//5)%len(cases)];role=(index//5)//len(cases)
        env,sealed=materialize(case,seed,args.num_agents,args.horizon,role)
        env.config=replace(env.config,**runtime_settings(args))
        from src.evaluation.manifests import capture_manifest
        sealed=capture_manifest(env,seed,'train',case)
        sealed['split']='train';sealed['manifest_hash']='';sealed['manifest_hash']=sha256_digest(sealed)
        return env,dict(seed=seed,scenario_hash=sealed['manifest_hash'],case_id=case,role_index=role,materialized=sealed)
    env,manifest=make_scenario(seed,args.num_agents,args.stage,'train',args.horizon,tiny=args.tiny,role_index=index,scaling=args.scaling,
        profile=args.profile,runtime_overrides=runtime_settings(args))
    manifest['case_id']=args.stage;manifest['role_index']=index
    return env,manifest


def collect_episode(env, policy, manifest, max_steps=None, train_callback=None,expert_labels=False):
    env.prepare_decision()
    if hasattr(policy,'reset_hidden_states'):
        policy.reset_hidden_states()
    observations=[env.get_local_observation(i) for i in range(env.num_agents)]
    states=env.get_training_state()
    masks=[env.get_action_mask(i) for i in range(env.num_agents)]
    data=dict(observations=[observations],states=[states],masks=[masks],alive=[[r['active'] for r in env.robots]],
              actions=[],rewards=[],terminated=[])
    latencies=[]
    action_counts=Counter();visited=[{(r['x'],r['y'])} for r in env.robots]
    robot_counts=[Counter() for _ in env.robots];mode_counts=[{} for _ in env.robots]
    reward_totals=Counter();entropy_sum=np.zeros(env.num_agents)
    teacher=NearestTaskDispatcher(env) if expert_labels else None
    if expert_labels:data['expert_actions']=[]
    for _ in range(max_steps or env.config.horizon):
        start=time.perf_counter_ns()
        actions=policy.act(env) if isinstance(policy,NearestTaskDispatcher) else policy.act(observations,masks)
        epsilon=float(getattr(policy,'epsilon',0.0))
        for i,(robot,action,mask) in enumerate(zip(env.robots,actions,masks)):
            robot_counts[i][action]+=1
            mode_counts[i].setdefault(robot['mode'],Counter())[action]+=1
            if epsilon==0.0 and hasattr(policy,'last_entropy'):
                entropy_sum[i]+=float(policy.last_entropy[i])
            else:
                k=sum(mask);p=epsilon/k;g=1-epsilon+p
                entropy_sum[i]+=-g*np.log(max(g,1e-30))-(k-1)*p*np.log(max(p,1e-30))
        latencies.append((time.perf_counter_ns()-start)/1e6)
        if teacher:data['expert_actions'].append(teacher.act(env))
        result=env.step(actions)
        reward_totals.update(result.get('reward_components',{}))
        action_counts.update(actions)
        for cells,robot in zip(visited,env.robots):cells.add((robot['x'],robot['y']))
        observations=[env.get_local_observation(i) for i in range(env.num_agents)]
        masks=[env.get_action_mask(i) for i in range(env.num_agents)]
        data['actions'].append(actions);data['rewards'].append(result['reward']);data['terminated'].append(result['terminated'])
        data['observations'].append(observations);data['masks'].append(masks)
        data['states'].append(env.get_training_state());data['alive'].append([r['active'] for r in env.robots])
        if train_callback:
            train_callback()
        if result['done']:
            break
    episode={k:np.asarray(v,dtype=bool if k in ('masks','alive','terminated') else np.int64 if k in ('actions','expert_actions') else np.float32) for k,v in data.items()}
    metrics=episode_metrics(env,manifest,latencies)
    metrics.update(action_counts=dict(action_counts),per_robot_unique_cells=[len(cells) for cells in visited],
        per_robot_action_counts=[dict(c) for c in robot_counts],
        per_robot_mode_actions=[{m:dict(c) for m,c in row.items()} for row in mode_counts],
        behavior_entropy_mean=(entropy_sum/max(1,len(data['actions']))).tolist(),reward_components=dict(reward_totals))
    return episode,metrics


def evaluate(model, seeds, n, stage='C5', split='development', horizon=400, tiny=False,scaling='fixed_map',profile='robust',runtime_overrides=None,workload='outbound',arrival_interval=20):
    if len(set(seeds))!=len(seeds):
        raise ValueError('Evaluation scenarios must be unique')
    epsilon=getattr(model,'epsilon',None)
    if epsilon is not None:
        model.epsilon=0
    results=[]
    try:
        for seed in seeds:
            if workload=='mixed':
                from src.environment.mixed_logistics import make_mixed_scenario
                env,manifest=make_mixed_scenario(seed,arrival_interval,horizon,n,split,runtime_overrides)
            else:
                env,manifest=make_scenario(seed,n,stage,split,horizon,tiny=tiny,scaling=scaling,
                    profile=profile,runtime_overrides=runtime_overrides)
            policy=NearestTaskDispatcher(env) if model is None else model
            _,metrics=collect_episode(env,policy,manifest)
            results.append(metrics)
    finally:
        if epsilon is not None:model.epsilon=epsilon
    return results


def readiness(results):
    return bool(results and all(r['completion_rate']>=.8 and r['executed_collisions']==0 and r['avoidable_depletions']==0 for r in results))


def _eval_rank(results):
    if not results:
        return (-1, -1e9, 0.0, 0.0, -1e9)
    penalties=sum(r.get('avoidable_depletions',0)+r.get('executed_collisions',0)+r.get('safety_violations',0) for r in results)
    return (
        int(readiness(results)),
        -float(penalties),
        float(np.mean([r['completion_rate'] for r in results])),
        float(np.mean([r.get('throughput',0.0) for r in results])),
        -float(np.mean([r.get('energy_used',0.0) for r in results])),
    )


def resume_learner(payload, args, env, config):
    """Resume exactly; reject silently ignored algorithm or learner overrides."""
    saved_workload=payload.get('metadata',{}).get('arguments',{}).get('workload','outbound')
    if saved_workload!=getattr(args,'workload','outbound'):
        raise ValueError('Resume workload differs from checkpoint')
    expected=dict(algorithm=args.algorithm,num_agents=args.num_agents,obs_dim=env.observation_dim,
                  state_dim=len(env.get_training_state()),action_dim=ACTION_DIM)
    for key,value in expected.items():
        if payload.get(key)!=value:
            raise ValueError(f'Resume {key} mismatch: checkpoint={payload.get(key)!r}, requested={value!r}')
    actual=LearnerConfig(**payload['config'])
    differences=[key for key,value in asdict(config).items() if getattr(actual,key)!=value]
    if differences:
        raise ValueError('Resume learner configuration mismatch: '+', '.join(differences)+'. Use the checkpoint settings for exact continuation.')
    model=RecurrentLearner.from_checkpoint(payload,device=args.device,resume=True)
    for group in model.optimizer.param_groups:
        if group['lr']!=model.config.learning_rate:
            raise ValueError('Checkpoint optimizer learning rate differs from its learner configuration')
    cursor=payload.get('metadata',{}).get('training_cursor')
    if cursor is None:
        raise ValueError('Checkpoint has no training cursor; exact scenario continuation cannot be established')
    keys=('seed','scaling','tiny','horizon','scenario_mix','train_every')
    if cursor.get('stage')==args.stage:keys+=('epsilon_steps',)
    for key in keys:
        if cursor.get(key)!=getattr(args,key):
            raise ValueError(f'Resume scenario/schedule setting differs from checkpoint: {key}')
    if cursor.get('stage')==args.stage:
        if cursor.get('profile',args.profile)!=args.profile or cursor.get('runtime_settings',runtime_settings(args))!=runtime_settings(args):
            raise ValueError('Resume profile/reward contract differs within the same stage')
    if cursor.get('epsilon_start',args.epsilon_start)!=args.epsilon_start:
        raise ValueError('Resume exploration start differs from checkpoint')
    return model,cursor['next_episode'] if cursor['stage']==args.stage else 0


def aggregate_dagger(model, demonstrations, args, metadata,groups=None):
    aggregate=list(demonstrations)
    aggregate_groups=list(groups) if groups is not None else None
    loss=model.behavior_clone(aggregate,args.bc_epochs)
    count=demonstration_count(args,args.bc_episodes)
    for round_index in range(getattr(args,'dagger_rounds',0)):
        corrections=[]
        for index in range(count):
            case_index=index if getattr(args,'full_demonstration_coverage',True) else index+(round_index+1)*count
            expert_env,expert_manifest=demonstration_case(args,130000+args.seed*100+round_index*count+index,case_index)
            data,_=collect_episode(expert_env,model,expert_manifest,expert_labels=True)
            corrections.append(data)
            if aggregate_groups is not None:aggregate_groups.append(expert_manifest.get('case_id',args.stage))
        aggregate.extend(corrections)
        loss=model.behavior_clone(aggregate,args.bc_epochs)
        metadata.setdefault('dagger',[]).append(dict(round=round_index,episodes=len(corrections),
            aggregate_episodes=len(aggregate),joint_steps=sum(len(d['actions']) for d in corrections)))
    model.demonstrations.add_representative(aggregate,aggregate_groups)
    return loss


def train(args,evaluation_callback=None):
    torch.set_num_threads(args.threads)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    env,manifest=make_scenario(100000+args.seed,args.num_agents,args.stage,'train',args.horizon,tiny=args.tiny,scaling=args.scaling,
        profile=args.profile,runtime_overrides=runtime_settings(args))
    if getattr(args,'workload','outbound')=='mixed':
        if args.balanced_demonstrations or args.tiny or args.stage!='C5':
            raise ValueError('Mixed workload requires C5 and mixed-flow demonstrations')
        env,manifest=training_case(args,100000+args.seed,0)
    config=LearnerConfig(batch_size=args.batch_size,burn_in=args.burn_in,unroll=args.unroll,
        learning_rate=args.learning_rate,gamma=args.gamma,target_every=args.target_every,bc_aux_weight=args.bc_aux_weight,
        replay_episodes=args.replay_episodes,mixer_hidden=args.mixer_hidden,hidden=args.hidden)
    model=RecurrentLearner(args.algorithm,args.num_agents,env.observation_dim,len(env.get_training_state()),config=config,seed=args.seed,device=args.device)
    episode_index=0
    if args.resume:
        model,episode_index=resume_learner(torch.load(args.resume,map_location='cpu',weights_only=True),args,env,config)
    same_stage=args.resume and model.metadata.get('stage')==args.stage
    stage_start_steps=model.metadata.get('stage_start_steps',0) if same_stage else model.environment_steps
    model.replay.set_stage(args.stage,.2 if args.stage!='C0' else 0.0)
    model.demonstrations.set_stage(args.stage,.2 if args.stage!='C0' else 0.0)
    run_id=f'{args.algorithm}_N{args.num_agents}_seed{args.seed}_{time.time_ns()}'
    run_dir=ROOT/'outputs'/'runs'/run_id
    run_dir.mkdir(parents=True)
    metadata=dict(run_id=run_id,seed=args.seed,environment_version=ENV_VERSION,observation_version=OBS_VERSION,
                  runtime=env.config.to_dict(),learner=asdict(model.config),stage=args.stage,started_at=datetime.now(timezone.utc).isoformat(),
                  arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},scenario_hashes=[],
                  software=dict(torch=str(torch.__version__),numpy=np.__version__,cuda_build=torch.version.cuda,
                    device=str(model.device),gpu_name=torch.cuda.get_device_name(model.device) if model.device.type=='cuda' else None),
                  provenance=LOADED_SOURCE,research_ready=False,stage_start_steps=stage_start_steps)
    if args.resume:
        metadata['resumed_from']=dict(checkpoint=str(args.resume),run_id=model.metadata.get('run_id'),
            provenance=model.metadata.get('provenance'),environment_steps=model.environment_steps)
        if same_stage:
            metadata['scenario_hashes']=list(model.metadata.get('scenario_hashes',[]))
            if not args.bc_episodes:
                for key in ('demonstrations','dagger'):
                    if key in model.metadata:metadata[key]=model.metadata[key]
                metadata['expert_preparation_restored']=True
    (run_dir/'config.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    log=(run_dir/'metrics.jsonl').open('w',encoding='utf-8')
    tracker=None
    if args.mlflow:
        import mlflow
        from src.tracking.mlflow_logger import MLFlowLogger
        mlflow.set_tracking_uri((ROOT/'outputs'/'mlruns').resolve().as_uri())
        tracker=MLFlowLogger(ENV_VERSION);tracker.start_run(run_id)
        tracker.log_params(dict(algorithm=model.algorithm,num_agents=args.num_agents,stage=args.stage,seed=args.seed,**asdict(model.config)))
    started=time.perf_counter()
    dev_seeds=list(range(300000,300000+args.eval_episodes))
    eval_args=dict(profile=args.profile,runtime_overrides=runtime_settings(args),workload=getattr(args,'workload','outbound'))
    initial_results=evaluate(model,dev_seeds,args.num_agents,args.stage,'development',args.horizon,args.tiny,args.scaling,**eval_args)
    (run_dir/'initial_development.json').write_text(json.dumps(initial_results,indent=2),encoding='utf-8')
    eval_every=getattr(args,'eval_every',0)
    next_evaluation=(model.environment_steps//eval_every+1)*eval_every if eval_every else None
    checkpoint_every=getattr(args,'checkpoint_every',5000)
    next_checkpoint=(model.environment_steps//checkpoint_every+1)*checkpoint_every if checkpoint_every else None
    demonstrations=[];coverage={};demonstration_groups=[]
    actual_demonstrations=demonstration_count(args,args.bc_episodes)
    for episode in range(actual_demonstrations):
        expert_env,expert_manifest=demonstration_case(args,120000+args.seed*100+episode,episode)
        data,metrics=collect_episode(expert_env,NearestTaskDispatcher(expert_env),expert_manifest)
        demonstrations.append(data)
        demonstration_groups.append(expert_manifest.get('case_id',args.stage))
        coverage.setdefault(expert_manifest.get('case_id',args.stage),set()).add(expert_manifest.get('role_index',episode)%(args.num_agents*4))
    if demonstrations:
        bc_start=time.perf_counter()
        groups=demonstration_groups if getattr(args,'balanced_demonstrations',False) else None
        bc_loss=aggregate_dagger(model,demonstrations,args,metadata,groups)
        model.replay.add_representative(demonstrations,groups)
        metadata['demonstrations']=dict(requested_episodes=args.bc_episodes,episodes=len(demonstrations),
            roles_per_family={key:sorted(value) for key,value in coverage.items()},
            balanced=getattr(args,'balanced_demonstrations',False),
            joint_steps=sum(len(d['actions']) for d in demonstrations),epochs=args.bc_epochs,optimization_seconds=time.perf_counter()-bc_start)
        log.write(json.dumps(dict(kind='behavior_cloning',loss=bc_loss,**metadata['demonstrations']))+'\n');log.flush()
        print(json.dumps(dict(run=run_id,bc_loss=bc_loss)),flush=True)
        bc_results=evaluate(model,dev_seeds,args.num_agents,args.stage,'development',args.horizon,args.tiny,args.scaling,**eval_args)
        (run_dir/'bc_development.json').write_text(json.dumps(bc_results,indent=2),encoding='utf-8')
        from src.deployment.policies import save_checkpoint
        save_checkpoint(model.checkpoint(dict(metadata,training_kind='bc_only',research_ready=False)),run_dir/'bc_policy.pt')
    # Preserve expensive expert preparation before the first RL transition.
    save_training_checkpoint(model,metadata,args,episode_index,run_dir/'latest_checkpoint.pt')
    last_loss=None
    best_rl=None
    while model.environment_steps<args.steps:
        env,manifest=training_case(args,100000+(args.seed*997+episode_index)%90000,episode_index)
        metadata['scenario_hashes'].append(manifest['scenario_hash'])
        eps_start=min(args.epsilon_start,.1) if model.demonstrations.episodes else args.epsilon_start
        model.epsilon=max(.05,eps_start-(eps_start-.05)*(model.environment_steps-stage_start_steps)/max(1,args.epsilon_steps))
        def update():
            nonlocal last_loss
            model.environment_steps+=1
            if model.environment_steps%args.train_every==0 and model.replay.episodes:
                last_loss=model.train_step()
        data,metrics=collect_episode(env,model,manifest,max_steps=min(env.config.horizon,args.steps-model.environment_steps),train_callback=update)
        model.replay.add(data)
        metrics.update(kind='rl_episode',case_id=manifest.get('case_id',args.stage),role_index=manifest.get('role_index',episode_index),episode=episode_index,environment_steps=model.environment_steps,updates=model.updates,epsilon=model.epsilon,
                       loss=None if last_loss is None else last_loss['loss'],elapsed_seconds=time.perf_counter()-started)
        log.write(json.dumps(metrics)+'\n');log.flush()
        if tracker:tracker.log_metrics({k:float(metrics[k]) for k in ('completion_rate','throughput','environment_steps','updates')})
        if episode_index%5==0:print(json.dumps({k:metrics[k] for k in ('episode','environment_steps','updates','completion_rate','loss')}),flush=True)
        episode_index+=1
        if next_checkpoint is not None and model.environment_steps>=next_checkpoint:
            save_training_checkpoint(model,metadata,args,episode_index,run_dir/'latest_checkpoint.pt')
            next_checkpoint=(model.environment_steps//checkpoint_every+1)*checkpoint_every
        if next_evaluation is not None and model.environment_steps>=next_evaluation:
            checkpoint_results=evaluate(model,dev_seeds,args.num_agents,args.stage,'development',args.horizon,args.tiny,args.scaling,**eval_args)
            if model.updates>0 and (best_rl is None or _eval_rank(checkpoint_results)>=best_rl['rank']):
                best_rl=dict(rank=_eval_rank(checkpoint_results),results=checkpoint_results,steps=model.environment_steps,updates=model.updates,
                    actor={k:v.detach().cpu().clone() for k,v in model.actor.state_dict().items()},
                    target_actor={k:v.detach().cpu().clone() for k,v in model.target_actor.state_dict().items()},
                    mixer={k:v.detach().cpu().clone() for k,v in model.mixer.state_dict().items()} if model.mixer else None,
                    target_mixer={k:v.detach().cpu().clone() for k,v in model.target_mixer.state_dict().items()} if model.target_mixer else None)
            record=dict(environment_steps=model.environment_steps,updates=model.updates,
                completion_mean=float(np.mean([r['completion_rate'] for r in checkpoint_results])),results=checkpoint_results)
            with (run_dir/'learning_curve.jsonl').open('a',encoding='utf-8') as curve:curve.write(json.dumps(record)+'\n')
            print(json.dumps(dict(kind='development',environment_steps=model.environment_steps,completion_mean=record['completion_mean'])),flush=True)
            if evaluation_callback is not None and evaluation_callback(model,args.stage,model.environment_steps) is True:
                break
            next_evaluation=(model.environment_steps//eval_every+1)*eval_every
    if model.updates==0 and model.replay.episodes:
        last_loss=model.train_step()
    results=evaluate(model,dev_seeds,args.num_agents,args.stage,'development',args.horizon,args.tiny,args.scaling,**eval_args)
    if best_rl is not None and best_rl['rank']>_eval_rank(results):
        model.actor.load_state_dict({k:v.to(model.device) for k,v in best_rl['actor'].items()})
        model.target_actor.load_state_dict({k:v.to(model.device) for k,v in best_rl['target_actor'].items()})
        if model.mixer and best_rl['mixer'] is not None:
            model.mixer.load_state_dict({k:v.to(model.device) for k,v in best_rl['mixer'].items()})
            model.target_mixer.load_state_dict({k:v.to(model.device) for k,v in best_rl['target_mixer'].items()})
        metadata['selected_rl_checkpoint']=dict(environment_steps=best_rl['steps'],updates=best_rl['updates'],
            final_completion_mean=float(np.mean([r['completion_rate'] for r in results])))
        results=best_rl['results']
    if evaluation_callback is not None:evaluation_callback(model,args.stage,model.environment_steps)
    metadata.update(finished_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started,
                    updates=model.updates,environment_steps=model.environment_steps,training_kind=model.training_kind,
                    agent_steps=model.environment_steps*args.num_agents,
                    development_gate_passed=readiness(results),research_ready=False,
                    note='Development results; research readiness requires the full held-out family/seed protocol, not this pilot gate.')
    metadata['training_cursor']=dict(stage=args.stage,next_episode=episode_index,
        profile=args.profile,runtime_settings=runtime_settings(args),epsilon_start=args.epsilon_start,
        **{key:getattr(args,key) for key in ('seed','scaling','tiny','horizon','scenario_mix','epsilon_steps','train_every')})
    metadata['source_unchanged_during_run']=provenance(ROOT)['source_sha256']==LOADED_SOURCE['source_sha256']
    scope='tiny' if args.tiny else args.scaling if args.scaling!='fixed_map' else '' if args.stage=='C5' else f'curriculum/{args.stage}'
    if getattr(args,'workload','outbound')=='mixed':scope='mixed_stock_v1'
    artifact=ROOT/'artifacts'/ENV_VERSION/scope/f'{args.algorithm}_policy_N{args.num_agents}.pt'
    artifact.parent.mkdir(parents=True,exist_ok=True)
    save_training_checkpoint(model,metadata,args,episode_index,run_dir/'checkpoint.pt')
    # Deployment artifact excludes replay; full run checkpoints retain it.
    from src.deployment.policies import save_checkpoint
    save_checkpoint(model.checkpoint(metadata),run_dir/'policy.pt')
    if getattr(args,'publish_artifact',True):save_checkpoint(model.checkpoint(metadata),artifact)
    (run_dir/'development.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    (run_dir/'summary.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    log.close()
    if tracker:tracker.end_run()
    print(json.dumps(dict(checkpoint=str(artifact if getattr(args,'publish_artifact',True) else run_dir/'policy.pt'),run_dir=str(run_dir),updates=model.updates,
                          completion_mean=float(np.mean([r['completion_rate'] for r in results])),
                          development_gate_passed=metadata['development_gate_passed'],research_ready=False)),flush=True)
    return model,metadata,results


def parser(default_algorithm='qmix'):
    p=argparse.ArgumentParser(description='Versioned recurrent warehouse training; all results carry their actual training stage')
    p.add_argument('--algorithm',choices=('iql','vdn','qmix','dqn'),default=default_algorithm)
    p.add_argument('--workload',choices=('outbound','mixed'),default='outbound')
    p.add_argument('--num-agents',type=int,default=5)
    p.add_argument('--steps',type=int,default=20000)
    p.add_argument('--horizon',type=int,default=400)
    p.add_argument('--stage',choices=STAGES,default='C5')
    p.add_argument('--scaling',choices=('fixed_map','area_per_robot'),default='fixed_map')
    p.add_argument('--scenario-mix',action=argparse.BooleanOptionalAction,default=True,help='Every fifth eligible episode materializes a rotating stage-specific skill case')
    p.add_argument('--seed',type=int,default=11)
    p.add_argument('--bc-episodes',type=int,default=0)
    p.add_argument('--balanced-demonstrations',action=argparse.BooleanOptionalAction,default=False,help='Balance current-stage examples and named skills within the requested episode budget')
    p.add_argument('--full-demonstration-coverage',action='store_true',help='Explicitly expand expert budget to cover all earlier families and every worker/heading')
    p.add_argument('--bc-epochs',type=int,default=3)
    p.add_argument('--dagger-rounds',type=int,default=0)
    p.add_argument('--bc-aux-weight',type=float,default=0,help='Explicit matched imitation-regularized ablation; zero in primary RL')
    p.add_argument('--eval-episodes',type=int,default=5)
    p.add_argument('--eval-every',type=int,default=0,help='Periodic development evaluation at episode boundaries; zero disables')
    p.add_argument('--checkpoint-every',type=int,default=5000,help='Atomic resumable save at completed-episode boundaries; zero disables periodic saves')
    p.add_argument('--checkpoint-pointer',type=Path,help='Atomic pointer for an external curriculum supervisor')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--burn-in',type=int,default=20)
    p.add_argument('--unroll',type=int,default=40)
    p.add_argument('--train-every',type=int,default=4)
    p.add_argument('--epsilon-steps',type=int,default=500000)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--device',default='cpu')
    p.add_argument('--profile',choices=('core','robust'),default='robust')
    p.add_argument('--potential-weight',type=float,default=2.0)
    p.add_argument('--idle-bay-cost',type=float,default=.02)
    p.add_argument('--delivery-reward',type=float,default=10.)
    p.add_argument('--conflict-cost',type=float,default=.05)
    p.add_argument('--epsilon-start',type=float,default=1.)
    p.add_argument('--replay-episodes',type=int,default=500)
    p.add_argument('--mixer-hidden',type=int,default=32)
    p.add_argument('--hidden',type=int,default=64)
    p.add_argument('--learning-rate',type=float,default=.0003)
    p.add_argument('--gamma',type=float,default=.99)
    p.add_argument('--target-every',type=int,default=200)
    p.add_argument('--mlflow',action='store_true')
    p.add_argument('--resume',type=Path)
    p.add_argument('--publish-artifact',action=argparse.BooleanOptionalAction,default=True,help='Publish the canonical deployment file; disable for concurrent research runs')
    p.add_argument('--tiny',action='store_true')
    default_config=ROOT/'configs'/'runtime_training.yaml'
    if default_config.exists():p.set_defaults(**yaml.safe_load(default_config.read_text(encoding='utf-8')))
    return p


def main(default_algorithm='qmix'):
    args=parser(default_algorithm).parse_args()
    if min(args.steps,args.horizon,args.eval_episodes,args.train_every,args.threads)<1 or not 1<=args.num_agents<=40 or args.bc_episodes<0 or args.checkpoint_every<0:
        raise ValueError('Invalid training budget/fleet')
    return train(args)


if __name__=='__main__':
    main()
