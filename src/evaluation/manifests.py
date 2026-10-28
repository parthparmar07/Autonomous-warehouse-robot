from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
from src.environment.spec import ENV_VERSION, OBS_VERSION
from src.environment.warehouse_env import WarehouseEnv
from src.environment.task_generator import generate_tasks
from src.evaluation.scenarios import make_scenario
from src.utils.hashing import sha256_digest
MANIFEST_DIR=Path('evaluation/manifests')/ENV_VERSION/'development'
SEED_RANGES={'train':(100000,199999),'development':(300000,399999),'test':(500000,599999),'test_ood':(700000,799999)}

def capture_manifest(env, seed, split='development', stage='C5'):
    if env.tick!=0: raise ValueError('Only initial decision states can be sealed')
    env.prepare_decision()
    manifest=dict(schema_version=3,environment_version=ENV_VERSION,observation_version=OBS_VERSION,
        layout=env.get_layout_spec(),grid_size=list(env.grid_size),num_agents=env.num_agents,
        seed=seed,environment_seed=env.seed,split=split,stage=stage,config=env.config.to_dict(),
        num_tasks=len(env.task_queue),tasks=deepcopy(env.task_queue),robot_initial_state=deepcopy(env.robots),
        chargers=deepcopy(env.chargers),events=deepcopy(env._events),
        initial_events=deepcopy(env.event_log),consumed_event_ids=sorted(env._consumed_events,key=str),faults=deepcopy(env._faults),
        metrics=deepcopy(env.episode_metrics),manifest_hash='')
    if hasattr(env,'inventory'):
        manifest.update(workload_version=env.workload_version,inventory=deepcopy(env.inventory),shelf_capacity=env.shelf_capacity)
    manifest=json.loads(json.dumps(manifest))
    manifest['manifest_hash']=sha256_digest(manifest)
    return manifest

def generate_evaluation_manifest(num_agents,seed,grid_size=(20,20),num_tasks=20,split='development',stage='C5'):
    if tuple(grid_size)==(20,20):
        env,_=make_scenario(seed,num_agents,stage,split,num_tasks=num_tasks)
    else:
        env=WarehouseEnv(grid_size=grid_size,num_agents=num_agents,seed=seed)
        env.task_queue=generate_tasks(env,num_tasks,seed)
    return capture_manifest(env,seed,split,stage)

def verify_manifest_hash(manifest):
    check=dict(manifest);check['manifest_hash']=''
    return bool(manifest.get('manifest_hash')) and sha256_digest(check)==manifest['manifest_hash']

def environment_from_manifest(manifest):
    if not verify_manifest_hash(manifest): raise ValueError('Manifest hash mismatch')
    if manifest.get('schema_version')!=3 or 'layout' not in manifest:
        raise ValueError('Legacy manifest: generate separate warehouse_v3_energy scenarios')
    if manifest.get('environment_version')!=ENV_VERSION or manifest.get('observation_version')!=OBS_VERSION:
        raise ValueError('Unsupported environment or sensor version')
    if manifest.get('workload_version')=='mixed_stock_v1':
        from src.environment.mixed_logistics import MixedWarehouseEnv
        env=MixedWarehouseEnv(num_agents=manifest['num_agents'],seed=manifest['environment_seed'],
            config=manifest['config'],shelf_capacity=manifest['shelf_capacity'])
        if json.loads(json.dumps(env.get_layout_spec()))!=manifest['layout']:
            raise ValueError('Mixed workload requires its fixed receiving/storage layout')
        env.inventory={int(k):deepcopy(v) for k,v in manifest['inventory'].items()}
        for stock in env.inventory.values():stock['storage_bay']=tuple(stock['storage_bay'])
    else:
        env=WarehouseEnv(grid_size=tuple(manifest['grid_size']),num_agents=manifest['num_agents'],
            seed=manifest['environment_seed'],config=manifest['config'],**manifest['layout'])
    if len(manifest['tasks'])!=manifest['num_tasks']: raise ValueError('Invalid manifest task count')
    env.robots=deepcopy(manifest['robot_initial_state'])
    for r in env.robots:
        r['parking']=tuple(r['parking'])
        if r['target'] is not None:r['target']=tuple(r['target'])
        if r['service_key'] is not None:r['service_key']=tuple(r['service_key'])
    env.task_queue=deepcopy(manifest['tasks'])
    for task in env.task_queue:
        for key in ('pickup','delivery','shelf'):
            if key in task:task[key]=tuple(task[key])
    env._ledger={t['task_id']:t for t in env.task_queue}
    env.chargers=deepcopy(manifest['chargers'])
    for c in env.chargers:
        c['position']=tuple(c['position']);c['queue_cells']=[tuple(p) for p in c['queue_cells']]
        c['requested']={int(k):v for k,v in c['requested'].items()}
    env._events=deepcopy(manifest['events'])
    env._consumed_events=set(manifest.get('consumed_event_ids',[]))
    env._faults={int(k):v for k,v in manifest.get('faults',{}).items()}
    env.event_log=deepcopy(manifest.get('initial_events',[]))
    env.episode_metrics.update(manifest.get('metrics',{}))
    env.prepare_decision()
    env._sensor_history.clear();env._capture_sensor_snapshot();env.assert_invariants()
    return env

def write_manifest(path,manifest):
    if not verify_manifest_hash(manifest): raise ValueError('Cannot write invalid manifest')
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
    content=json.dumps(manifest,indent=2)
    if target.exists() and target.read_text(encoding='utf-8')!=content:
        raise FileExistsError(f'Immutable manifest already exists: {target}')
    target.write_text(content,encoding='utf-8')
    return manifest['manifest_hash']

def generate_manifest_set(output_dir=MANIFEST_DIR,split='development',count=20):
    if split not in SEED_RANGES or count<1 or count>10000:raise ValueError('Invalid partition/count')
    start=SEED_RANGES[split][0]
    paths=[]
    for n in (5,10,20):
        for seed in range(start,start+count):
            target=Path(output_dir)/f'eval_N{n}_seed{seed}.json'
            write_manifest(target,generate_evaluation_manifest(n,seed,num_tasks=2*n,split=split))
            paths.append(str(target))
    return paths
