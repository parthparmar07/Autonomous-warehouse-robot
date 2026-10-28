from __future__ import annotations
import hashlib
from pathlib import Path
import torch
from src.baselines.nearest_task import NearestTaskDispatcher
from src.environment.spec import ENV_VERSION, OBS_VERSION
from src.environment.warehouse_env import WarehouseEnv
from src.training.learner import RecurrentLearner

ROOT=Path(__file__).resolve().parents[2]
_ARTIFACT_DIR=ROOT/'artifacts'/ENV_VERSION
ALGORITHM_DESCRIPTIONS={
 'shortest_path':'Privileged static shortest paths without cooperative reservations; same allocator and physics.',
 'nearest_task':'Energy-aware allocator and privileged-map reservation planner.',
 'iql':'Shared recurrent independent Q-learning; same team objective and actor as VDN/QMIX.',
 'vdn':'Shared recurrent utilities with additive team value decomposition.',
 'qmix':'Shared recurrent utilities with a monotonic state-conditioned team mixer.',
 'dqn':'Compatibility name for shared recurrent independent Q-learning.'}

def specific_checkpoint(name,num_agents):
    return _ARTIFACT_DIR/f'{name}_policy_N{num_agents}.pt'

def generic_checkpoint(name):
    return _ARTIFACT_DIR/f'{name}_policy.pt'

def resolve_checkpoint(name,num_agents):
    path=specific_checkpoint(name,num_agents)
    return path if path.is_file() else None

def checkpoint_exists(name,num_agents=5):
    return resolve_checkpoint(name,num_agents) is not None

def save_checkpoint(payload,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(payload,path)
    path.with_suffix(path.suffix+'.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest(),encoding='ascii')

def load_policy(name,num_agents,env=None,checkpoint=None):
    env=env or WarehouseEnv(num_agents=num_agents)
    if name in ('nearest_task','shortest_path'):
        return NearestTaskDispatcher(env,cooperative=name=='nearest_task')
    if name not in ALGORITHM_DESCRIPTIONS:
        raise ValueError(f'Unknown algorithm: {name}')
    path=Path(checkpoint) if checkpoint is not None else resolve_checkpoint(name,num_agents)
    if path is None or not path.is_file():
        raise FileNotFoundError(f'No compatible {name} checkpoint for N={num_agents}; train the current environment first')
    digest_path=path.with_suffix(path.suffix+'.sha256')
    if not digest_path.is_file() or digest_path.read_text().strip()!=hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError('Checkpoint integrity hash is missing or mismatched')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('num_agents')!=num_agents or payload.get('obs_dim')!=env.observation_dim or payload.get('state_dim')!=len(env.get_training_state()):
        raise ValueError('Checkpoint dimensions/fleet are incompatible; automatic network cycling is disabled')
    if payload.get('algorithm')!=name:
        raise ValueError('Checkpoint algorithm mismatch')
    model=RecurrentLearner.from_checkpoint(payload)
    model.metadata=payload.get('metadata',{})
    model.epsilon=0
    return model

def describe_algorithms(num_agents=5):
    return [dict(name=name,trained=name in ('nearest_task','shortest_path') or checkpoint_exists(name,num_agents),
                 description=description,environment_version=ENV_VERSION,observation_version=OBS_VERSION,
                 fleet_size=num_agents) for name,description in ALGORITHM_DESCRIPTIONS.items()]
