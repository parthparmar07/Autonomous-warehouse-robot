from __future__ import annotations
import os
import re
import time
import torch
from threading import RLock
from contextlib import contextmanager
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.baselines.nearest_task import NearestTaskDispatcher
from src.deployment.inference import inference_source, predict_actions
from src.deployment.policies import describe_algorithms, load_policy
from src.deployment.schemas import ObservationRequest, PredictResponse, ResetRequest, StepRequest
from src.environment.runtime_config import RuntimeConfig
from src.environment.spec import SENSOR_CHANNELS, OBS_VERSION
from src.environment.task_generator import generate_tasks
from src.environment.warehouse_env import WarehouseEnv

# Small recurrent actors incur severe overhead with large CPU thread pools.
torch.set_num_threads(max(1,int(os.getenv('WAREHOUSE_CPU_THREADS','1'))))

app = FastAPI(title='Warehouse Fleet API', version='1.0.0')
app.add_middleware(CORSMiddleware, allow_origins=os.getenv('WAREHOUSE_CORS_ORIGINS',
    'http://127.0.0.1:8080,http://localhost:8080').split(','), allow_credentials=False,
    allow_methods=['GET','POST','DELETE'], allow_headers=['Content-Type'])
_DEFAULT_SESSION = 'default'
MAX_SESSIONS = 32
SESSION_TTL = 1800
_registry_lock = RLock()
_sessions: dict[str, Any] = {}

class _Session:
    def __init__(self, payload):
        self.lock = RLock()
        self.last_access = time.monotonic()
        self.algorithm = payload.algorithm
        self.num_agents, self.num_tasks, self.seed = payload.num_agents, payload.num_tasks, payload.seed
        self.workload = payload.workload
        self._task_batch = 0
        config = RuntimeConfig(horizon=1000000, finite_batch=False, initial_soc=payload.initial_soc,
            front_fov=payload.sensor_front_fov, rear_fov=payload.sensor_rear_fov,
            sensor_dropout=payload.sensor_dropout, sensor_latency=payload.sensor_latency)
        if self.workload=='mixed':
            from src.environment.mixed_logistics import make_mixed_scenario
            self.env,_=make_mixed_scenario(payload.seed,payload.arrival_interval,600,payload.num_agents,'demo',
                dict(initial_soc=payload.initial_soc,front_fov=payload.sensor_front_fov,rear_fov=payload.sensor_rear_fov,
                     sensor_dropout=payload.sensor_dropout,sensor_latency=payload.sensor_latency))
        else:
            self.env = WarehouseEnv(num_agents=payload.num_agents, seed=payload.seed, config=config)
            self.env.task_queue = generate_tasks(self.env, payload.num_tasks, payload.seed)
        self.env.prepare_decision()
        self.env._sensor_history.clear()
        self.env._capture_sensor_snapshot()
        self.dispatcher = NearestTaskDispatcher(self.env)
        checkpoint=None
        if self.workload=='mixed' and self.algorithm not in ('nearest_task','shortest_path'):
            from src.deployment.policies import ROOT
            from src.environment.spec import ENV_VERSION
            checkpoint=ROOT/'artifacts'/ENV_VERSION/'mixed_stock_v1'/f'{self.algorithm}_policy_N{self.num_agents}.pt'
        self.policy = load_policy(self.algorithm, self.num_agents, env=self.env,checkpoint=checkpoint)
        if hasattr(self.policy,'reset_hidden_states'): self.policy.reset_hidden_states()

    def state(self):
        package_counts={}
        if self.workload=='mixed':
            package_counts=dict(stored=sum(s['state']=='shelf' for s in self.env.inventory.values()),
                shipped=sum(s['state']=='shipped' for s in self.env.inventory.values()))
        return dict(self.env.get_global_state(), package_counts=package_counts, algorithm=self.algorithm, demo=True,
            policy_status='privileged map planner' if self.algorithm in ('nearest_task','shortest_path') else self.policy.metadata)

    def advance(self, actions=None):
        self.env.prepare_decision()
        if actions is None:
            actions = self.policy.act(self.env) if self.algorithm in ('nearest_task','shortest_path') else self.policy.act(
                [self.env.get_local_observation(i) for i in range(self.num_agents)],
                [self.env.get_action_mask(i) for i in range(self.num_agents)])
        result = self.env.step(actions)
        if self.workload=='outbound' and not self.env.task_queue:
            self._task_batch += 1
            self.env.task_queue = generate_tasks(self.env, self.num_tasks, self.seed+self._task_batch,
                start_task_id=self._task_batch*self.num_tasks)
            self.env.prepare_decision()
        return dict(self.state(), actions=actions, reward=result['reward'],
                    done=result['truncated'], terminated=False, truncated=result['truncated'])

def _validate_id(session_id):
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}',session_id):
        raise HTTPException(422,'Session ID must contain 1..80 letters, digits, underscores or hyphens')

def _get_session(session_id):
    _validate_id(session_id)
    with _registry_lock:
        now=time.monotonic()
        expired=[key for key,s in _sessions.items() if now-s.last_access>SESSION_TTL]
        for key in expired: _sessions.pop(key,None)
        session=_sessions.get(session_id)
        if session is None: raise HTTPException(404,'Session missing or expired; call /reset')
        session.last_access=now
        return session

@contextmanager
def _locked_session(session_id):
    # Lookup and lock acquisition are atomic with reset/delete. A waiting tick
    # cannot accidentally operate on a detached, replaced session.
    with _registry_lock:
        session=_get_session(session_id)
        session.lock.acquire()
    try:yield session
    finally:session.lock.release()

@app.get('/health')
def health(): return dict(status='ok',session_storage='single-process; run one worker')

@app.get('/algorithms')
def algorithms(num_agents:int=5):
    if not 1<=num_agents<=40: raise HTTPException(422,'Fleet must be 1..40')
    return describe_algorithms(num_agents)

@app.post('/predict',response_model=PredictResponse)
def predict(payload:ObservationRequest):
    try:
        actions=predict_actions(payload.observations,payload.masks,payload.algorithm)
    except (FileNotFoundError,ValueError) as exc: raise HTTPException(409,str(exc)) from exc
    return PredictResponse(actions=actions,message=inference_source())

@app.post('/reset')
def reset(payload:ResetRequest,session_id:str=_DEFAULT_SESSION):
    _validate_id(session_id)
    # Construct first: a missing/incompatible checkpoint never damages a live session.
    try: replacement=_Session(payload)
    except (FileNotFoundError,ValueError) as exc: raise HTTPException(409,str(exc)) from exc
    with _registry_lock:
        now=time.monotonic()
        for key in [k for k,s in _sessions.items() if now-s.last_access>SESSION_TTL]: _sessions.pop(key,None)
        if session_id not in _sessions and len(_sessions)>=MAX_SESSIONS: raise HTTPException(429,'Session capacity reached')
        previous=_sessions.get(session_id)
        if previous:
            with previous.lock: _sessions[session_id]=replacement
        else: _sessions[session_id]=replacement
    return dict(replacement.state(),session_id=session_id)

@app.get('/state')
def state(session_id:str=_DEFAULT_SESSION):
    with _locked_session(session_id) as session:return session.state()

@app.get('/sensor')
def sensor(agent_id:int=0,session_id:str=_DEFAULT_SESSION):
    with _locked_session(session_id) as session:
        if not 0<=agent_id<session.num_agents: raise HTTPException(422,'Invalid robot ID')
        frame,age=session.env.get_sensor_frame(agent_id)
        robot=session.env.robots[agent_id]
        return dict(agent_id=agent_id,frame=frame.tolist(),channels=SENSOR_CHANNELS,age_ticks=age,
            version=OBS_VERSION,heading=robot['orientation'],origin=[robot['x'],robot['y']],
            observation=session.env.get_local_observation(agent_id).tolist(),
            action_mask=session.env.get_action_mask(agent_id))

@app.post('/tick')
def tick(session_id:str=_DEFAULT_SESSION):
    with _locked_session(session_id) as session:return session.advance()

@app.post('/step')
def step(payload:StepRequest,session_id:str=_DEFAULT_SESSION):
    with _locked_session(session_id) as session:
        if len(payload.actions)!=session.num_agents: raise HTTPException(422,f'Expected {session.num_agents} actions')
        return session.advance(payload.actions)

@app.delete('/session')
def delete_session(session_id:str):
    _validate_id(session_id)
    with _registry_lock:
        session=_sessions.get(session_id)
        if session:
            with session.lock: _sessions.pop(session_id,None)
    return dict(deleted=session_id)
