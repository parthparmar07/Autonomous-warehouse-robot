import math
from typing import Literal
from pydantic import BaseModel, Field, StrictInt, StrictBool, model_validator
from src.environment.spec import ACTION_DIM, OBS_DIM, OBS_VERSION
ALGORITHM_NAMES=('nearest_task','dqn','iql','vdn','qmix','shortest_path')

class ObservationRequest(BaseModel):
    observations:list[list[float]]=Field(min_length=1,max_length=40)
    masks:list[list[StrictBool]]
    algorithm:str='qmix'
    observation_version:str=OBS_VERSION
    @model_validator(mode='after')
    def validate_batch(self):
        if self.algorithm not in ('dqn','iql','vdn','qmix') or self.observation_version!=OBS_VERSION:
            raise ValueError('Incompatible algorithm/observation version')
        if any(len(o)!=OBS_DIM or not all(math.isfinite(v) for v in o) for o in self.observations):
            raise ValueError(f'Each observation must contain {OBS_DIM} finite features')
        if len(self.masks)!=len(self.observations) or any(len(m)!=ACTION_DIM or not any(m) for m in self.masks):
            raise ValueError('Each observation requires a seven-action mask with a legal action')
        return self
class PredictResponse(BaseModel):
    actions:list[int]
    message:str='versioned checkpoint inference'
class StepRequest(BaseModel):
    actions:list[StrictInt]=Field(min_length=1,max_length=40)
    @model_validator(mode='after')
    def validate_actions(self):
        if any(a not in range(ACTION_DIM) for a in self.actions):raise ValueError('Actions must be in 0..6')
        return self
class ResetRequest(BaseModel):
    workload:Literal['outbound','mixed']='outbound'
    arrival_interval:StrictInt=Field(default=20,ge=8,le=400)
    algorithm:str='nearest_task'
    num_agents:StrictInt=Field(default=5,ge=1,le=40)
    num_tasks:StrictInt=Field(default=10,ge=1,le=128)
    seed:StrictInt=Field(default=42,ge=0,le=2**32-1)
    initial_soc:float=Field(default=1,ge=0,le=1)
    sensor_front_fov:float=Field(default=360,ge=0,le=360)
    sensor_rear_fov:float=Field(default=0,ge=0,le=360)
    sensor_dropout:float=Field(default=0,ge=0,le=1)
    sensor_latency:StrictInt=Field(default=0,ge=0,le=50)
    @model_validator(mode='after')
    def validate_algorithm(self):
        if self.algorithm not in ALGORITHM_NAMES:raise ValueError('Unknown algorithm')
        return self
class AlgorithmInfo(BaseModel):
    name:str
    trained:bool
    description:str
