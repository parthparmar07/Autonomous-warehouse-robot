"""Gymnasium-space parallel multi-agent adapter with team reward and explicit masks.

Keeps physical robots in the agent list during repairable faults. This is a
parallel interface, not a claim of full PettingZoo API certification.
"""
import numpy as np
from src.environment.warehouse_env import WarehouseEnv
from src.environment.spec import ENV_VERSION

class WarehouseParallelEnv:
    metadata={'name':ENV_VERSION,'is_parallelizable':True}
    def __init__(self,**kwargs):
        from gymnasium import spaces
        self.env=WarehouseEnv(**kwargs)
        self.possible_agents=[f'robot_{i}' for i in range(self.env.num_agents)]
        self.agents=list(self.possible_agents)
        self.observation_spaces={a:spaces.Dict(dict(observation=spaces.Box(-np.inf,np.inf,(self.env.observation_dim,),np.float32),action_mask=spaces.MultiBinary(7))) for a in self.agents}
        self.action_spaces={a:spaces.Discrete(7) for a in self.agents}
    def observation_space(self,agent):return self.observation_spaces[agent]
    def action_space(self,agent):return self.action_spaces[agent]
    def _observations(self):
        return {a:dict(observation=self.env.get_local_observation(i),action_mask=np.asarray(self.env.get_action_mask(i),dtype=np.int8)) for i,a in enumerate(self.possible_agents)}
    def reset(self,seed=None,options=None):
        from src.environment.task_generator import generate_tasks
        self.env.reset(seed=seed)
        if options and options.get('num_tasks'):
            self.env.task_queue=generate_tasks(self.env,options['num_tasks'],self.env.seed)
        self.env.prepare_decision();self.env._sensor_history.clear();self.env._capture_sensor_snapshot()
        self.agents=list(self.possible_agents)
        return self._observations(),{a:{} for a in self.agents}
    def step(self,actions):
        if set(actions)!=set(self.agents) or not self.agents:raise ValueError('One action per current agent is required')
        names=list(self.agents)
        result=self.env.step([actions[a] for a in names])
        observations=self._observations()
        rewards={a:result['reward'] for a in names}
        terminations={a:result['terminated'] for a in names}
        truncations={a:result['truncated'] for a in names}
        infos={a:dict(metrics=result['metrics']) for a in names}
        if result['done']:self.agents=[]
        return observations,rewards,terminations,truncations,infos
    def state(self):return self.env.get_training_state()
    def close(self):self.agents=[]
