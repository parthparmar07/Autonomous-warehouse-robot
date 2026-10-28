"""Compatibility name for the matched shared recurrent independent learner."""
from src.algorithms.qmix.trainer import QMIXTrainer
from src.training.learner import RecurrentLearner
from src.environment.warehouse_env import WarehouseEnv

class IQLTrainer(RecurrentLearner):
    def __init__(self,num_agents=5,input_dim=None,state_dim=None,**kwargs):
        env=WarehouseEnv(num_agents=num_agents)
        super().__init__('iql',num_agents,input_dim or env.observation_dim,state_dim or len(env.get_training_state()),**kwargs)
