"""Compatibility import for the verified shared recurrent QMIX learner."""
from src.training.learner import RecurrentLearner
from src.environment.warehouse_env import WarehouseEnv

class QMIXTrainer(RecurrentLearner):
    def __init__(self,num_agents=5,input_dim=None,action_dim=7,state_dim=None,**kwargs):
        env=WarehouseEnv(num_agents=num_agents)
        super().__init__('qmix',num_agents,input_dim or env.observation_dim,state_dim or len(env.get_training_state()),action_dim,**kwargs)
