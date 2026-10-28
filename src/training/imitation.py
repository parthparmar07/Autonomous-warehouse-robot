"""Sequence-aware expert datasets; flat zero-hidden imitation is intentionally retired."""
import numpy as np
from src.evaluation.scenarios import make_scenario
from src.baselines.nearest_task import NearestTaskDispatcher

def collect_expert_episodes(num_agents=5,episodes=20,max_steps=400,seed_start=120000,stage='C5'):
    from src.training.runner import collect_episode
    data=[]
    for index in range(episodes):
        env,manifest=make_scenario(seed_start+index,num_agents,stage,horizon=max_steps,role_index=index)
        episode,_=collect_episode(env,NearestTaskDispatcher(env),manifest)
        data.append(episode)
    return data

def collect_dagger_episodes(model,episodes=20,max_steps=400,seed_start=130000,stage='C5'):
    from src.training.runner import collect_episode
    data=[]
    for index in range(episodes):
        env,manifest=make_scenario(seed_start+index,model.num_agents,stage,horizon=max_steps,role_index=index)
        episode,_=collect_episode(env,model,manifest,expert_labels=True)
        data.append(episode)
    return data

def behavior_clone(model,episodes,epochs=3):return model.behavior_clone(episodes,epochs)

def class_weights_for(actions,action_dim=7):
    counts=np.bincount(np.asarray(actions).ravel().astype(int),minlength=action_dim)
    weights=1/np.maximum(counts,1)
    return weights/weights.mean()
