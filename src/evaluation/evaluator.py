from src.baselines.nearest_task import NearestTaskDispatcher
from src.training.runner import evaluate

def evaluate_policy(policy,num_agents=5,seeds=(300000,300001,300002),episodes=1):
    if policy is None: raise ValueError('Explicit policy required; evaluation cannot silently substitute a fallback')
    if episodes<1:raise ValueError('episodes must be positive')
    # Cartesian IDs avoid overlapping seed+episode arithmetic.
    unique=[int(seed)*10000+episode for seed in seeds for episode in range(episodes)]
    if len(set(unique))!=len(unique):raise ValueError('Duplicate evaluation scenarios')
    return evaluate(None if isinstance(policy,NearestTaskDispatcher) else policy,unique,num_agents)
