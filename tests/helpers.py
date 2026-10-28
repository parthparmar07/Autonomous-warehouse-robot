from src.environment.warehouse_env import WarehouseEnv
from src.environment.runtime_config import RuntimeConfig


def picked(n=1,service=3,**kwargs):
    env=WarehouseEnv(grid_size=(20,20),num_agents=n,obstacles=[],safety_corridors=[],
        config=RuntimeConfig(service_ticks=service,**kwargs))
    env.task_queue=[dict(task_id=0,pickup=(1,1),delivery=(12,12),status='pending')]
    env.prepare_decision()
    for _ in range(service):env.step([5]+[4]*(n-1))
    assert env.robots[0]['package_id']==0
    return env
