"""Materialized seeded environments, immutable event streams and role randomization."""
from dataclasses import replace
import math
import numpy as np
from src.environment.layout import default_layout
from src.environment.runtime_config import RuntimeConfig
from src.environment.warehouse_env import WarehouseEnv
from src.environment.task_generator import generate_tasks, schedule_order
from src.utils.hashing import sha256_digest
from src.evaluation.curriculum import STAGES


def make_scenario(seed, num_agents=5, stage='C5', split='train', horizon=400, num_tasks=None, tiny=False,role_index=None,scaling='fixed_map',profile='robust',runtime_overrides=None):
    if stage not in STAGES:raise ValueError('Unknown curriculum stage')
    if scaling not in ('fixed_map','area_per_robot'):raise ValueError('Unknown scaling study')
    if profile not in ('core','robust'):raise ValueError('Unknown environment profile')
    streams={name:int(sha256_digest([int(seed),name])[:8],16) for name in ('layout','demand','energy','faults','sensor','roles')}
    rng=np.random.default_rng(streams['energy'])
    if tiny:
        config=RuntimeConfig(horizon=horizon,service_ticks=1,max_orders=8)
        env=WarehouseEnv(grid_size=(7,7),num_agents=num_agents,obstacles=[(2,2)],safety_corridors=[],
                         pickup_bays=[(2,1)],drop_bays=[(4,1)],config=config)
        env.task_queue=generate_tasks(env,num_tasks or 1,seed=streams['demand'])
    else:
        config=RuntimeConfig(horizon=horizon,shuffle_roles=True)
        if profile=='robust' and stage in ('C4','C5'):
            sensor_rng=np.random.default_rng(streams['sensor'])
            config=replace(config,sensor_latency=int(sensor_rng.integers(0,3)),sensor_dropout=float(sensor_rng.choice([0,.05])),
                front_fov=float(sensor_rng.choice([90,120,360])),rear_fov=float(sensor_rng.choice([0,90])))
        side=20 if scaling=='fixed_map' else max(12,math.ceil(20*math.sqrt(num_agents/5)))
        layout=default_layout((side,side))
        if stage=='C0':
            layout.update(obstacles=[(5,5)],pickup_bays=[(5,4)],drop_bays=[(9,4)])
            layout['staging_cells']=[c for c in layout['staging_cells'] if tuple(c) not in {(5,5),(5,4),(9,4)}]
        if split=='test_ood':
            for key in ('obstacles','safety_corridors','pickup_bays','drop_bays','charger_bays','transfer_bays','staging_cells'):
                layout[key]=[(side-1-x,y) for x,y in layout[key]]
        env=WarehouseEnv(grid_size=(side,side),num_agents=num_agents,seed=streams['roles'],config=config,**layout)
        default_tasks=1 if stage=='C0' else 2 if stage=='C0.5' else min(2,num_agents)*2 if stage=='C0.75' else num_agents*2
        env.task_queue=generate_tasks(env,num_tasks or default_tasks,seed=streams['demand'])
        worker=(int(seed) if role_index is None else role_index)%num_agents
        if stage=='C0':
            previous=(env.robots[worker]['x'],env.robots[worker]['y'])
            for robot in env.robots:
                if robot['id']!=worker and (robot['x'],robot['y'])==(4,4):robot.update(x=previous[0],y=previous[1],parking=previous)
            env.robots[worker].update(x=4,y=4,parking=(4,4),orientation=((int(seed) if role_index is None else role_index)//num_agents)%4)
        if stage in ('C0','C0.5','C0.75'):
            workers=min(2,num_agents) if stage=='C0.75' else 1
            active_ids={(worker+offset)%num_agents for offset in range(workers)}
            for robot in env.robots:
                if robot['id'] not in active_ids:robot.update(active=False,health='STANDBY',mode='FAULTED')
            env.robots[worker]['orientation']=((int(seed) if role_index is None else role_index)//num_agents)%4
        if stage in ('C2','C3','C4','C5'):
            for robot in env.robots:
                robot['capacity_eu']=float(rng.uniform(.7,1.1)*config.capacity) if profile=='robust' else config.capacity
                robot['energy_eu']=float(rng.uniform(.15,1) if profile=='robust' else rng.uniform(.2,.65))*robot['capacity_eu']
                robot['energy_factor']=float(rng.uniform(.8,1.2)) if profile=='robust' else 1.0
        if role_index is not None:
            worker=role_index%num_agents
            env.robots[worker]['orientation']=(role_index//num_agents)%4
        if profile=='robust' and stage in ('C3','C4','C5') and num_agents>1 and horizon>=15:
            fault_rng=np.random.default_rng(streams['faults'])
            when=int(fault_rng.integers(10,max(11,horizon//3)))
            victim=int(fault_rng.integers(num_agents)) if role_index is None else role_index%num_agents
            events=[dict(id='robot-fault',type='fault',tick=when,robot_id=victim),
                    dict(id='robot-repair',type='repair',tick=min(horizon-1,when+30),robot_id=victim)]
            if stage in ('C4','C5'):
                # A disjoint ID range, arrivals independent of deliveries or policy speed.
                for index,task in enumerate(generate_tasks(env,num_agents,seed=streams['demand']+1,start_task_id=1000)):
                    release=when+index*2
                    if release<horizon:
                        task=schedule_order(task,release)
                        events.append(dict(id=f'arrival-{index}',type='order',tick=release,order=task))
                if split=='test_ood' and env.chargers:
                    events.extend([dict(id='outage',type='charger_outage',tick=when,charger_id=0),
                                   dict(id='outage-repair',type='charger_online',tick=min(horizon-1,when+50),charger_id=0)])
            env.set_events(events)
    if runtime_overrides:
        env.config=replace(env.config,**runtime_overrides)
    env.prepare_decision()
    env._sensor_history.clear()
    env._capture_sensor_snapshot()
    manifest=dict(seed=int(seed),num_agents=num_agents,stage=stage,split=split,streams=streams,scaling=scaling,profile=profile,
                  layout=env.get_layout_spec(),config=env.config.to_dict(),robots=env.get_global_state()['robots'],
                  tasks=env.get_global_state()['task_queue'],events=env._events)
    manifest['scenario_hash']=sha256_digest(manifest)
    return env,manifest
