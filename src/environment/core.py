from __future__ import annotations
from collections import deque
from copy import deepcopy
import math
import numpy as np
from src.environment.runtime_config import RuntimeConfig
from src.environment.sensors import sensor_frame
from src.environment.spec import (ACTION_DIM, ACTION_HISTORY_LENGTH, ENV_VERSION, MODES,
    OBS_VERSION, SELF_FIELDS, SENSOR_CHANNELS, SENSOR_CONFIG_FIELDS)
from src.environment.logistics import LogisticsMixin
from src.environment.physics import PhysicsMixin
from src.safety.action_mask import orientation_delta


class WarehouseEnv(PhysicsMixin, LogisticsMixin):
    """Atomic warehouse physics and logistics with explicitly local actor sensors."""
    def __init__(self, grid_size=(20,20), num_agents=5, seed=0, obstacles=None,
                 safety_corridors=None, pickup_bays=None, drop_bays=None, staging_cells=None,
                 layout_version=ENV_VERSION, charger_bays=None, transfer_bays=None, config=None):
        self.grid_size=tuple(grid_size)
        if len(self.grid_size)!=2 or any(not isinstance(v,int) or v<3 for v in self.grid_size):
            raise ValueError('Grid must have integer dimensions at least 3x3')
        if not isinstance(num_agents,int) or not 1<=num_agents<=40:
            raise ValueError('Fleet size must be 1..40')
        self.num_agents,self.seed=num_agents,int(seed)
        self.config=config if isinstance(config,RuntimeConfig) else RuntimeConfig(**(config or {}))
        if all(v is None for v in (obstacles,safety_corridors,pickup_bays,drop_bays,staging_cells,charger_bays,transfer_bays)):
            from src.environment.layout import default_layout
            layout=default_layout(self.grid_size)
            obstacles,safety_corridors=layout['obstacles'],layout['safety_corridors']
            pickup_bays,drop_bays=layout['pickup_bays'],layout['drop_bays']
            staging_cells,charger_bays=layout['staging_cells'],layout['charger_bays']
            transfer_bays,layout_version=layout['transfer_bays'],layout['layout_version']
        self.layout_version=layout_version
        self.obstacles={tuple(c) for c in (obstacles or [])}
        self.safety_corridors={tuple(c) for c in (safety_corridors or [])}
        self.pickup_bays=sorted({tuple(c) for c in (pickup_bays or [])})
        self.drop_bays=sorted({tuple(c) for c in (drop_bays or [])})
        self.charger_bays=sorted({tuple(c) for c in (charger_bays or [])})
        self.transfer_bays=sorted({tuple(c) for c in (transfer_bays or [])})
        self.staging_cells=[tuple(c) for c in (staging_cells or [])]
        self.grid=np.zeros((self.grid_size[1],self.grid_size[0]),dtype=np.int8)
        for cells,value in ((self.obstacles,1),(self.safety_corridors,2),(self.pickup_bays,3),(self.drop_bays,4),(self.charger_bays,5),(self.transfer_bays,6)):
            for x,y in cells:
                if not self._is_in_bounds(x,y) or self.grid[y,x]!=0:
                    raise ValueError('Layout cells must be distinct and in bounds')
                self.grid[y,x]=value
        for x,y in self.pickup_bays:
            if not any((x+dx,y+dy) in self.obstacles for dx,dy in ((1,0),(-1,0),(0,1),(0,-1))):
                raise ValueError('Pickup bays must touch a shelf')
        if len(set(self.staging_cells))!=len(self.staging_cells) or any(not self._is_in_bounds(x,y) or self.grid[y,x]!=0 for x,y in self.staging_cells):
            raise ValueError('Staging cells must be unique ordinary floor cells')
        self._base_grid=self.grid.copy()
        self.robots=[]
        self.task_queue=[]
        self.reset()

    @property
    def tick(self):
        return int(self.episode_metrics['step_count'])

    @property
    def observation_dim(self):
        return ((2*self.config.sensor_radius+1)**2*len(SENSOR_CHANNELS)+len(SELF_FIELDS)
                +len(MODES)+len(SENSOR_CONFIG_FIELDS)+ACTION_HISTORY_LENGTH*ACTION_DIM)

    def reset(self, seed=None):
        if seed is not None:
            self.seed=int(seed)
        self.rng=np.random.default_rng(self.seed)
        self.grid=self._base_grid.copy()
        self.task_queue,self.finished_tasks,self.event_log=[],[],[]
        self._ledger,self._events,self._consumed_events,self._faults={},[],set(),{}
        self._paths={}
        self._sensor_history=deque(maxlen=self.config.sensor_latency+2)
        self.completed_tasks=0
        self.episode_metrics={k:0.0 for k in ('collision_count','step_count','reward','safety_overrides','executed_collisions',
            'energy_used','energy_charged','avoidable_depletions','reserve_interventions','backlog_ticks','tardiness_ticks',
            'deadlocks','blocked_ticks','idle_with_assignment_ticks','delivered','cancellations','charger_rejections','infeasible_assignments','executed_safety_violations','charging_ticks','charge_queue_ticks','charge_locked_infeasible_ticks')}
        self._last_progress,self._deadlock_reported=0,False
        self.reward_components={}
        self.chargers=[]
        queue_cells=set()
        for i,(x,y) in enumerate(self.charger_bays):
            cells=[(x+dx,y+dy) for dx,dy in ((-1,0),(0,-1),(1,0),(0,1)) if self._is_in_bounds(x+dx,y+dy) and self.grid[y+dy,x+dx]==0]
            queue_cells.update(cells)
            self.chargers.append(dict(id=i,position=(x,y),online=True,owner=None,queue=[],queue_cells=cells,requested={},services=0))
        starts=[c for c in self.staging_cells if c not in queue_cells]
        if not self.staging_cells:
            starts=[(x,y) for y in range(1,self.grid_size[1]-1) for x in range(1,self.grid_size[0]-1) if self.grid[y,x]==0 and (x,y) not in queue_cells]
        if len(starts)<self.num_agents:
            raise ValueError('Insufficient distinct staging cells for fleet')
        if self.config.shuffle_roles:
            self.rng.shuffle(starts)
        self.robots=[dict(id=i,x=x,y=y,orientation=int(self.rng.integers(4)) if self.config.shuffle_roles else 1,
            current_task=None,target=None,load_status=0,package_id=None,active=True,health='OK',mode='IDLE',
            energy_eu=self.config.capacity*self.config.initial_soc,capacity_eu=self.config.capacity,energy_factor=1.0,
            charger_id=None,service_remaining=0,service_key=None,wait_reason='NO_WORK',lease_tick=0,last_progress_tick=0,parking=(x,y),
            executed_actions=[None]*ACTION_HISTORY_LENGTH)
            for i,(x,y) in enumerate(starts[:self.num_agents])]
        for robot in self.robots:
            if robot['energy_eu']==0:
                robot.update(active=False,mode='DEPLETED')
        self._capture_sensor_snapshot()
        return self.get_global_state()

    def _capture_sensor_snapshot(self):
        self._sensor_history.append((self.tick,self.grid.copy(),deepcopy(self.robots)))

    def get_sensor_frame(self, agent_id):
        if self.config.sensor_latency:
            eligible=[s for s in self._sensor_history if s[0]<=self.tick-self.config.sensor_latency]
            if not eligible:
                n=2*self.config.sensor_radius+1
                return np.zeros((n,n,len(SENSOR_CHANNELS)),dtype=np.float32),self.tick
            stamp,grid,robots=eligible[-1]
        else:
            stamp,grid,robots=self.tick,self.grid,self.robots
        return sensor_frame(grid,robots,agent_id,self.config,self.seed,stamp),self.tick-stamp

    def get_local_observation(self, agent_id):
        robot=self.robots[agent_id]
        channels,age=self.get_sensor_frame(agent_id)
        target=robot['target'] or (robot['x'],robot['y'])
        dx,dy=target[0]-robot['x'],target[1]-robot['y']
        forward=((0,-1),(1,0),(0,1),(-1,0))[robot['orientation']]
        right=(-forward[1],forward[0])
        scale=max(self.grid_size)
        task=self._task(robot['current_task'])
        heading=robot['orientation']*math.pi/2
        # Never infer hidden remote blockages through a ground-truth route estimate.
        # The actor receives only its own energy minus the declared fixed reserve.
        margin=robot['energy_eu']-self.config.reserve
        features=[(dx*right[0]+dy*right[1])/scale,(dx*forward[0]+dy*forward[1])/scale,math.hypot(dx,dy)/math.hypot(*self.grid_size),
            float(robot['load_status']),math.sin(heading),math.cos(heading),robot['energy_eu']/robot['capacity_eu'],robot['capacity_eu']/self.config.capacity,
            robot['energy_factor'],robot['service_remaining']/self.config.service_ticks,
            (self.tick-task['creation_timestep'])/self.config.horizon if task else 0,
            np.clip((task['deadline']-self.tick)/self.config.horizon,-1,1) if task else 0,float(task is not None),
            np.clip(margin/robot['capacity_eu'],-1,1),age/50,float(channels[:,:,0].any()),
            robot['x']/self.grid_size[0],robot['y']/self.grid_size[1]]
        sensor_config=[self.config.front_fov/360,self.config.rear_fov/360,
                       self.config.sensor_dropout,self.config.sensor_latency/50]
        history=np.zeros((ACTION_HISTORY_LENGTH,ACTION_DIM),dtype=np.float32)
        for lag,action in enumerate(robot['executed_actions']):
            if action is not None:history[lag,action]=1
        result=np.concatenate((channels.ravel(),np.array(features+[float(robot['mode']==m) for m in MODES]
            +sensor_config,dtype=np.float32),history.ravel())).astype(np.float32)
        if result.shape!=(self.observation_dim,) or not np.isfinite(result).all():
            raise AssertionError('Observation contract violated')
        return result

    def get_training_state(self):
        values=[]
        for r in self.robots:
            goal=r['target'] or (r['x'],r['y'])
            values.extend([r['x']/self.grid_size[0],r['y']/self.grid_size[1],math.sin(r['orientation']*math.pi/2),math.cos(r['orientation']*math.pi/2),r['load_status'],r['energy_eu']/r['capacity_eu'],float(r['active']),MODES.index(r['mode'])/(len(MODES)-1),goal[0]/self.grid_size[0],goal[1]/self.grid_size[1],r['service_remaining']/self.config.service_ticks,float(r['current_task'] is not None)])
        values.extend((self.grid.astype(np.float32)/6).ravel())
        tasks=sorted(self.task_queue,key=lambda t:t['task_id'])
        for index in range(self.config.max_orders):
            if index>=len(tasks):
                values.extend([0]*10)
                continue
            t=tasks[index]
            values.extend([1,t['pickup'][0]/self.grid_size[0],t['pickup'][1]/self.grid_size[1],t['delivery'][0]/self.grid_size[0],t['delivery'][1]/self.grid_size[1],{'pending':0,'picked':.5,'blocked_recovery':1}.get(t['status'],0),((t['owner']+1)/self.num_agents if t.get('owner') is not None else 0),(self.tick-t['creation_timestep'])/self.config.horizon,np.clip((t['deadline']-self.tick)/self.config.horizon,-1,1),t['priority']/3])
        for c in self.chargers:
            values.extend([c['position'][0]/self.grid_size[0],c['position'][1]/self.grid_size[1],float(c['online']),0 if c['owner'] is None else (c['owner']+1)/self.num_agents,len(c['queue'])/max(1,self.config.queue_capacity)])
        values.extend([self.tick/self.config.horizon,self.completed_tasks/self.config.max_orders,len(self.task_queue)/self.config.max_orders])
        # Present physical parameters and controller timers, never future events.
        health={'OK':0,'STANDBY':1,'IMMOBILIZED':2}
        for r in self.robots:
            values.extend([r['capacity_eu']/self.config.capacity,r['energy_factor'],
                health.get(r['health'],3)/3,r.get('charger_id',None) is not None,
                max(0,self.tick-r.get('charger_approach_tick',self.tick))/self.config.lease_ticks,
                max(0,self.tick-r['last_progress_tick'])/self.config.lease_ticks,
                max(0,r.get('charge_retry_after',self.tick)-self.tick)/self.config.lease_ticks,
                max(0,r.get('lease_excluded_until',self.tick)-self.tick)/self.config.lease_ticks,
                max(0,self.tick-self._faults.get(r['id'],self.tick))/self.config.heartbeat_timeout])
        for c in self.chargers:
            for index in range(self.config.queue_capacity):
                agent=c['queue'][index] if index<len(c['queue']) else None
                values.extend([0 if agent is None else (agent+1)/self.num_agents,
                    0 if agent is None else (self.tick-c['requested'][agent])/self.config.lease_ticks])
        values.extend([self.config.front_fov/360,self.config.rear_fov/360,
                       self.config.sensor_dropout,self.config.sensor_latency/50])
        return np.asarray(values,dtype=np.float32)

    def idle_service_bay_occupants(self):
        bays=set(self.pickup_bays+self.drop_bays+self.charger_bays+self.transfer_bays)
        return sum(r['active'] and r['current_task'] is None and r['package_id'] is None
            and r['charger_id'] is None and (r['x'],r['y']) in bays for r in self.robots)

    def charge_locked_infeasible_agents(self):
        """Count decision states constrained by both assignment and the SOC tier.

        This is not a laziness metric: moving or waiting may both be reasonable.
        Physically unreachable missions do not count as energy-only failures.
        """
        available=[t for t in self.task_queue if t.get('owner') is None
            and t['status']=='pending' and t['creation_timestep']<=self.tick]
        if not self.chargers or not available:return 0
        count=0
        for robot in self.robots:
            if (not robot['active'] or robot['current_task'] is not None or robot['package_id'] is not None
                or robot['charger_id'] is not None
                or robot['energy_eu']/robot['capacity_eu']<=self.config.charge_trigger):continue
            costs=[self.mission_energy(robot,task) for task in available]
            finite=[cost for cost in costs if math.isfinite(cost)]
            count+=bool(finite and min(finite)>robot['energy_eu']+1e-9)
        return count

    def task_potential(self):
        """Bounded remaining-work potential for tasks and assigned support goals."""
        value=0.0
        for r in self.robots:
            if not r['active']:continue
            task=self._task(r['current_task'])
            if task is not None:
                if task['status']=='pending':
                    a=self.path((r['x'],r['y']),task['pickup']);b=self.path(task['pickup'],task['delivery'])
                    handling=r['service_remaining'] if r['service_key'] is not None else self.config.service_ticks
                    distance=(len(a)-1+len(b)-1+handling+self.config.service_ticks) if a and b else math.inf
                else:
                    route=self.path((r['x'],r['y']),task['delivery'])
                    handling=r['service_remaining'] if r['service_key'] is not None else self.config.service_ticks
                    distance=len(route)-1+handling if route else math.inf
                value+=math.exp(-distance/self.config.potential_distance_scale)
            elif r.get('target') is not None and (r['x'],r['y'])!=tuple(r['target']):
                route=self.path((r['x'],r['y']),r['target'])
                distance=len(route)-1 if route else math.inf
                value+=.3*math.exp(-distance/self.config.potential_distance_scale)
            elif r.get('target') is not None:
                value+=.3
        return value

    def assert_invariants(self):
        positions=[(r['x'],r['y']) for r in self.robots]
        if len(set(positions))!=len(positions):
            raise AssertionError('Robot overlap')
        owners=[t['owner'] for t in self.task_queue if t.get('owner') is not None]
        if len(owners)!=len(set(owners)):
            raise AssertionError('Robot owns multiple orders')
        packages=[r['package_id'] for r in self.robots if r['package_id'] is not None]
        if len(packages)!=len(set(packages)):
            raise AssertionError('Duplicated package')
        for r in self.robots:
            if not self._is_in_bounds(r['x'],r['y']) or self.grid[r['y'],r['x']] in (1,2):
                raise AssertionError('Invalid physical robot position')
            if not 0<=r['energy_eu']<=r['capacity_eu']+1e-8:
                raise AssertionError('Invalid energy')
            if bool(r['load_status'])!=(r['package_id'] is not None):
                raise AssertionError('Load/package mismatch')
            task=self._task(r['current_task'])
            if task is not None and task.get('owner')!=r['id']:
                raise AssertionError('Task ownership mismatch')
            if r['package_id'] is not None and (task is None or task['package_location']!=dict(kind='robot',robot_id=r['id'])):
                raise AssertionError('Package ledger mismatch')
            if r['mode'] in ('PICKING','DROPPING') and (r['service_key'] is None or r['service_remaining']<=0):
                raise AssertionError('Stale service mode without active service progress')
        for c in self.chargers:
            ids=c['queue']+([] if c['owner'] is None else [c['owner']])
            if len(ids)!=len(set(ids)) or len(c['queue'])>min(self.config.queue_capacity,len(c['queue_cells'])):
                raise AssertionError('Invalid charger queue')
        for task in self.task_queue:
            owner=task.get('owner')
            if owner is not None and self.robots[owner]['current_task']!=task['task_id']:
                raise AssertionError('Order owner has no matching assignment')
            if task.get('status') in ('picked','blocked_recovery'):
                if owner is None or self.robots[owner]['package_id']!=task['package_id']:
                    raise AssertionError('Picked package has no physical carrier')
        all_ids=[t['task_id'] for t in self.task_queue+self.finished_tasks]
        if len(all_ids)!=len(set(all_ids)):
            raise AssertionError('Order exists in multiple lifecycle states')

    def get_global_state(self):
        return dict(grid_size=self.grid_size,grid=self.grid.tolist(),robots=deepcopy(self.robots),task_queue=deepcopy(self.task_queue),
            metrics=dict(self.episode_metrics),completed_tasks=self.completed_tasks,layout=self.get_layout_spec(),chargers=deepcopy(self.chargers),
            environment_version=ENV_VERSION,observation_version=OBS_VERSION,observation_dim=self.observation_dim,action_dim=ACTION_DIM)

    def get_layout_spec(self):
        return dict(layout_version=self.layout_version,obstacles=[list(c) for c in sorted(self.obstacles)],safety_corridors=[list(c) for c in sorted(self.safety_corridors)],
            pickup_bays=[list(c) for c in self.pickup_bays],drop_bays=[list(c) for c in self.drop_bays],charger_bays=[list(c) for c in self.charger_bays],
            transfer_bays=[list(c) for c in self.transfer_bays],staging_cells=[list(c) for c in self.staging_cells])

    def _is_in_bounds(self,x,y):
        return 0<=x<self.grid_size[0] and 0<=y<self.grid_size[1]

    def _action_delta(self,orientation,action):
        return orientation_delta(orientation,action)

    def render(self):
        grid=self.grid.copy()
        for r in self.robots:
            grid[r['y'],r['x']]=10+r['id']
        return '\n'.join(' '.join(str(int(c)) for c in row) for row in grid)
