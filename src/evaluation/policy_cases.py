"""Executable skill scenarios. Passing simulator checks does not imply these pass.

Materializers use the same 20x20/2-charger critic shape as the standard fleet.
They expose current task/energy data, never the scheduled future events.
"""
from copy import deepcopy
from dataclasses import replace
from src.environment.core import WarehouseEnv
from src.environment.runtime_config import RuntimeConfig
from src.environment.task_generator import generate_tasks, schedule_order
from src.evaluation.scenarios import make_scenario
from src.evaluation.manifests import capture_manifest

POLICY_CASES=('HND06','HND_TRANSFER','COO01','COO02','COO04','COO05','COO06','CHARGE_CYCLE','CHARGE_APPROACH','DEM02','DEM03','ROB01','ROB03','ROB04','ROB05','EVA05')

def materialize(case,seed,n=5,horizon=600,role=0):
    if case not in POLICY_CASES:raise ValueError('Unknown skill case')
    if n<2:raise ValueError('Skill cases require at least two physical robots')
    if case=='EVA05':horizon=max(horizon,3000)
    if case=='COO01' and n>18:raise ValueError('Passing-bay fixture supports up to eighteen physical slots')
    if case=='COO01':
        worker=role%n;relief=(worker+1)%n
        free={(x,10) for x in range(2,18)}|{(9,9),(2,9),(17,9),(2,8),(17,8)}|{(x,2) for x in range(1,n+1)}
        obstacles=[(x,y) for y in range(1,19) for x in range(1,19) if (x,y) not in free]
        env=WarehouseEnv(num_agents=n,obstacles=obstacles,safety_corridors=[],pickup_bays=[(3,10),(16,10)],
            drop_bays=[(4,10),(15,10)],charger_bays=[(2,8),(17,8)],staging_cells=[(x,2) for x in range(1,n+1)],config=RuntimeConfig(horizon=horizon))
        for i,r in enumerate(env.robots):
            active=i in (worker,relief)
            r.update(active=active,health='OK' if active else 'STANDBY',mode='IDLE' if active else 'FAULTED')
        env.robots[worker].update(x=3,y=10,parking=(2,10),orientation=(role//n)%4)
        env.robots[relief].update(x=16,y=10,parking=(17,10),orientation=((role//n)+2)%4)
        # Already carried cargo creates opposing flow across a real passing pocket.
        for index,(i,goal) in enumerate(zip((worker,relief),((16,10),(3,10)))):
            order=dict(task_id=i,pickup=(3,10) if index==0 else (16,10),delivery=goal,status='picked',owner=i,
                creation_timestep=0,deadline=horizon,priority=1,package_id=i,picked_tick=0,package_location=dict(kind='robot',robot_id=i))
            env.task_queue.append(order)
            env.robots[i].update(current_task=i,package_id=i,load_status=1,target=goal,mode='TO_DROP')
    else:
        env,_=make_scenario(seed,n,'C1','development',horizon)
        env.task_queue=generate_tasks(env,n*2,seed)
        # Rebuild assignments after changing the materialized initial state.
        env._ledger={};env.event_log=[]
        for r in env.robots:r.update(current_task=None,target=None,load_status=0,package_id=None,service_key=None,service_remaining=0)
        events=[];worker=role%n;relief=(worker+1)%n
        env.robots[worker]['orientation']=(role//n)%4
        if case in ('CHARGE_CYCLE','CHARGE_APPROACH'):
            later=env.task_queue[:2];env.task_queue=[]
            for r in env.robots:
                if r['id']!=worker:r.update(active=False,health='STANDBY',mode='FAULTED')
            dock=env.chargers[role%len(env.chargers)]['position']
            # Approach is a distinct prerequisite: the policy must request,
            # navigate to the reservation and dock before it can earn delivery.
            start=(dock[0],dock[1]+4 if role%len(env.chargers)==0 else dock[1]-4) if case=='CHARGE_APPROACH' else dock
            env.robots[worker].update(x=start[0],y=start[1],parking=start,energy_eu=20)
            for i,order in enumerate(later):
                order=schedule_order(order,120+i*30)
                events.append(dict(id=f'post-charge-order-{i}',type='order',tick=120+i*30,order=order))
        elif case=='HND_TRANSFER':
            env.task_queue=env.task_queue[:1]
            env._normalize_tasks();order=env.task_queue[0]
            dock=env.transfer_bays[0]
            adjacent=next((dock[0]+dx,dock[1]+dy) for dx,dy in ((-1,0),(0,-1),(1,0),(0,1))
                if env._is_in_bounds(dock[0]+dx,dock[1]+dy) and env.grid[dock[1]+dy,dock[0]+dx]==0)
            for r in env.robots:
                if r['id'] not in (worker,relief):r.update(active=False,health='STANDBY',mode='FAULTED')
            order.update(owner=worker,status='blocked_recovery',picked_tick=0,
                         package_location=dict(kind='robot',robot_id=worker))
            env.robots[worker].update(x=dock[0],y=dock[1],current_task=order['task_id'],
                package_id=order['package_id'],load_status=1,target=tuple(order['delivery']),
                active=False,health='IMMOBILIZED',mode='BLOCKED_RECOVERY')
            env.robots[relief].update(x=adjacent[0],y=adjacent[1],orientation=(role//n)%4,target=adjacent)
            env.transfer_package(worker,relief,authorized=True)
        elif case=='HND06':
            env.task_queue=env.task_queue[:1]
            for i,r in enumerate(env.robots):
                if i not in (worker,relief):r.update(active=False,health='STANDBY',mode='FAULTED')
            env._normalize_tasks();order=env.task_queue[0]
            order['owner']=worker
            env.robots[worker].update(current_task=order['task_id'],target=tuple(order['pickup']),energy_eu=20,orientation=(role//n)%4)
        elif case=='COO02':
            env.config=replace(env.config,finite_batch=False)
            env.task_queue=[]
            env.robots[worker].update(x=env.drop_bays[0][0],y=env.drop_bays[0][1])
        elif case in ('COO04','COO05'):
            if case=='COO05':
                for r in env.robots:
                    if r['id']!=worker:r.update(active=False,health='STANDBY',mode='FAULTED')
                env.task_queue=env.task_queue[:2]
            else:
                later=deepcopy(env.task_queue);env.task_queue=env.task_queue[:1]
                for i,order in enumerate(later):
                    order=schedule_order(order,horizon//3);order['task_id']=1000+i
                    events.append(dict(id=f'reserve-demand-{i}',tick=horizon//3,type='order',order=order))
        elif case in ('COO06','ROB04'):
            env._normalize_tasks();order=env.task_queue[0]
            env.robots[worker].update(x=order['pickup'][0],y=order['pickup'][1])
            if case=='COO06':
                order.update(owner=worker,status='picked',picked_tick=0,package_location=dict(kind='robot',robot_id=worker))
                env.robots[worker].update(current_task=order['task_id'],load_status=1,package_id=order['package_id'],target=tuple(order['delivery']),mode='TO_DROP')
            events.extend([dict(id='fault',type='fault',tick=10,robot_id=worker),dict(id='repair',type='repair',tick=40,robot_id=worker)])
        elif case=='DEM02':
            for order in env.task_queue:order.update(pickup=env.pickup_bays[0],shelf=next(c for c in env.obstacles if abs(c[0]-env.pickup_bays[0][0])+abs(c[1]-env.pickup_bays[0][1])==1))
        elif case in ('DEM03','ROB05','EVA05'):
            count=n*4 if case!='EVA05' else 100
            later=generate_tasks(env,count,seed+1,start_task_id=1000)
            for i,order in enumerate(later):
                when=20 if case!='EVA05' else 20+i*20
                if when<horizon:
                    order=schedule_order(order,when);events.append(dict(id=f'burst-{i}',type='order',tick=when,order=order))
            if case=='ROB05':events.extend([dict(id='outage',type='charger_outage',tick=15,charger_id=0),dict(id='online',type='charger_online',tick=90,charger_id=0)])
            if case=='EVA05':env.config=replace(env.config,move_cost=.5,loaded_move_cost=.75,turn_cost=.1)
        elif case=='ROB01':
            occupied={(r['x'],r['y']) for r in env.robots}
            cell=next((x,y) for y in range(3,17) for x in range(2,18) if env.grid[y,x]==0 and (x,y) not in occupied)
            events.extend([dict(id='block',type='block',tick=15,cell=cell),dict(id='unblock',type='unblock',tick=70,cell=cell)])
        elif case=='ROB03':
            for r in env.robots:r.update(capacity_eu=50,energy_eu=40,energy_factor=1.2)
        if events:env.set_events(events)
    env.prepare_decision();env._sensor_history.clear();env._capture_sensor_snapshot();env.assert_invariants()
    manifest=capture_manifest(env,seed,'development',case)
    return env,manifest

def skill_outcome(case,env,metrics):
    # Operational success is a measured prerequisite, not universal robustness.
    clean=metrics['executed_collisions']==0 and metrics['safety_violations']==0 and metrics['avoidable_depletions']==0
    if case=='COO02':
        # Operational requirement is clear service bays, not an arbitrary exact
        # parking coordinate when no work exists. Dispatcher may still suggest it.
        success=env.idle_service_bay_occupants()==0
    else:success=metrics['completion_rate']>=.95
    if case=='EVA05':
        cycles=sum(e['type']=='charge_completed' for e in env.event_log)
        success=success and cycles>=3
    if case=='ROB04':success=success and all(r['active'] for r in env.robots)
    if case=='HND_TRANSFER':success=success and any(e['type']=='physical_transfer' for e in env.event_log)
    if case in ('CHARGE_CYCLE','CHARGE_APPROACH'):
        started=[e['tick'] for e in env.event_log if e['type']=='charge_completed']
        success=success and bool(started) and any(e['type']=='delivered' and e['tick']>min(started) for e in env.event_log)
    return dict(operational_prerequisite_passed=bool(clean and success),
        note='Case-specific statistical claims also require planner feasibility, repeated training seeds, role coverage and the catalog oracle; no automatic research-ready upgrade.')
