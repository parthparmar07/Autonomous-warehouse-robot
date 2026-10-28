from dataclasses import replace
import numpy as np
import pytest
from src.environment.warehouse_env import WarehouseEnv
from src.environment.runtime_config import RuntimeConfig
from src.environment.movement import resolve_moves
from src.environment.spec import OBS_DIM, SELF_FIELDS, SENSOR_CHANNELS


def floor(n=1, **kwargs):
    return WarehouseEnv(grid_size=(8,8),num_agents=n,obstacles=[],safety_corridors=[],**kwargs)


def test_sensor_occlusion_unknown_and_neighbor_detection():
    env=floor(2)
    env.robots[0].update(x=3,y=3,orientation=0)
    env.robots[1].update(x=3,y=2)
    frame,_=env.get_sensor_frame(0)
    assert frame[1,2,7]==1
    assert frame[0,2,0]==0  # other robot occludes the cell behind it
    env.robots[1].update(x=6,y=6)
    env.grid[2,3]=1
    frame,_=env.get_sensor_frame(0)
    assert frame[1,2,2]==1
    assert np.array_equal(frame[0,2],np.zeros(len(SENSOR_CHANNELS)))


@pytest.mark.parametrize('heading,front',[(0,(3,2)),(1,(4,3)),(2,(3,4)),(3,(2,3))])
def test_egocentric_front_and_fov(heading,front):
    env=floor(2,config=RuntimeConfig(front_fov=90,rear_fov=0))
    env.robots[0].update(x=3,y=3,orientation=heading)
    env.robots[1].update(x=front[0],y=front[1])
    frame,_=env.get_sensor_frame(0)
    assert frame[1,2,7]==1
    assert frame[3,2,0]==0


def test_sensor_read_order_noise_remote_privacy_and_boundary():
    env=floor(2,config=RuntimeConfig(sensor_dropout=.5))
    first=env.get_local_observation(0)
    assert first.shape==(OBS_DIM,)
    assert np.array_equal(first,env.get_local_observation(0))
    env.robots[1].update(x=6,y=6)
    before=env.get_local_observation(0)
    env.robots[1].update(x=5,y=6,energy_eu=1)
    assert np.array_equal(before,env.get_local_observation(0))
    clear=floor()
    clear.robots[0].update(x=0,y=0,orientation=0)
    frame,_=clear.get_sensor_frame(0)
    assert frame[1,2,1]==1  # known outside, not free floor


def test_latency_is_timestamped_and_not_observer_dependent():
    env=floor(config=RuntimeConfig(sensor_latency=2))
    assert not env.get_sensor_frame(0)[0].any()
    env.step([4]);env.step([4])
    frame,age=env.get_sensor_frame(0)
    assert frame.any() and age==2


def test_joint_resolution_swaps_dependencies_following_cycles():
    assert resolve_moves([(1,1),(2,1)],[(2,1),(1,1)],[0,1])[0]==[(1,1),(2,1)]
    assert resolve_moves([(1,1),(2,1),(3,1)],[(2,1),(3,1),(3,1)],[0,1,2])[0]==[(1,1),(2,1),(3,1)]
    assert resolve_moves([(1,1),(2,1)],[(2,1),(3,1)],[0,1])[0]==[(2,1),(3,1)]
    starts=[(1,1),(2,1),(2,2),(1,2)]
    destinations=starts[1:]+starts[:1]
    assert resolve_moves(starts,destinations,[0,1,2,3])[0]==destinations


def test_random_joint_actions_preserve_occupancy_and_energy():
    env=WarehouseEnv(num_agents=20)
    rng=np.random.default_rng(1)
    initial=sum(r['energy_eu'] for r in env.robots)
    for _ in range(200):
        env.step([int(rng.choice(np.flatnonzero(env.get_action_mask(i)))) for i in range(20)])
        env.assert_invariants()
    final=sum(r['energy_eu'] for r in env.robots)
    assert final==pytest.approx(initial-env.episode_metrics['energy_used']+env.episode_metrics['energy_charged'])


def test_exclusive_assignment_and_interrupted_service():
    env=floor(2,config=RuntimeConfig(service_ticks=3))
    env.task_queue=[dict(task_id=0,pickup=(1,1),delivery=(5,5),status='pending')]
    env.prepare_decision()
    assert sum(r['current_task']==0 for r in env.robots)==1
    env.step([5,4]);env.step([5,4])
    assert env.robots[0]['package_id'] is None
    env.step([4,4]);env.step([5,4])
    assert env.robots[0]['package_id'] is None
    env.step([5,4]);env.step([5,4])
    assert env.robots[0]['package_id']==0


def test_fault_keeps_loaded_package_and_releases_unpicked_work():
    env=floor(2,config=RuntimeConfig(service_ticks=1,heartbeat_timeout=2))
    env.task_queue=[dict(task_id=0,pickup=(1,1),delivery=(5,5),status='pending')]
    env.prepare_decision();env.step([5,4])
    env.set_events([dict(id='fault',tick=env.tick,type='fault',robot_id=0)])
    for _ in range(3):env.step([4,4])
    assert env.robots[0]['package_id']==0 and env.robots[1]['package_id'] is None
    assert env.task_queue[0]['status']=='blocked_recovery'


def test_future_events_do_not_leak_and_release_at_exact_tick():
    env=floor()
    before=env.get_local_observation(0).copy()
    state=env.get_training_state().copy()
    env.set_events([dict(id='order',tick=3,type='order',order=dict(task_id=0,pickup=(2,2),delivery=(5,5)))])
    assert np.array_equal(before,env.get_local_observation(0))
    assert np.array_equal(state,env.get_training_state())
    for _ in range(2):env.step([4])
    assert not env.task_queue
    env.step([4]);assert len(env.task_queue)==1


def test_charging_has_exclusive_lease_energy_conservation_and_taper():
    env=floor(2,charger_bays=[(5,5)],config=RuntimeConfig(charge_release=.99))
    env.robots[0].update(x=5,y=5,energy_eu=79)
    env.robots[1].update(energy_eu=20)
    env.request_charge(0,automatic=True);env.request_charge(1)
    assert env.chargers[0]['owner']==0 and env.chargers[0]['queue']==[1]
    before=env.robots[0]['energy_eu']
    env.step([5,4])
    assert 80<env.robots[0]['energy_eu']<81
    assert env.robots[0]['energy_eu']==pytest.approx(before-env.config.wait_cost+env.episode_metrics['energy_charged'])


def test_complete_state_includes_last_agent():
    env=WarehouseEnv(num_agents=20)
    before=env.get_training_state()
    env.robots[19]['energy_eu']=50
    assert not np.array_equal(before,env.get_training_state())
    assert len(before)>64
