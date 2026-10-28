import pytest
from src.environment.mixed_logistics import make_mixed_scenario
from src.training.runner import collect_episode, parser, training_case
from src.baselines.nearest_task import NearestTaskDispatcher


def test_stock_must_be_physically_stored_before_shipping():
    env,manifest=make_mixed_scenario(910010,80)
    env.episode_metrics['step_count']=40
    env.prepare_decision()
    outgoing=next(t for t in env.task_queue if t['flow']=='outbound')
    assert outgoing['status']=='waiting_stock'
    assert outgoing['owner'] is None
    assert env.inventory[0]['state']=='receiving'
    env,manifest=make_mixed_scenario(910010,80)
    _,metrics=collect_episode(env,NearestTaskDispatcher(env),manifest)
    assert metrics['completed_tasks']==10
    assert len(env.inventory)==5
    assert all(s['state']=='shipped' and not s['reserved'] for s in env.inventory.values())
    assert sum(e['type']=='stored' for e in env.event_log)==5
    assert sum(e['type']=='shipped' for e in env.event_log)==5
    assert all(t['package_location']['kind']=='shelf' for t in env.finished_tasks if t['flow']=='inbound')
    env.assert_invariants()


def test_capacity_blocks_receiving_without_losing_or_conjuring_stock():
    env,_=make_mixed_scenario(910011,8)
    env.shelf_capacity=1
    bay=env.pickup_bays[0]
    # Deliberately send two physical incoming packages to the same single slot.
    orders=[e['order'].copy() for e in env._events if e['order']['flow']=='inbound'][:2]
    env.task_queue=[];env._ledger={};env.inventory={};env._events=[]
    for r in env.robots:r.update(current_task=None,target=None)
    for order in orders:
        order.update(storage_bay=bay,delivery=bay,creation_timestep=0)
    env.task_queue=orders
    env.prepare_decision()
    assert sum(t['status']=='waiting_capacity' for t in orders)==1
    assert sum(s['reserved'] for s in env.inventory.values())==1
    assert len(env.inventory)==2
    env.assert_invariants()


def test_rate_schedule_is_fixed_before_policy_acts_and_state_shape_is_constant():
    slow,sm=make_mixed_scenario(910012,80)
    fast,fm=make_mixed_scenario(910012,8)
    assert sm['expected_packages']==5 and fm['expected_packages']==50
    assert len(slow.get_training_state())==len(fast.get_training_state())
    assert slow.get_layout_spec()==fast.get_layout_spec()
    assert slow.receiving_bays!=slow.drop_bays
    expected=fm['events'].copy()
    for _ in range(45):fast.step([4]*5)
    assert fm['events']==expected
    assert sum(t['flow']=='inbound' for t in fast._ledger.values())==6
    assert sum(t['flow']=='outbound' for t in fast._ledger.values())==1
    assert len(fast.inventory)==6
    assert all(t['owner'] is None for t in fast.task_queue if t['flow']=='outbound')
    with pytest.raises(ValueError):make_mixed_scenario(1,1)
    with pytest.raises(ValueError):make_mixed_scenario(1,0)


def test_runner_uses_mixed_flow_in_demonstrations_and_training():
    args=parser().parse_args(['--workload','mixed','--horizon','600'])
    from src.training.demonstrations import demonstration_case
    env,m=demonstration_case(args,120011,2)
    assert m['arrival_interval']==20
    assert env.workload_version=='mixed_stock_v1'
    assert [training_case(args,120011,i)[1]['arrival_interval'] for i in range(5)]==[8,12,20,40,80]


def test_api_exposes_receiving_and_uses_fixed_arrivals_without_auto_refill():
    from fastapi.testclient import TestClient
    from src.deployment.api import app
    with TestClient(app) as client:
        response=client.post('/reset?session_id=mixed-test',json=dict(workload='mixed',algorithm='nearest_task',arrival_interval=80,initial_soc=.8))
        assert response.status_code==200
        state=response.json()
        assert state['workload_version']=='mixed_stock_v1'
        assert len(state['layout']['receiving_bays'])==2
        assert len(state['inventory'])==1
        assert all(abs(r['energy_eu']/r['capacity_eu']-.8)<1e-8 for r in state['robots'])
        step=client.post('/step?session_id=mixed-test',json=dict(actions=[4]*5))
        assert step.status_code==200
        assert len(step.json()['inventory'])==1

def test_fault_preserves_the_carried_package_and_storage_reservation():
    env,manifest=make_mixed_scenario(910013,80)
    teacher=NearestTaskDispatcher(env)
    for _ in range(150):
        env.step(teacher.act(env))
        carrier=next((r for r in env.robots if r['package_id'] is not None),None)
        if carrier is not None:break
    assert carrier is not None
    package=carrier['package_id'];task_id=carrier['current_task'];tick=env.tick
    env.set_events([dict(id='fault',type='fault',tick=tick,robot_id=carrier['id']),
                    dict(id='repair',type='repair',tick=tick+3,robot_id=carrier['id'])])
    assert carrier['package_id']==package and not carrier['active']
    assert env.inventory[package]['state']=='inbound_transit'
    assert env.inventory[package]['reserved']
    for _ in range(3):env.step([4]*5)
    assert carrier['active'] and carrier['current_task']==task_id
    assert carrier['package_id']==package
    env.assert_invariants()


def test_duplicate_physical_package_is_rejected_before_dispatch():
    env,_=make_mixed_scenario(910014,80)
    duplicate=env.task_queue[0].copy()
    duplicate.update(task_id=999,owner=None)
    env.task_queue.append(duplicate)
    with pytest.raises(ValueError,match='one job per flow'):env.prepare_decision()

def test_low_initial_battery_is_applied_before_assignment():
    env,_=make_mixed_scenario(910015,80,runtime_overrides=dict(initial_soc=.2))
    assert all(r['current_task'] is None for r in env.robots)
    assert all(env.get_action_mask(i)[6] for i in range(5))


def test_resume_cannot_silently_switch_package_workflows():
    from src.training.runner import resume_learner
    args=parser().parse_args(['--workload','mixed'])
    with pytest.raises(ValueError,match='workload differs'):
        resume_learner(dict(metadata=dict(arguments=dict(workload='outbound'))),args,None,None)


def test_mixed_scenario_can_be_sealed_and_replayed_with_inventory():
    import numpy as np
    from src.evaluation.manifests import capture_manifest,environment_from_manifest
    env,_=make_mixed_scenario(910016,20)
    sealed=capture_manifest(env,910016)
    replay=environment_from_manifest(sealed)
    assert replay.inventory==env.inventory
    import json
    assert json.loads(json.dumps(replay._events))==json.loads(json.dumps(env._events))
    assert np.array_equal(replay.get_training_state(),env.get_training_state())
    assert capture_manifest(replay,910016)==sealed
