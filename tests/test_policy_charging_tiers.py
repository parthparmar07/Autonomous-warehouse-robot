import numpy as np
import pytest
from src.environment.warehouse_env import WarehouseEnv
from src.environment.runtime_config import RuntimeConfig
from src.baselines.nearest_task import NearestTaskDispatcher
from src.training.learner import RecurrentLearner


def floor(capacity=100,service=1):
    return WarehouseEnv(grid_size=(8,8),num_agents=1,obstacles=[],safety_corridors=[],charger_bays=[(5,5)],
        config=RuntimeConfig(capacity=capacity,service_ticks=service,finite_batch=False))


@pytest.mark.parametrize('capacity',[70.,100.,110.])
@pytest.mark.parametrize('soc,allowed',[(.300001,False),(.3,True),(.299999,True)])
def test_exact_soc_boundary_survives_repeated_decision_preparation(capacity,soc,allowed):
    env=floor(capacity);robot=env.robots[0];robot['energy_eu']=capacity*soc
    env.task_queue=[dict(task_id=0,pickup=(2,1),delivery=(3,1))] if allowed else []
    for _ in range(3):env.prepare_decision()
    assert robot['charger_id'] is None and robot['current_task'] is None
    assert bool(env.get_action_mask(0)[6]) is allowed
    assert env.request_charge(0) is allowed


def test_order_completion_exposes_charge_decision_before_next_order_assignment():
    env=floor();robot=env.robots[0]
    env.task_queue=[dict(task_id=0,pickup=(1,1),delivery=(2,1)),
        dict(task_id=1,pickup=(3,1),delivery=(4,1))]
    env.prepare_decision();env.step([5])
    robot['energy_eu']=30.2
    env.step([0])
    assert robot['energy_eu']<30 and robot['current_task']==0 and robot['package_id']==0
    assert not env.get_action_mask(0)[6]
    env.step([5])
    assert env.completed_tasks==1 and robot['current_task'] is None and robot['charger_id'] is None
    assert env.task_queue[0]['owner'] is None and env.get_action_mask(0)[6]
    env.step([4])
    assert robot['charger_id'] is None  # no hidden dispatcher takeover on WAIT
    env.step([6])
    assert robot['charger_id']==0
    event=next(e for e in env.event_log if e['type']=='charge_requested')
    assert event['source']=='policy'


def test_no_mission_feasibility_fallback_bypasses_above_threshold_rule():
    env=floor();robot=env.robots[0];robot.update(energy_eu=40,energy_factor=10)
    env.task_queue=[dict(task_id=0,pickup=(7,7),delivery=(0,7))]
    env.prepare_decision()
    assert robot['current_task'] is None and robot['charger_id'] is None
    assert not env.get_action_mask(0)[6] and not env.request_charge(0)
    assert env.task_queue[0]['owner'] is None


def test_policy_mask_overrides_arbitrarily_high_charge_q_value_above_threshold():
    import torch
    torch.set_num_threads(1)
    env=floor();env.robots[0]['energy_eu']=80;env.prepare_decision()
    model=RecurrentLearner('qmix',1,env.observation_dim,len(env.get_training_state()))
    with torch.no_grad():
        model.actor.head.weight.zero_();model.actor.head.bias.zero_();model.actor.head.bias[6]=1e6
    action=model.act([env.get_local_observation(0)],[env.get_action_mask(0)])
    assert action[0]!=6
    env.robots[0]['energy_eu']=30;env.prepare_decision()
    action=model.act([env.get_local_observation(0)],[env.get_action_mask(0)])
    assert action==[6]


def test_charge_service_continues_above_thirty_until_release_and_planner_requests_explicitly():
    env=floor();robot=env.robots[0];robot.update(x=5,y=5,energy_eu=30)
    planner=NearestTaskDispatcher(env)
    assert planner.act(env)==[6]
    env.step([6])
    while robot['charger_id'] is not None:
        assert not env.get_action_mask(0)[6] and env.get_action_mask(0)[5]
        env.step([5])
    assert robot['energy_eu']/robot['capacity_eu']>=.9
    assert not env.get_action_mask(0)[6]
