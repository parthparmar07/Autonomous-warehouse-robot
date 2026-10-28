from dataclasses import replace
import numpy as np
import pytest
from src.environment.warehouse_env import WarehouseEnv
from src.environment.runtime_config import RuntimeConfig
from src.evaluation.scenarios import make_scenario
from src.evaluation.policy_cases import materialize,skill_outcome
from src.training.runner import collect_episode
from src.baselines.nearest_task import NearestTaskDispatcher


def test_service_bay_clearance_has_a_real_objective_and_safe_idle_floor_is_free():
    env,manifest=materialize('COO02',300100,5,100)
    assert not skill_outcome('COO02',env,dict(executed_collisions=0,safety_violations=0,avoidable_depletions=0))['operational_prerequisite_passed']
    env.step([4]*5)
    assert env.reward_components['idle_bay']==pytest.approx(-.02)
    robot=env.robots[0]
    worker=next(r for r in env.robots if (r['x'],r['y']) in env.drop_bays)
    worker.update(x=worker['x']+1)
    env.step([4]*5)
    assert env.reward_components['idle_bay']==0
    assert skill_outcome('COO02',env,dict(executed_collisions=0,safety_violations=0,avoidable_depletions=0))['operational_prerequisite_passed']


def test_discounted_potential_telescopes_and_wait_loop_cannot_mint_positive_reward():
    cfg=RuntimeConfig(potential_weight=2.,potential_gamma=.99,finite_batch=False)
    env=WarehouseEnv(grid_size=(8,8),num_agents=1,obstacles=[],safety_corridors=[],config=cfg)
    env.task_queue=[dict(task_id=0,pickup=(4,1),delivery=(6,1))]
    env.prepare_decision();initial=env.task_potential();shaping=[]
    for action in (0,1,4,2,3):
        outcome=env.step([action]);shaping.append(outcome['reward_components']['shaping'])
        assert outcome['reward']==pytest.approx(sum(outcome['reward_components'].values()))
    assert sum(cfg.potential_gamma**t*r for t,r in enumerate(shaping))==pytest.approx(
        cfg.potential_weight*(cfg.potential_gamma**len(shaping)*env.task_potential()-initial))
    assert sum(cfg.potential_gamma**t*r for t,r in enumerate(shaping))<0
    outcome=env.step([4]);assert outcome['reward_components']['shaping']<=0


def test_potential_handles_terminal_delivery_and_physical_service_progress():
    env=WarehouseEnv(grid_size=(8,8),num_agents=1,obstacles=[],safety_corridors=[],
        config=RuntimeConfig(potential_weight=1.,service_ticks=3))
    env.task_queue=[dict(task_id=0,pickup=(1,1),delivery=(1,1))]
    env.prepare_decision();values=[env.task_potential()]
    for _ in range(5):env.step([5]);values.append(env.task_potential())
    assert all(b>a for a,b in zip(values,values[1:]))
    out=env.step([5]);assert out['terminated']
    assert out['reward_components']['shaping']==pytest.approx(-values[-1])


def test_core_profile_holds_geometry_and_robot_parameters_fixed():
    env,_=make_scenario(100001,5,'C5',profile='core')
    assert not env._events and env.config.sensor_latency==0 and env.config.sensor_dropout==0
    assert len(env.pickup_bays)==12
    assert all(r['capacity_eu']==100 and r['energy_factor']==1 for r in env.robots)


def test_critic_distinguishes_equal_soc_different_physics_and_queue_order():
    env=WarehouseEnv(num_agents=5)
    before=env.get_training_state()
    env.robots[0].update(capacity_eu=50,energy_eu=50,energy_factor=1.2)
    assert not np.array_equal(before,env.get_training_state())
    for r in env.robots:r['energy_eu']=20
    env.request_charge(0);env.request_charge(1);env.request_charge(2)
    before=env.get_training_state();env.chargers[0]['queue'].reverse()
    # Requests may choose different stations; force a two-member same-station queue.
    env.chargers[0].update(queue=[1,2],requested={1:0,2:0})
    a=env.get_training_state();env.chargers[0]['queue'].reverse()
    assert not np.array_equal(a,env.get_training_state())


