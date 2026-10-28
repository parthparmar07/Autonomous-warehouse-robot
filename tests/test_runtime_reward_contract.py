"""Hand-calculated reward checks against actual transitions, not the legacy helper."""
import pytest
from src.environment.warehouse_env import WarehouseEnv
from src.environment.runtime_config import RuntimeConfig
from tests.helpers import picked


def floor(n=1):
    return WarehouseEnv(grid_size=(8,8),num_agents=n,obstacles=[],safety_corridors=[],config=RuntimeConfig(potential_weight=0.))


def test_idle_and_charging_do_not_generate_positive_reward():
    env=floor()
    assert env.step([4])['reward']==pytest.approx(-.00005)
    env=WarehouseEnv(grid_size=(8,8),num_agents=1,obstacles=[],safety_corridors=[],charger_bays=[(5,5)],config=RuntimeConfig(potential_weight=0.))
    env.robots[0].update(x=5,y=5,energy_eu=1)
    env.request_charge(0)
    assert env.step([5])['reward']==pytest.approx(-.00005)


def test_pending_overdue_priority_three_order_and_wait_cost():
    env=floor()
    env.task_queue=[dict(task_id=0,pickup=(5,5),delivery=(6,6),creation_timestep=0,deadline=0,priority=3)]
    assert env.step([4])['reward']==pytest.approx(-.00705)


def test_delivery_reward_excludes_completed_order_from_backlog():
    env=picked(service=1,potential_weight=0.)
    env.robots[0].update(x=12,y=12)
    result=env.step([5])
    assert env.completed_tasks==1 and result['reward']==pytest.approx(9.999)


def test_depletion_penalty_and_actual_clipped_energy_debit():
    env=floor();env.robots[0]['energy_eu']=.001
    result=env.step([4])
    assert env.episode_metrics['avoidable_depletions']==1
    assert result['reward']==pytest.approx(-10.00001)


def test_conflicting_moves_charge_one_movement_one_wait_and_one_conflict():
    env=floor(2)
    env.robots[0].update(x=2,y=2,orientation=1)
    env.robots[1].update(x=4,y=2,orientation=3)
    result=env.step([0,0])
    assert env.episode_metrics['collision_count']==1
    assert result['reward']==pytest.approx(-.05205)


def test_future_order_is_charged_backlog_only_after_release():
    env=floor()
    env.set_events([dict(id='arrival',type='order',tick=2,order=dict(task_id=0,pickup=(5,5),delivery=(6,6),creation_timestep=2,deadline=10,priority=1))])
    assert env.step([4])['reward']==pytest.approx(-.00005)
    assert env.step([4])['reward']==pytest.approx(-.00105)
