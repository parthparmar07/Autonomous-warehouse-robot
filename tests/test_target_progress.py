import math
import pytest
from src.environment.warehouse_env import WarehouseEnv
from src.environment.runtime_config import RuntimeConfig
from src.training.runner import parser


def test_shaping_defaults_and_support_goal_progress():
    assert RuntimeConfig().potential_weight==parser().parse_args([]).potential_weight==2.
    env=WarehouseEnv(grid_size=(8,8),num_agents=1,obstacles=[],safety_corridors=[],
        charger_bays=[(5,5)],config=RuntimeConfig(finite_batch=False))
    robot=env.robots[0]
    for mode in ('TO_CHARGER','QUEUED_CHARGE','TO_PARK'):
        robot.update(x=2,y=5,target=(5,5),mode=mode)
        before=env.task_potential();robot['x']=3
        assert env.task_potential()>before
        robot['x']=5
        assert env.task_potential()==pytest.approx(.3)


def test_legal_shelf_detours_refresh_charger_and_task_progress():
    env=WarehouseEnv(grid_size=(8,8),num_agents=1,
        obstacles=[(4,y) for y in range(2,5)],safety_corridors=[],charger_bays=[(5,3)],
        config=RuntimeConfig(finite_batch=False,lease_ticks=2))
    for _ in range(15):env.step([4])
    robot=env.robots[0];robot.update(x=3,y=3,orientation=0,energy_eu=20)
    requested=env.tick
    assert env.request_charge(0)
    assert robot['last_progress_tick']==robot['charger_approach_tick']==requested
    for _ in range(2):
        origin=(robot['x'],robot['y']);goal=robot['target'];old=env.path(origin,goal)
        env.step([0]);position=(robot['x'],robot['y'])
        assert math.dist(position,goal)>math.dist(origin,goal)
        assert len(env.path(position,goal))<len(old)
        assert robot['charger_id']==0
    assert robot['last_progress_tick']==robot['charger_approach_tick']>requested
    assert not any(e['type']=='charger_lease_expired' for e in env.event_log)


def test_critic_includes_capacity_and_consumption_without_200_order_padding():
    env=WarehouseEnv(num_agents=5)
    assert env.config.max_orders==128
    before=env.get_training_state()
    env.robots[0].update(capacity_eu=50,energy_eu=50,energy_factor=1.2)
    after=env.get_training_state()
    assert len(after)==1818 and (before!=after).any()
    from src.deployment.schemas import ResetRequest
    with pytest.raises(ValueError):ResetRequest(num_tasks=129)
