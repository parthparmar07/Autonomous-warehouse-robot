"""Retention uses its trained task definition; charging includes real approach."""
import pytest

from src.baselines.nearest_task import NearestTaskDispatcher
from src.evaluation import curriculum_gate
from src.evaluation.curriculum import eligible_cases
from src.evaluation.policy_cases import materialize, skill_outcome
from src.training.runner import collect_episode
from src.environment.core import WarehouseEnv
from src.evaluation.operational import episode_metrics
from src.evaluation.scenarios import make_scenario
from src.baselines.pair_planner import plan_pair


def test_retention_keeps_core_energy_profile_after_robust_stage(monkeypatch):
    calls=[]
    def evaluate(model,seeds,n,stage,**kwargs):
        calls.append((stage,kwargs['profile']))
        return [dict(completion_rate=1.,executed_collisions=0,avoidable_depletions=0)]*20
    monkeypatch.setattr(curriculum_gate,'evaluate',evaluate)
    monkeypatch.setattr(curriculum_gate,'eligible_cases',lambda stage,n:())
    result=curriculum_gate.evaluate_curriculum_gate(None,['C2','C3'],5,
        profile='robust',stage_profiles={'C2':'core','C3':'robust'})
    assert calls==[('C2','core'),('C3','robust')]
    assert result['stage_profiles']=={'C2':'core','C3':'robust'}
    assert result['passed']
    assert {row['profile'] for row in result['generic_results'] if row['family']=='C2'}=={'core'}
    with pytest.raises(ValueError,match='profile'):
        curriculum_gate.evaluate_curriculum_gate(None,['C2'],5,stage_profiles={'C2':'typo'})


def test_default_gate_profile_remains_backward_compatible(monkeypatch):
    calls=[]
    def evaluate(model,seeds,n,stage,**kwargs):
        calls.append(kwargs['profile'])
        return [dict(completion_rate=1.,executed_collisions=0,avoidable_depletions=0)]*20
    monkeypatch.setattr(curriculum_gate,'evaluate',evaluate)
    monkeypatch.setattr(curriculum_gate,'eligible_cases',lambda stage,n:())
    assert curriculum_gate.evaluate_curriculum_gate(None,['C0','C2'],5,profile='core')['passed']
    assert calls==['core','core']


@pytest.mark.parametrize('role',range(20))
def test_charge_approach_covers_workers_headings_and_both_docks(role):
    env,manifest=materialize('CHARGE_APPROACH',304000+role,5,400,role)
    worker=env.robots[role%5]
    start=(worker['x'],worker['y'])
    intended_dock=env.chargers[role%2]['position']
    assert worker['orientation']==role//5
    assert abs(start[0]-intended_dock[0])+abs(start[1]-intended_dock[1])==4
    assert worker['charger_id'] is None and env.get_action_mask(worker['id'])[6]
    assert 'CHARGE_APPROACH' in eligible_cases('C2',5)
    _,metrics=collect_episode(env,NearestTaskDispatcher(env),
        dict(seed=manifest['seed'],scenario_hash=manifest['manifest_hash']))
    requests=[event for event in env.event_log if event['type']=='charge_requested']
    assert requests and requests[0]['charger_id']==role%2
    assert skill_outcome('CHARGE_APPROACH',env,metrics)['operational_prerequisite_passed']


def test_approach_oracle_rejects_delivery_without_completed_charging():
    env,_=materialize('CHARGE_APPROACH',304000,5,400,0)
    metrics=dict(completion_rate=1.,executed_collisions=0,safety_violations=0,avoidable_depletions=0)
    env.event_log=[dict(type='delivered',tick=200)]
    assert not skill_outcome('CHARGE_APPROACH',env,metrics)['operational_prerequisite_passed']


def test_charge_tier_energy_constraint_is_reported_without_early_charge():
    env=WarehouseEnv(num_agents=1)
    robot=env.robots[0]
    robot.update(capacity_eu=70,energy_eu=22)
    env.task_queue=[dict(task_id=0,pickup=(15,6),delivery=(2,18))]
    env.prepare_decision()
    assert env.mission_energy(robot,env.task_queue[0])==pytest.approx(23.1)
    assert robot['current_task'] is None and not env.get_action_mask(0)[6]
    assert env.charge_locked_infeasible_agents()==1
    # Historical resumable metrics need no schema migration.
    env.episode_metrics.pop('charge_locked_infeasible_ticks')
    env.step([4])
    assert env.episode_metrics['charge_locked_infeasible_ticks']==1
    metrics=episode_metrics(env,dict(seed=0,scenario_hash='diagnostic'),[])
    assert metrics['charge_locked_infeasible_ticks']==1
    assert robot['charger_id'] is None
    robot['energy_eu']=21
    assert env.charge_locked_infeasible_agents()==0
    assert env.get_action_mask(0)[6]


def test_route_infeasibility_is_not_misreported_as_energy_only_constraint():
    env=WarehouseEnv(num_agents=1)
    env.robots[0].update(capacity_eu=70,energy_eu=22)
    env.task_queue=[dict(task_id=0,pickup=(15,6),delivery=(2,18))]
    env.prepare_decision()
    env.grid[10,1:19]=1
    env._paths.clear()
    assert env.charge_locked_infeasible_agents()==0


@pytest.mark.parametrize('blocker',['robot','new_obstacle'])
def test_committed_pair_plan_is_invalidated_when_next_cell_changes(blocker):
    env=WarehouseEnv(grid_size=(8,8),num_agents=3,obstacles=[],safety_corridors=[])
    for robot,position,goal in zip(env.robots,[(2,2),(5,5),(3,2)],[(5,2),(5,4),(2,3)]):
        robot.update(x=position[0],y=position[1],orientation=1,parking=goal)
    env.prepare_decision()
    planner=NearestTaskDispatcher(env)
    stale=((0,1),((5,2),(5,4)),[((2,2),(5,5)),((3,2),(5,4)),((4,2),(5,4)),((5,2),(5,4))])
    planner._pair_plan=stale
    if blocker=='new_obstacle':
        env.robots[2].update(x=1,y=3)
        env.grid[2,3]=1
        env._paths.clear()
    actions=planner.act(env)
    assert planner._pair_plan is not stale
    assert actions[0]!=0  # cannot keep asking to drive into the changed cell


@pytest.mark.parametrize('seed',[301006,301014])
def test_fault_stage_planner_recovers_when_parking_traffic_enters_joint_route(seed):
    env,manifest=make_scenario(seed,5,'C3','development',400,profile='robust')
    _,metrics=collect_episode(env,NearestTaskDispatcher(env),manifest)
    assert metrics['completed_tasks']==10
    assert metrics['avoidable_depletions']==0


def test_joint_route_does_not_make_robot_beside_charger_follow_departing_partner():
    env=WarehouseEnv(num_agents=2)
    env.robots[0].update(x=17,y=16,target=(17,17),energy_eu=2)
    env.robots[1].update(x=17,y=17,target=(2,13))
    route=plan_pair(env,0,1,budget=2000)
    assert route and route[-1]==((17,17),(2,13))
    moves=sum(before[0]!=after[0] for before,after in zip(route,route[1:]))
    assert moves==1  # partner can leave; charging robot need not tour the aisle


@pytest.mark.parametrize('role',[0,10,13])
def test_long_shift_planner_does_not_spend_charger_approach_energy_on_joint_detour(role):
    env,manifest=materialize('EVA05',304000+role,5,400,role)
    _,metrics=collect_episode(env,NearestTaskDispatcher(env),
        dict(seed=manifest['seed'],scenario_hash=manifest['manifest_hash']))
    assert skill_outcome('EVA05',env,metrics)['operational_prerequisite_passed']
