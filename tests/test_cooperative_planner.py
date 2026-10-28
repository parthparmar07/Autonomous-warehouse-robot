from src.evaluation.policy_cases import materialize
from src.baselines.nearest_task import NearestTaskDispatcher

def test_joint_planner_yields_into_passing_pocket_and_finishes_opposing_cargo():
    env,_=materialize('COO01',303000,5,200)
    planner=NearestTaskDispatcher(env)
    used_pocket=False
    for _ in range(120):
        env.step(planner.act(env))
        used_pocket|=any((r['x'],r['y'])==(9,9) for r in env.robots)
        if env.completed_tasks==2:break
    assert used_pocket and env.completed_tasks==2
    assert env.episode_metrics['executed_collisions']==0


def test_pair_plan_cannot_suppress_service_for_partner_already_at_its_bay():
    env,_=materialize('COO01',303000,5,200)
    env.robots[0].update(x=16,y=10);env.robots[1].update(x=4,y=10)
    planner=NearestTaskDispatcher(env)
    planner._pair_plan=((0,1),((16,10),(3,10)),[
        ((15,10),(4,10)),((16,10),(3,10))])
    assert planner.act(env)[0]==5


def test_preflight_regression_full_fleet_does_not_freeze_at_a_drop_bay():
    from src.evaluation.scenarios import make_scenario
    env,_=make_scenario(301012,5,'C1','development',400,profile='core')
    planner=NearestTaskDispatcher(env)
    for _ in range(400):
        result=env.step(planner.act(env))
        if result['done']:break
    assert env.completed_tasks==10
