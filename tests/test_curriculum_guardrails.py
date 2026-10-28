from copy import deepcopy
import numpy as np
import pytest
from src.environment.warehouse_env import WarehouseEnv
from src.environment.runtime_config import RuntimeConfig
from src.environment.spec import ACTION_DIM,OBS_DIM,MODES,SELF_FIELDS,SENSOR_CHANNELS
from src.evaluation.scenarios import make_scenario
from src.evaluation.policy_cases import materialize,skill_outcome
from src.evaluation.curriculum import STAGES
from src.evaluation.curriculum_gate import prerequisite_gate,evaluate_curriculum_gate
from src.training.runner import collect_episode
from src.baselines.nearest_task import NearestTaskDispatcher


def floor(n=1,**kwargs):
    return WarehouseEnv(grid_size=(8,8),num_agents=n,obstacles=[],safety_corridors=[],
        charger_bays=[(5,5)],config=RuntimeConfig(finite_batch=False),**kwargs)


@pytest.mark.parametrize('soc,assigned,allowed',[(.83,True,False),(.4,True,False),
    (.83,False,False),(.45,False,False),(.44,False,False),(.300001,False,False),
    (.3,False,True),(.299999,False,True)])
def test_voluntary_charge_guards_mask_and_direct_transition(soc,assigned,allowed):
    env=floor();r=env.robots[0];r['energy_eu']=soc*100
    if assigned:
        env.task_queue=[dict(task_id=0,pickup=(2,1),delivery=(4,4))]
        env.prepare_decision();assert r['current_task']==0
    env.prepare_decision()
    assert bool(env.get_action_mask(0)[6]) is allowed
    assert env.request_charge(0) is allowed
    if not allowed:
        before=(r['current_task'],r['target'])
        env.step([6])  # callers cannot bypass the mask with a raw action
        assert r['current_task']==before[0]
        if assigned:assert r['target']==before[1]
        assert r['charger_id'] is None
        assert r['executed_actions'][0]==4


def test_crossing_threshold_does_not_cancel_unpicked_or_loaded_orders():
    env=floor();env.task_queue=[dict(task_id=0,pickup=(1,1),delivery=(6,6))]
    env.prepare_decision();r=env.robots[0]
    r['energy_eu']=20;env.prepare_decision()
    assert r['charger_id'] is None and r['current_task']==0
    assert not env.get_action_mask(0)[6]
    assert env.task_queue[0]['package_location']['kind']=='shelf'
    env=floor();env.task_queue=[dict(task_id=0,pickup=(1,1),delivery=(6,6))]
    env.prepare_decision()
    for _ in range(3):env.step([5])
    r=env.robots[0];r['energy_eu']=20;env.prepare_decision()
    assert r['charger_id'] is None and r['package_id']==0 and r['current_task']==0
    assert not env.get_action_mask(0)[6]


def test_global_pair_assignment_prefers_nearer_robot_and_is_idempotent():
    env=floor(2);env.robots[0].update(x=1,y=1);env.robots[1].update(x=5,y=4)
    env.task_queue=[dict(task_id=0,pickup=(5,4),delivery=(6,4))]
    env.prepare_decision();assert env.task_queue[0]['owner']==1
    before=(deepcopy(env.robots),deepcopy(env.event_log))
    for _ in range(4):env.prepare_decision()
    assert before==(env.robots,env.event_log)
    owners=set()
    for seed in range(4):
        env=floor(2,seed=seed)
        env.robots[0].update(x=2,y=3);env.robots[1].update(x=4,y=3)
        env.task_queue=[dict(task_id=0,pickup=(3,3),delivery=(3,4))]
        env.prepare_decision();owners.add(env.task_queue[0]['owner'])
    assert owners=={0,1}


def test_queued_robot_gets_a_fresh_lease_on_promotion_and_real_progress_renews_it():
    env=floor(2);env.config=RuntimeConfig(lease_ticks=3,finite_batch=False)
    for r in env.robots:r['energy_eu']=40
    assert env.request_charge(0,automatic=True) and env.request_charge(1,automatic=True)
    for _ in range(3):env.step([4,4])
    assert env.chargers[0]['owner']==1
    assert env.robots[1]['charger_approach_tick']==3
    for _ in range(4):env.prepare_decision()
    assert env.chargers[0]['owner']==1
    env.robots[1].update(x=4,y=5,orientation=1)
    env.step([4,0])
    assert (env.robots[1]['x'],env.robots[1]['y'])==(5,5)
    assert env.robots[1]['charger_approach_tick']==3


def test_observation_contains_timestamped_sensor_configuration_and_actual_action_feedback():
    env=floor();env.config=RuntimeConfig(front_fov=90,rear_fov=90,sensor_latency=2,sensor_dropout=.05)
    r=env.robots[0];r.update(x=1,y=1,orientation=1)
    initial=env.get_local_observation(0)
    assert len(initial)==OBS_DIM==348 and not initial[-14:].any()
    config_start=25*len(SENSOR_CHANNELS)+len(SELF_FIELDS)+len(MODES)
    np.testing.assert_allclose(initial[config_start:config_start+4],[.25,.25,.05,.04])
    env.step([2]);env.step([0])  # left then forward (north)
    hist=env.get_local_observation(0)[-14:].reshape(2,ACTION_DIM)
    assert hist.argmax(1).tolist()==[0,2]
    r.update(x=0,y=0,orientation=0)
    env.step([0])  # out-of-bounds request is actually WAIT
    assert r['executed_actions']==[4,0]
    assert env.get_local_observation(0)[-14:].reshape(2,7).argmax(1).tolist()==[4,0]
    env.reset();assert not env.get_local_observation(0)[-14:].any()


def test_collision_feedback_reports_wait_for_blocked_motion():
    env=floor(2)
    env.robots[0].update(x=2,y=2,orientation=1)
    env.robots[1].update(x=4,y=2,orientation=3)
    env.step([0,0])
    assert sorted(r['executed_actions'][0] for r in env.robots)==[0,4]


def test_bridge_stages_preserve_full_geometry_and_rotate_physical_workers():
    assert STAGES[:4]==('C0','C0.5','C0.75','C1')
    solo_roles=set();pairs=set()
    for role in range(20):
        env,_=make_scenario(100000+role,5,'C0.5',role_index=role)
        assert len(env.obstacles)==60 and len(env.pickup_bays)==12 and len(env.task_queue)==2
        active=[r for r in env.robots if r['active']]
        assert len(active)==1
        solo_roles.add((active[0]['id'],active[0]['orientation']))
        env,_=make_scenario(100000+role,5,'C0.75',role_index=role)
        active=[r for r in env.robots if r['active']]
        assert len(active)==2 and len(env.task_queue)==4
        pairs.add(tuple(r['id'] for r in active))
    assert len(solo_roles)==20 and len(pairs)==5
    for stage in ('C1','C2','C3'):
        env,_=make_scenario(100001,5,stage)
        assert env.config.sensor_latency==0 and env.config.sensor_dropout==0


def test_real_transfer_prerequisite_requires_physical_handover_and_delivery():
    for role in (0,7,19):
        env,manifest=materialize('HND_TRANSFER',304000+role,5,400,role)
        _,metrics=collect_episode(env,NearestTaskDispatcher(env),
            dict(seed=manifest['seed'],scenario_hash=manifest['manifest_hash']))
        assert any(e['type']=='physical_transfer' for e in env.event_log)
        assert metrics['completed_tasks']==1
        assert skill_outcome('HND_TRANSFER',env,metrics)['operational_prerequisite_passed']


def test_skill_failure_or_missing_role_cannot_hide_behind_generic_success():
    rows=[dict(case='passing',role=i,operational_prerequisite_passed=True) for i in range(20)]
    assert prerequisite_gate(rows,5)['passed']
    rows[-1]['operational_prerequisite_passed']=False
    assert not prerequisite_gate(rows,5)['passed']
    rows[-1].update(role=0,operational_prerequisite_passed=True)
    assert not prerequisite_gate(rows,5)['passed']


def test_generic_success_cannot_promote_a_policy_that_fails_named_skills(monkeypatch):
    from src.evaluation import curriculum_gate as gate
    success=dict(completion_rate=1.,executed_collisions=0,avoidable_depletions=0)
    monkeypatch.setattr(gate,'evaluate',lambda *a,**k:[dict(success)]*20)
    monkeypatch.setattr(gate,'materialize',lambda case,seed,n,horizon,role:(None,dict(seed=seed,manifest_hash=str(role))))
    monkeypatch.setattr(gate,'NearestTaskDispatcher',lambda env:None)
    monkeypatch.setattr(gate,'collect_episode',lambda *a,**k:(None,dict(success)))
    monkeypatch.setattr(gate,'skill_outcome',lambda *a:dict(operational_prerequisite_passed=False))
    result=evaluate_curriculum_gate(None,['C0','C0.5','C0.75'],5)
    assert result['generic_gate']['passed'] and not result['passed']
    assert result['required_cases']==['COO01'] and len(result['skill_results'])==20


def test_old_checkpoint_is_rejected_after_observation_and_environment_change():
    from src.training.learner import RecurrentLearner
    env=floor();model=RecurrentLearner('qmix',1,env.observation_dim,len(env.get_training_state()))
    payload=model.checkpoint()
    payload.update(environment_version='warehouse_v3_energy',observation_version='egocentric_range_v1')
    with pytest.raises(ValueError,match='version is incompatible'):
        RecurrentLearner.from_checkpoint(payload)
