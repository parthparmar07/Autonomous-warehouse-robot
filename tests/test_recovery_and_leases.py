from src.evaluation.policy_cases import materialize
from src.training.runner import collect_episode
from src.baselines.nearest_task import NearestTaskDispatcher
import numpy as np
from src.environment.warehouse_env import WarehouseEnv
from src.environment.runtime_config import RuntimeConfig

def floor(n=2,**kwargs):return WarehouseEnv(num_agents=n,grid_size=(8,8),obstacles=[],safety_corridors=[],**kwargs)

def test_stalled_unpicked_lease_reassigns_without_teleportation():
    env=floor(config=RuntimeConfig(lease_ticks=3))
    env.task_queue=[dict(task_id=0,pickup=(5,5),delivery=(6,6))]
    env.prepare_decision();owner=env.task_queue[0]['owner']
    for _ in range(3):env.step([4,4])
    assert env.task_queue[0]['owner']==1-owner and env.robots[owner]['current_task'] is None
    assert env.task_queue[0]['package_location']['kind']=='shelf'

def test_stalled_charger_approach_does_not_hold_station_forever():
    env=floor(charger_bays=[(5,5)],config=RuntimeConfig(lease_ticks=3))
    for r in env.robots:r['energy_eu']=20
    env.request_charge(0);env.request_charge(1)
    for _ in range(3):env.step([4,4])
    assert env.chargers[0]['owner']==1 and env.robots[0]['charger_id'] is None
    assert any(e['type']=='charger_lease_expired' for e in env.event_log)

def test_connected_healthy_robot_can_recharge_from_zero_without_energy_creation():
    env=floor(1,charger_bays=[(5,5)])
    robot=env.robots[0];robot.update(x=5,y=5,energy_eu=1)
    env.request_charge(0);robot.update(energy_eu=0,active=False,mode='DEPLETED')
    assert env.get_action_mask(0)[5]
    env.step([5]);assert robot['active'] and robot['energy_eu']==2
    assert env.episode_metrics['energy_used']==0 and env.episode_metrics['energy_charged']==2

def test_evaluation_does_not_advance_exploration_rng():
    from src.training.learner import RecurrentLearner
    import copy
    env=floor()
    model=RecurrentLearner('iql',2,env.observation_dim,len(env.get_training_state()))
    before=copy.deepcopy(model.rng.bit_generator.state)
    model.act([env.get_local_observation(i) for i in range(2)],[env.get_action_mask(i) for i in range(2)])
    assert model.rng.bit_generator.state==before

def test_lost_transfer_adjacency_discards_partial_service():
    from tests.helpers import picked
    env=picked(2);env.transfer_bays=[(5,5)]
    env.robots[0].update(x=5,y=5);env.robots[1].update(x=6,y=5)
    env.transfer_package(0,1,authorized=True)
    env.step([4,5]);env.step([4,5])
    env.robots[0]['y']=6
    env.step([4,5])
    env.robots[0]['y']=5
    env.transfer_package(0,1,authorized=True)
    env.step([4,5]);assert env.robots[0]['package_id']==0
    env.step([4,5]);assert env.robots[0]['package_id']==0
    env.step([4,5]);assert env.robots[1]['package_id']==0

def test_service_completion_event_uses_actual_completed_tick():
    from tests.helpers import picked
    env=picked(service=1)
    env.robots[0].update(x=12,y=12)
    env.step([5])
    delivered=next(e for e in env.event_log if e['type']=='delivered')
    assert delivered['tick']==env.finished_tasks[0]['completed_tick']==env.tick


def test_new_charger_owner_gets_a_fresh_approach_lease():
    env=WarehouseEnv(grid_size=(8,8),num_agents=1,obstacles=[],safety_corridors=[],charger_bays=[(5,5)],
        config=RuntimeConfig(initial_soc=.5,finite_batch=False))
    for _ in range(150):env.step([4])
    assert env.request_charge(0,automatic=True)
    env.prepare_decision()
    assert env.robots[0]['charger_id']==0



def test_parking_compliance_is_not_rewarded_less_than_remaining_in_a_service_bay():
    class Wait:
        def act(self,observations,masks):return [4]*len(observations)
    rewards=[]
    for label in ('wait','compliant'):
        env,manifest=materialize('COO02',300100,5,100)
        policy=Wait() if label=='wait' else NearestTaskDispatcher(env)
        _,metrics=collect_episode(env,policy,dict(seed=manifest['seed'],scenario_hash=manifest['manifest_hash']))
        rewards.append(metrics['episode_reward'])
        if label=='compliant':assert all((r['x'],r['y'])==r['parking'] for r in env.robots)
    assert rewards[1]>=rewards[0]

