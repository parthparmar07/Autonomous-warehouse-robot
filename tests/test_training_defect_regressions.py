from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from src.environment.warehouse_env import WarehouseEnv
from src.environment.task_generator import schedule_order
from src.evaluation.scenarios import make_scenario
from src.evaluation.policy_cases import materialize
from src.training.learner import RecurrentLearner, LearnerConfig, weighted_imitation_loss
from src.training import runner


@pytest.mark.parametrize('algorithm',['qmix','vdn','iql'])
def test_empty_battery_revival_receives_actor_credit(algorithm):
    torch.set_num_threads(1);torch.manual_seed(9)
    env=WarehouseEnv(grid_size=(8,8),num_agents=1,obstacles=[],safety_corridors=[],charger_bays=[(5,5)])
    robot=env.robots[0];robot.update(x=5,y=5,energy_eu=1)
    env.request_charge(0);robot.update(energy_eu=0,active=False,mode='DEPLETED')
    class Service:
        def act(self,*args):return [5]
    data,_=runner.collect_episode(env,Service(),dict(seed=100001,scenario_hash='diagnostic'),max_steps=1)
    assert data['alive'].tolist()==[[False],[True]]
    assert robot['energy_eu']==2
    model=RecurrentLearner(algorithm,1,env.observation_dim,len(env.get_training_state()),
        config=LearnerConfig(batch_size=1,burn_in=0,unroll=1))
    model.replay.add(data)
    model.train_step()
    assert sum(p.grad.abs().sum().item() for p in model.actor.parameters() if p.grad is not None)>0


def test_bc_balances_labels_across_time_with_one_live_robot():
    logits=torch.tensor([[[0.,1.,2.]]]*4,requires_grad=True)
    labels=torch.tensor([[0],[0],[0],[2]])
    weights=torch.tensor([1/3,1.,1.])
    eligible=torch.ones(4,1)
    loss=weighted_imitation_loss(logits,labels,eligible,weights)
    expected=F.cross_entropy(logits.reshape(4,3),labels.flatten(),weight=weights)
    assert loss.item()==pytest.approx(expected.item())
    assert loss.item()!=pytest.approx(F.cross_entropy(logits.reshape(4,3),labels.flatten()).item())
    eligible[-1]=0
    masked=weighted_imitation_loss(logits,labels,eligible,weights)
    masked.backward()
    assert logits.grad[-1].abs().sum()==0


def resume_fixture(algorithm='qmix'):
    args=runner.parser(algorithm).parse_args([]);args.num_agents=1;args.tiny=True;args.stage='C0'
    env,_=make_scenario(100001,1,'C0',tiny=True,horizon=args.horizon)
    config=LearnerConfig(batch_size=args.batch_size,burn_in=args.burn_in,unroll=args.unroll,
        learning_rate=args.learning_rate,gamma=args.gamma,target_every=args.target_every,bc_aux_weight=args.bc_aux_weight)
    model=RecurrentLearner(algorithm,1,env.observation_dim,len(env.get_training_state()),config=config)
    cursor=dict(stage='C0',next_episode=17,**{k:getattr(args,k) for k in
        ('seed','scaling','tiny','horizon','scenario_mix','epsilon_steps','train_every')})
    return args,env,config,model,model.checkpoint(dict(training_cursor=cursor),include_replay=True)


@pytest.mark.parametrize('field,value',[('algorithm','iql'),('state_dim',999),('action_dim',99),('num_agents',9)])
def test_resume_rejects_incompatible_model(field,value):
    args,env,config,_,payload=resume_fixture()
    payload[field]=value
    with pytest.raises(ValueError,match='mismatch'):runner.resume_learner(payload,args,env,config)


def test_resume_rejects_ignored_config_and_missing_cursor():
    args,env,config,_,payload=resume_fixture()
    requested=LearnerConfig(**dict(asdict(config),bc_aux_weight=.1))
    with pytest.raises(ValueError,match='bc_aux_weight'):runner.resume_learner(payload,args,env,requested)
    requested=LearnerConfig(**dict(asdict(config),learning_rate=.001))
    with pytest.raises(ValueError,match='learning_rate'):runner.resume_learner(payload,args,env,requested)
    payload['metadata']={}
    with pytest.raises(ValueError,match='cursor'):runner.resume_learner(payload,args,env,config)


def test_resume_continues_scenario_cursor_and_restores_optimizer_rng():
    args,env,config,model,payload=resume_fixture()
    from src.baselines.nearest_task import NearestTaskDispatcher
    data,_=runner.collect_episode(env,NearestTaskDispatcher(env),dict(seed=100001,scenario_hash='diagnostic'))
    model.replay.add(data);model.train_step()
    payload=deepcopy(model.checkpoint(payload['metadata'],include_replay=True))
    expected=model.train_step()
    loaded,cursor=runner.resume_learner(payload,args,env,config)
    actual=loaded.train_step()
    assert cursor==17 and actual['loss']==pytest.approx(expected['loss'],abs=1e-7)
    assert all(torch.equal(v,loaded.actor.state_dict()[k]) for k,v in model.actor.state_dict().items())
    args.stage='C1'
    assert runner.resume_learner(payload,args,env,config)[1]==0
    args.horizon+=1
    with pytest.raises(ValueError,match='horizon'):runner.resume_learner(payload,args,env,config)


def test_dagger_aggregates_all_rounds_and_retains_expert_labels(monkeypatch):
    from src.training.learner import SequenceReplay
    def episode(marker):
        return dict(observations=np.full((2,1,2),marker,dtype=np.float32),states=np.zeros((2,2)),
            masks=np.ones((2,1,7),bool),alive=np.ones((2,1),bool),actions=np.array([[4]]),
            expert_actions=np.array([[5]]),rewards=np.zeros(1),terminated=np.zeros(1,bool))
    original=episode(0);original.pop('expert_actions')
    corrections=iter([episode(1),episode(2)])
    calls=[]
    model=SimpleNamespace(demonstrations=SequenceReplay())
    model.behavior_clone=lambda data,epochs:calls.append([int(e['observations'][0,0,0]) for e in data]) or 0.
    monkeypatch.setattr(runner,'training_case',lambda *a:(None,{}))
    monkeypatch.setattr(runner,'collect_episode',lambda *a,**k:(next(corrections),{}))
    monkeypatch.setattr(runner,'NearestTaskDispatcher',lambda env:None)
    args=SimpleNamespace(bc_epochs=1,dagger_rounds=2,bc_episodes=1,seed=11)
    runner.aggregate_dagger(model,[original],args,{})
    assert calls==[[0],[0,1],[0,1,2]]
    assert len(model.demonstrations.episodes)==3
    assert [int(e['expert_actions'][0,0]) for e in model.demonstrations.episodes]==[4,5,5]
    batch=model.demonstrations.sample(32,0,1,np.random.default_rng(9))
    assert set(batch['expert_actions'].flatten())=={4,5}
    assert np.all(batch['actions']==4)


def test_auxiliary_imitation_uses_correction_labels_not_executed_actions():
    torch.set_num_threads(1);torch.manual_seed(3)
    env,_=make_scenario(100001,1,'C0',tiny=True,horizon=20)
    from src.baselines.nearest_task import NearestTaskDispatcher
    data,_=runner.collect_episode(env,NearestTaskDispatcher(env),dict(seed=100001,scenario_hash='diagnostic'))
    model=RecurrentLearner('qmix',1,env.observation_dim,len(env.get_training_state()),
        config=LearnerConfig(bc_aux_weight=1,batch_size=1,burn_in=0,unroll=2))
    model.replay.add(data)
    corrected=deepcopy(data);corrected['expert_actions']=np.full_like(data['actions'],5)
    model.demonstrations.add(corrected)
    clone=RecurrentLearner.from_checkpoint(deepcopy(model.checkpoint()))
    clone.replay.add(data);clone.demonstrations.add(data)
    assert model.train_step()['loss']!=pytest.approx(clone.train_step()['loss'],abs=1e-7)


def test_future_order_deadlines_keep_release_relative_slack():
    original=dict(task_id=0,creation_timestep=5,deadline=45)
    shifted=schedule_order(original,2000)
    assert shifted['deadline']==2040 and original['deadline']==45
    for seed in (100001,300000):
        env,_=make_scenario(seed,5,'C5',horizon=2000)
        arrivals=[e for e in env._events if e['type']=='order']
        assert arrivals and all(e['order']['deadline']>e['tick']==e['order']['creation_timestep'] for e in arrivals)
    for case in ('COO04','DEM03','EVA05'):
        env,_=materialize(case,100001)
        arrivals=[e for e in env._events if e['type']=='order']
        assert arrivals and all(e['order']['deadline']>e['tick'] for e in arrivals)


def test_interrupted_service_reverts_mode_to_navigation():
    env=WarehouseEnv(grid_size=(8,8),num_agents=1,obstacles=[(2,0)],safety_corridors=[],
        pickup_bays=[(2,1)],drop_bays=[(5,5)])
    robot=env.robots[0];robot.update(x=2,y=1,orientation=1)
    env.task_queue=[dict(task_id=0,pickup=(2,1),delivery=(5,5),status='pending')]
    env.prepare_decision()
    env.step([5])
    assert robot['mode']=='PICKING' and robot['service_remaining']==2
    env.step([0])  # exploratory move off the pickup bay interrupts service
    assert robot['service_remaining']==0 and robot['service_key'] is None
    assert robot['mode']=='TO_PICKUP'
    env.assert_invariants()


def test_low_soc_unloaded_robot_can_request_charge_and_backs_off_when_offline():
    env=WarehouseEnv(grid_size=(8,8),num_agents=1,obstacles=[],safety_corridors=[],charger_bays=[(7,7)])
    robot=env.robots[0];robot.update(x=1,y=1,energy_eu=2.0)
    env.prepare_decision()
    assert env.get_action_mask(0)[6]
    assert env.request_charge(0)
    assert robot['charger_id']==0 and robot['mode']=='TO_CHARGER'
    # If all chargers are offline, a failed request backs off instead of looping every tick.
    env=WarehouseEnv(grid_size=(8,8),num_agents=1,obstacles=[],safety_corridors=[],charger_bays=[(7,7)])
    env.chargers[0]['online']=False
    env.robots[0].update(x=1,y=1,energy_eu=10.0)
    env.prepare_decision()
    assert env.get_action_mask(0)[6]
    env.step([6])
    assert env.episode_metrics['charger_rejections']==1
    assert not env.get_action_mask(0)[6]


def test_transfer_bay_is_sensed_in_local_observation_grid():
    env=WarehouseEnv(grid_size=(8,8),num_agents=1,obstacles=[],safety_corridors=[],transfer_bays=[(3,2)])
    env.robots[0].update(x=3,y=3,orientation=0)
    frame,_=env.get_sensor_frame(0)
    assert frame[1,2,0]==1 and frame[1,2,5]==1


def test_qmix_behavior_clone_calibrates_mixer_and_records_policy_entropy():
    torch.set_num_threads(1);torch.manual_seed(5)
    env,_=make_scenario(100001,1,'C0',tiny=True,horizon=20)
    from src.baselines.nearest_task import NearestTaskDispatcher
    data,metrics=runner.collect_episode(env,NearestTaskDispatcher(env),dict(seed=100001,scenario_hash='diagnostic'))
    model=RecurrentLearner('qmix',1,env.observation_dim,len(env.get_training_state()),
        config=LearnerConfig(bc_aux_weight=0.5,batch_size=1,burn_in=0,unroll=4))
    initial_mixer={k:v.clone() for k,v in model.mixer.state_dict().items()}
    model.behavior_clone([data],epochs=2)
    assert any(not torch.equal(v,model.mixer.state_dict()[k]) for k,v in initial_mixer.items())
    assert all(torch.equal(v,model.target_mixer.state_dict()[k]) for k,v in model.mixer.state_dict().items())
    _,eval_metrics=runner.collect_episode(env,model,dict(seed=100001,scenario_hash='diagnostic'))
    assert eval_metrics['behavior_entropy_mean'][0]>0


