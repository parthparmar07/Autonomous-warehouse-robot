from types import SimpleNamespace
from collections import defaultdict
import numpy as np
import pytest
from src.training.demonstrations import demonstration_schedule, demonstration_count, demonstration_case
from src.training.learner import SequenceReplay


def args(stage='C2', balanced=True):
    return SimpleNamespace(stage=stage,num_agents=5,balanced_demonstrations=balanced,
        tiny=False,scaling='fixed_map',profile='core',horizon=400,
        potential_weight=2.,gamma=.99,idle_bay_cost=.02,delivery_reward=10.,conflict_cost=.05)


def test_all_prerequisites_receive_every_worker_and_heading():
    schedule=demonstration_schedule(args())
    roles=defaultdict(set)
    for kind,family,role in schedule:roles[kind,family].add(role)
    assert ('generic','C0') in roles and ('generic','C2') in roles
    assert ('skill','CHARGE_CYCLE') in roles and ('skill','COO01') in roles
    assert all(v==set(range(20)) for v in roles.values())
    assert demonstration_count(args(),40)==len(schedule)
    assert demonstration_count(args(),0)==0
    assert demonstration_count(args(balanced=False),40)==40


def test_direct_c5_balancing_respects_requested_budget_and_covers_current_roles():
    from src.training.runner import parser
    a=parser().parse_args(['--stage','C5','--bc-episodes','60','--balanced-demonstrations'])
    assert demonstration_count(a,60)==60
    schedule=demonstration_schedule(a)[:60]
    generic=[(family,role) for kind,family,role in schedule if kind=='generic']
    assert {family for family,role in generic}=={'C5'}
    assert {role for family,role in generic}==set(range(20))
    assert len({family for kind,family,role in schedule if kind=='skill'})==14


def test_balanced_cases_are_training_only_and_materialize_selected_role():
    a=args('C0.5');schedule=demonstration_schedule(a)
    index=schedule.index(('generic','C0',19))
    env,manifest=demonstration_case(a,123456,index)
    assert manifest['split']=='train' and manifest['role_index']==19
    assert env.robots[4]['active'] and env.robots[4]['orientation']==3
    with pytest.raises(ValueError,match='training-partition'):
        demonstration_case(a,301000,0)


def test_prior_stage_demonstrations_retain_original_profile():
    a=args('C3');a.profile='robust';a.stage_profiles={'C2':'core','C3':'robust'}
    index=demonstration_schedule(a).index(('generic','C2',0))
    _,manifest=demonstration_case(a,123456,index)
    assert manifest['profile']=='core'
    with pytest.raises(ValueError,match='training-partition'):
        demonstration_case(a,200000,index)


def episode(value):
    return dict(observations=np.full((2,1,1),value,dtype=float),
        states=np.zeros((2,1)),masks=np.ones((2,1,7),bool),alive=np.ones((2,1),bool),
        actions=np.zeros((1,1),int),rewards=np.zeros(1),terminated=np.ones(1,bool))


def test_representative_experts_and_prior_stage_sampling_survive_capacity():
    replay=SequenceReplay(10);replay.set_stage('C0')
    replay.add_representative([episode(i) for i in range(30)])
    values=[e['observations'][0,0,0] for e in replay.episodes]
    assert values[0]==0 and values[-1]==29
    replay.set_stage('C1',.2)
    replay.add_representative([episode(100+i) for i in range(50)])
    assert len(replay.prior_episodes)==2
    sample=replay.sample(200,0,1,np.random.default_rng(3))
    initial=sample['observations'][:,0,0,0]
    assert (initial<100).any() and (initial>=100).any()


def test_representative_storage_preserves_families_under_stride_aliasing():
    replay=SequenceReplay(2)
    # A flat linspace would select endpoints 0 and 4: both from family A.
    replay.add_representative([episode(i) for i in range(5)],['A','B','B','B','A'])
    values={int(e['observations'][0,0,0]) for e in replay.episodes}
    assert values.intersection({0,4}) and values.intersection({1,2,3})
    with pytest.raises(ValueError,match='labels'):
        replay.add_representative([episode(0)],[])
    with pytest.raises(ValueError,match='capacity'):
        replay.add_representative([episode(i) for i in range(3)],['A','B','C'])


def test_every_generated_family_survives_small_balanced_reservoir():
    schedule=demonstration_schedule(args('C5'))
    groups=[(kind,family) for kind,family,role in schedule]
    replay=SequenceReplay(len(set(groups))*2)
    replay.add_representative([episode(i) for i in range(len(groups))],groups)
    retained=[groups[int(e['observations'][0,0,0])] for e in replay.episodes]
    assert set(retained)==set(groups)
    assert all(retained.count(group)==2 for group in set(groups))
