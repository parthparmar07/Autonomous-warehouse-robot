import json
from pathlib import Path

import pytest
import torch

from src.training import runner
from src.training.learner import LearnerConfig,RecurrentLearner
from src.evaluation.scenarios import make_scenario


def arguments():
    args=runner.parser().parse_args([])
    for key,value in dict(num_agents=1,stage='C0',tiny=True,horizon=20,steps=40,
            bc_episodes=1,bc_epochs=1,dagger_rounds=0,eval_episodes=1,eval_every=10,
            checkpoint_every=1,hidden=16,batch_size=1,burn_in=0,unroll=2,
            train_every=4,threads=1,publish_artifact=False).items():setattr(args,key,value)
    return args


def test_failed_atomic_save_preserves_previous_checkpoint(tmp_path,monkeypatch):
    args=arguments();target=tmp_path/'latest.pt';target.write_bytes(b'previous-good-checkpoint')
    class Model:
        environment_steps=15;updates=2
        def checkpoint(self,*args,**kwargs):return {}
    def fail(payload,path):
        Path(path).write_bytes(b'incomplete')
        raise OSError('simulated disk interruption')
    monkeypatch.setattr(torch,'save',fail)
    with pytest.raises(OSError,match='interruption'):
        runner.save_training_checkpoint(Model(),dict(run_id='test'),args,3,target)
    assert target.read_bytes()==b'previous-good-checkpoint'
    assert not target.with_name(target.name+'.tmp').exists()


def test_interrupted_training_resumes_saved_episode_without_repeating_bc(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'LOADED_SOURCE',runner.provenance(tmp_path))
    args=arguments();args.checkpoint_pointer=tmp_path/'pointer.json'
    def interrupt(*unused):raise RuntimeError('simulated process interruption')
    with pytest.raises(RuntimeError,match='process interruption'):
        runner.train(args,evaluation_callback=interrupt)
    pointer=json.loads(args.checkpoint_pointer.read_text())
    checkpoint=Path(pointer['checkpoint'])
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    saved_steps=payload['environment_steps'];cursor=payload['metadata']['training_cursor']
    assert saved_steps>=10 and cursor['next_episode']>=1 and payload['demonstrations']
    args.resume=checkpoint;args.bc_episodes=0;args.eval_every=0;args.steps=saved_steps+20
    model,metadata,_=runner.train(args)
    assert model.environment_steps==args.steps and metadata['expert_preparation_restored']
    assert metadata['demonstrations']==payload['metadata']['demonstrations']
    assert model.demonstrations.episodes
    expected_seed=100000+(args.seed*997+cursor['next_episode'])%90000
    _,expected=runner.training_case(args,expected_seed,cursor['next_episode'])
    previous_count=len(payload['metadata']['scenario_hashes'])
    assert metadata['scenario_hashes'][previous_count]==expected['scenario_hash']
    assert metadata['training_cursor']['next_episode']>cursor['next_episode']


def test_stage_change_allows_new_exploration_decay_but_same_stage_does_not():
    torch.set_num_threads(1)
    args=arguments()
    env,_=make_scenario(100001,1,'C0',tiny=True,horizon=20)
    cfg=LearnerConfig(hidden=args.hidden,batch_size=1,burn_in=0,unroll=2)
    model=RecurrentLearner('qmix',1,env.observation_dim,len(env.get_training_state()),config=cfg)
    cursor=dict(stage='C0',next_episode=7,profile=args.profile,
        runtime_settings=runner.runtime_settings(args),epsilon_start=args.epsilon_start,
        **{k:getattr(args,k) for k in ('seed','scaling','tiny','horizon','scenario_mix','epsilon_steps','train_every')})
    payload=model.checkpoint(dict(training_cursor=cursor),include_replay=True)
    args.epsilon_steps+=100
    with pytest.raises(ValueError,match='epsilon_steps'):runner.resume_learner(payload,args,env,cfg)
    args.stage='C0.5'
    assert runner.resume_learner(payload,args,env,cfg)[1]==0
