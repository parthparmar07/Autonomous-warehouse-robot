import numpy as np
import pytest
import torch
from src.training.learner import RecurrentLearner, LearnerConfig, MonotonicMixer, masked_double_q
from src.training.runner import collect_episode
from src.evaluation.scenarios import make_scenario
from src.baselines.nearest_task import NearestTaskDispatcher


def test_double_q_uses_legal_online_argmax_and_target_value():
    online=torch.tensor([[1000.,2.,3.]])
    target=torch.tensor([[2000.,99.,4.]])
    assert masked_double_q(online,target,torch.tensor([[False,True,True]])).item()==4
    with pytest.raises(ValueError):masked_double_q(online,target,torch.zeros_like(online).bool())


@pytest.mark.parametrize('algorithm',['iql','vdn','qmix'])
def test_sequence_targets_updates_resume_and_no_memory_leak(algorithm):
    torch.set_num_threads(1)
    env,manifest=make_scenario(100001,1,'C0',horizon=40,tiny=True)
    data,_=collect_episode(env,NearestTaskDispatcher(env),manifest)
    model=RecurrentLearner(algorithm,1,env.observation_dim,len(env.get_training_state()),config=LearnerConfig(batch_size=2,burn_in=2,unroll=5,target_every=2))
    model.replay.add(data)
    initial={k:v.clone() for k,v in model.target_actor.state_dict().items()}
    assert np.isfinite(model.train_step()['loss'])
    assert all(torch.equal(v,model.target_actor.state_dict()[k]) for k,v in initial.items())
    assert np.isfinite(model.train_step()['loss'])
    assert all(torch.equal(v,model.target_actor.state_dict()[k]) for k,v in model.actor.state_dict().items())
    loaded=RecurrentLearner.from_checkpoint(model.checkpoint(include_replay=True),resume=True)
    assert loaded.updates==2 and len(loaded.replay.episodes)==1
    observations=data['observations'][0]
    masks=data['masks'][0]
    model.reset_hidden_states();first=model.act(observations,masks)
    model.act(observations,masks)
    model.reset_hidden_states();assert model.act(observations,masks)==first


def test_mixer_monotonicity_is_an_actual_gradient_test_at_twenty_agents():
    mixer=MonotonicMixer(20,250)
    q=torch.randn(3,20,requires_grad=True)
    mixer(q,torch.randn(3,250)).sum().backward()
    assert torch.isfinite(q.grad).all() and (q.grad>=-1e-7).all()


def test_padding_does_not_affect_loss():
    torch.set_num_threads(1)
    env,manifest=make_scenario(100001,1,'C0',horizon=40,tiny=True)
    data,_=collect_episode(env,NearestTaskDispatcher(env),manifest)
    model=RecurrentLearner('qmix',1,env.observation_dim,len(env.get_training_state()),config=LearnerConfig(batch_size=1,burn_in=2,unroll=100))
    model.replay.add(data)
    batch=model.replay.sample(1,2,100,np.random.default_rng(1))
    # Padded observations cannot change recurrent state or contribute to TD loss.
    clone=RecurrentLearner.from_checkpoint(model.checkpoint())
    original=model.train_step(batch)['loss']
    batch['observations'][~batch['valid']]=12345
    altered=clone.train_step(batch)['loss']
    assert original==pytest.approx(altered,abs=1e-5)
