"""Executed only when the test environment has a usable CUDA device."""
from copy import deepcopy
import io
import numpy as np
import pytest
import torch
from src.training.learner import RecurrentLearner,LearnerConfig
from src.training.runner import collect_episode
from src.evaluation.scenarios import make_scenario
from src.baselines.nearest_task import NearestTaskDispatcher


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA device unavailable')
@pytest.mark.parametrize('algorithm',['qmix','vdn','iql'])
def test_cuda_update_matches_cpu_loss_and_restores_cuda_rng(algorithm):
    torch.set_num_threads(1);torch.manual_seed(7)
    env,manifest=make_scenario(100001,1,'C0',tiny=True,horizon=30)
    episode,_=collect_episode(env,NearestTaskDispatcher(env),manifest)
    cpu=RecurrentLearner(algorithm,1,env.observation_dim,len(env.get_training_state()),
        config=LearnerConfig(batch_size=2,burn_in=2,unroll=5))
    cpu.replay.add(episode)
    gpu=RecurrentLearner.from_checkpoint(deepcopy(cpu.checkpoint()),device='cuda')
    batch=cpu.replay.sample(2,2,5,np.random.default_rng(3))
    expected=cpu.train_step(batch);actual=gpu.train_step(batch)
    assert actual['loss']==pytest.approx(expected['loss'],rel=1e-4,abs=1e-5)
    assert next(gpu.actor.parameters()).is_cuda
    gpu.replay.add(episode)
    buffer=io.BytesIO();torch.save(gpu.checkpoint(include_replay=True),buffer);buffer.seek(0)
    saved=torch.load(buffer,map_location='cpu',weights_only=True)
    assert all(not value.is_cuda for item in saved['replay'] for value in item.values())
    draw=torch.rand(4,device='cuda').cpu()
    loaded=RecurrentLearner.from_checkpoint(saved,device='cuda',resume=True)
    assert torch.equal(draw,torch.rand(4,device='cuda').cpu())
    assert loaded.updates==1 and next(loaded.actor.parameters()).is_cuda and len(loaded.replay.episodes)==1
    assert all(state['exp_avg'].is_cuda for state in loaded.optimizer.state.values())
