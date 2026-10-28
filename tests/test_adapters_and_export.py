import pytest
import numpy as np
from src.environment.runtime_config import RuntimeConfig

def test_parallel_adapter_spaces_horizon_and_reset():
    pytest.importorskip('gymnasium')
    from src.environment.parallel_env import WarehouseParallelEnv
    parallel=WarehouseParallelEnv(num_agents=2,config=RuntimeConfig(horizon=2))
    observations,_=parallel.reset(seed=42,options=dict(num_tasks=2))
    assert all(parallel.observation_space(a).contains(o) for a,o in observations.items())
    parallel.step({a:4 for a in parallel.agents})
    _,rewards,terminated,truncated,_=parallel.step({a:4 for a in parallel.agents})
    assert len(set(rewards.values()))==1 and all(truncated.values()) and not any(terminated.values())
    assert not parallel.agents
    observations,_=parallel.reset(seed=42,options=dict(num_tasks=2))
    assert len(observations)==2

def test_recurrent_onnx_multistep_parity(tmp_path):
    pytest.importorskip('onnx');pytest.importorskip('onnxruntime')
    import torch
    from src.training.learner import RecurrentLearner
    from src.deployment.onnx_policy import export_actor
    torch.set_num_threads(1)
    model=RecurrentLearner('qmix',2,328,250)
    metadata=export_actor(model,tmp_path/'actor.onnx')
    assert metadata['parity_ticks']==20 and metadata['explicit_recurrent_state']
