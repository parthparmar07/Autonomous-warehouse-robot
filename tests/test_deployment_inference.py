import numpy as np
import pytest
from src.deployment.inference import predict_actions
from src.deployment import policies
from src.environment.warehouse_env import WarehouseEnv
from src.training.learner import RecurrentLearner

def test_missing_incompatible_and_corrupt_checkpoints_fail_closed(tmp_path,monkeypatch):
    monkeypatch.setattr(policies,'_ARTIFACT_DIR',tmp_path)
    env=WarehouseEnv(num_agents=2)
    obs=[env.get_local_observation(i) for i in range(2)]
    masks=[[True,False,False,False,False,False,False],[False,True,False,False,False,False,False]]
    with pytest.raises(FileNotFoundError):predict_actions(obs,masks)
    model=RecurrentLearner('qmix',2,env.observation_dim,len(env.get_training_state()))
    path=policies.specific_checkpoint('qmix',2)
    policies.save_checkpoint(model.checkpoint({}),path)
    assert predict_actions(obs,masks)==[0,1]
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='hash'):predict_actions(obs,masks)
