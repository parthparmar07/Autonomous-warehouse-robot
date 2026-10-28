"""Export the decentralized recurrent actor, including masks and explicit memory."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.environment.spec import ENV_VERSION,OBS_VERSION

class ActorExport(nn.Module):
    def __init__(self,actor,action_dim):
        super().__init__();self.actor=actor;self.action_dim=action_dim
    def forward(self,observation,hidden,previous,mask):
        q,next_hidden=self.actor(observation,hidden,previous)
        actions=q.masked_fill(~mask,-1e30).argmax(-1)
        return actions,next_hidden,F.one_hot(actions,self.action_dim).float()

def export_actor(model,path):
    import onnx
    import onnxruntime as ort
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    model.actor.eval()
    device=model.device;n=model.num_agents
    wrapper=ActorExport(model.actor,model.action_dim)
    example=(torch.zeros(n,model.obs_dim,device=device),torch.zeros(n,model.config.hidden,device=device),
             torch.zeros(n,model.action_dim,device=device),torch.ones(n,model.action_dim,dtype=torch.bool,device=device))
    names=['observation','hidden','previous','mask','actions','next_hidden','next_previous']
    torch.onnx.export(wrapper,example,str(path),input_names=names[:4],output_names=names[4:],
        dynamic_axes={name:{0:'robots'} for name in names},opset_version=18,dynamo=False)
    onnx.checker.check_model(onnx.load(path))
    session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
    rng=np.random.default_rng(19);hidden=example[1];previous=example[2]
    ort_hidden=hidden.cpu().numpy();ort_previous=previous.cpu().numpy()
    for _ in range(20):
        obs=rng.normal(size=(n,model.obs_dim)).astype(np.float32)
        mask=rng.random((n,model.action_dim))>.3;mask[:,4]=True
        with torch.no_grad():
            actions,hidden,previous=wrapper(torch.as_tensor(obs,device=device),hidden,previous,torch.as_tensor(mask,device=device))
        ort_actions,ort_hidden,ort_previous=session.run(None,dict(observation=obs,hidden=ort_hidden,previous=ort_previous,mask=mask))
        np.testing.assert_array_equal(ort_actions,actions.cpu().numpy())
        np.testing.assert_allclose(ort_hidden,hidden.cpu().numpy(),rtol=1e-4,atol=1e-5)
    metadata=dict(observation_dim=model.obs_dim,hidden=model.config.hidden,actions=model.action_dim,
        environment_version=ENV_VERSION,observation_version=OBS_VERSION,training_metadata=getattr(model,'metadata',{}),
        trained_num_agents=model.num_agents,algorithm=model.algorithm,updates=model.updates,
        actor_only=True,explicit_recurrent_state=True,parity_ticks=20,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        note='Export/parity verification does not establish policy quality or fleet transfer.')
    path.with_suffix('.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return metadata
