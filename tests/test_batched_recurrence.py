from copy import deepcopy
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from src.training.learner import RecurrentLearner,LearnerConfig


def test_batched_recurrence_matches_cell_reference_values_and_gradients():
    torch.set_num_threads(1);torch.manual_seed(2)
    model=RecurrentLearner('qmix',2,6,9,config=LearnerConfig(hidden=8,burn_in=3,unroll=5))
    encoder=deepcopy(model.actor.encoder);head=deepcopy(model.actor.head)
    cell=nn.GRUCell(8,8)
    cell.load_state_dict({k.removesuffix('_l0'):v for k,v in model.actor.gru.state_dict().items()})
    obs=torch.randn(3,9,2,6)
    actions=torch.randint(7,(3,8,2))
    valid=torch.tensor([[False,False,True,True,True,True,True,True,True],
                        [True,True,True,True,True,False,False,False,False],
                        [False,False,False,True,True,True,False,False,False]])
    hidden=torch.zeros(3,2,8);previous=torch.zeros(3,2,7);values=[]
    for t in range(9):
        with torch.set_grad_enabled(t>=3):
            encoded=encoder(torch.cat((obs[:,t],previous),-1)).reshape(-1,8)
            proposed=cell(encoded,hidden.reshape(-1,8)).reshape(3,2,8)
            values.append(head(proposed))
            hidden=torch.where(valid[:,t,None,None],proposed,hidden)
        if t<3:hidden=hidden.detach()
        if t<8:previous=F.one_hot(actions[:,t],7).float()*valid[:,t,None,None]
    reference=torch.stack(values,1)
    actual=model._roll(model.actor,obs,actions,valid,True)
    mask=valid[:,3:]
    assert torch.allclose(actual[:,3:][mask],reference[:,3:][mask],atol=1e-6,rtol=1e-5)
    actual[:,3:][mask].square().mean().backward()
    reference[:,3:][mask].square().mean().backward()
    for current,expected in zip(model.actor.encoder.parameters(),encoder.parameters()):
        assert torch.allclose(current.grad,expected.grad,atol=1e-6,rtol=1e-4)
    for name,value in model.actor.gru.named_parameters():
        assert torch.allclose(value.grad,dict(cell.named_parameters())[name.removesuffix('_l0')].grad,atol=1e-6,rtol=1e-4)


def test_historical_gru_cell_checkpoint_migrates_without_policy_change():
    torch.manual_seed(4)
    model=RecurrentLearner('qmix',2,6,9,config=LearnerConfig(hidden=8))
    payload=deepcopy(model.checkpoint())
    for key in ('actor','target_actor'):
        payload[key]={name.removesuffix('_l0') if name.startswith('gru.') else name:value for name,value in payload[key].items()}
    payload.pop('recurrent_backend')
    loaded=RecurrentLearner.from_checkpoint(payload)
    from src.utils.hashing import policy_fingerprint
    assert policy_fingerprint(payload)==policy_fingerprint(loaded.checkpoint())
    obs=torch.randn(2,6);hidden=torch.randn(2,8);previous=torch.zeros(2,7)
    a,h=model.actor(obs,hidden,previous);b,g=loaded.actor(obs,hidden,previous)
    assert torch.equal(a,b) and torch.equal(h,g)
