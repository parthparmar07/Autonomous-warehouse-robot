"""Matched shared recurrent IQL/VDN/QMIX with sequence replay and legal Double-Q."""
from __future__ import annotations
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from src.environment.spec import ENV_VERSION, OBS_VERSION, ACTION_DIM


@dataclass
class LearnerConfig:
    hidden: int = 64
    learning_rate: float = .0003
    gamma: float = .99
    batch_size: int = 8
    burn_in: int = 20
    unroll: int = 40
    replay_episodes: int = 500
    target_every: int = 200
    gradient_clip: float = 10
    bc_aux_weight: float = 0.0
    mixer_hidden: int = 32

    def __post_init__(self):
        for key in ('hidden','batch_size','unroll','replay_episodes','target_every','mixer_hidden'):
            value=getattr(self,key)
            if isinstance(value,bool) or not isinstance(value,int) or value<1:raise ValueError(f'Invalid {key}')
        if not isinstance(self.burn_in,int) or self.burn_in<0:raise ValueError('Invalid burn-in')
        if not 0<=self.gamma<=1 or not np.isfinite(self.gamma):raise ValueError('Invalid discount')
        if not all(np.isfinite(v) and v>0 for v in (self.learning_rate,self.gradient_clip)):raise ValueError('Invalid optimizer configuration')
        if not np.isfinite(self.bc_aux_weight) or self.bc_aux_weight<0:raise ValueError('Invalid imitation auxiliary weight')


class SharedAgent(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(obs_dim+action_dim,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU())
        self.gru=nn.GRU(hidden,hidden,batch_first=True)
        self.head=nn.Linear(hidden,action_dim)
        self.hidden=hidden

    def forward(self, obs, hidden, previous):
        shape=obs.shape[:-1]
        encoded=self.encoder(torch.cat((obs,previous),dim=-1)).reshape(-1,1,self.hidden)
        _,next_hidden=self.gru(encoded,hidden.reshape(1,-1,self.hidden))
        next_hidden=next_hidden.reshape(*shape,self.hidden)
        return self.head(next_hidden),next_hidden


class MonotonicMixer(nn.Module):
    def __init__(self, num_agents, state_dim, hidden=32):
        super().__init__()
        self.n,self.h=num_agents,hidden
        self.state_encoder=nn.Sequential(nn.Linear(state_dim,64),nn.ReLU())
        self.w1=nn.Linear(64,num_agents*hidden)
        self.b1=nn.Linear(64,hidden)
        self.w2=nn.Linear(64,hidden)
        self.b2=nn.Linear(64,1)

    def forward(self, utilities, state):
        shape=utilities.shape[:-1]
        latent=self.state_encoder(state).reshape(-1,64)
        q=utilities.reshape(-1,1,self.n)
        hidden=F.elu(torch.bmm(q,self.w1(latent).abs().reshape(-1,self.n,self.h)).squeeze(1)+self.b1(latent))
        result=torch.bmm(hidden.unsqueeze(1),self.w2(latent).abs().unsqueeze(2)).flatten()+self.b2(latent).flatten()
        return result.reshape(shape)


def masked_double_q(online, target, masks):
    masks=masks.bool()
    if not bool(masks.any(dim=-1).all()):
        raise ValueError('Every next state requires at least one legal action')
    actions=online.masked_fill(~masks,-torch.inf).argmax(-1,keepdim=True)
    return target.gather(-1,actions).squeeze(-1)


def learning_eligibility(alive, masks):
    """Include recovery decisions even when the robot cannot currently move.

    WAIT-only inactive robots remain excluded. Healthy empty robots connected
    to external power can legally SERVICE and must receive action credit.
    Works with both NumPy and torch boolean arrays, including old replay.
    """
    return alive | masks[..., :4].any(-1) | masks[..., 5:].any(-1)


def weighted_imitation_loss(utilities, labels, eligible, weights):
    per_agent=F.cross_entropy(utilities.reshape(-1,utilities.shape[-1]),labels.reshape(-1),
                             weight=weights,reduction='none').reshape(labels.shape)
    return (per_agent*eligible).sum()/(weights[labels]*eligible).sum().clamp_min(1e-8)


class SequenceReplay:
    def __init__(self, capacity=500):
        self.episodes=deque(maxlen=capacity)
        self.prior_episodes=[]
        self.stage=None
        self.prior_fraction=0.0

    def set_stage(self,stage,prior_fraction=.2):
        if not 0<=prior_fraction<1:raise ValueError('Invalid prior-stage replay fraction')
        if self.stage is not None and self.stage!=stage:
            pool=self.prior_episodes+list(self.episodes)
            keep=max(1,int(self.episodes.maxlen*prior_fraction))
            indices=np.linspace(0,len(pool)-1,min(keep,len(pool)),dtype=int)
            self.prior_episodes=[pool[i] for i in indices]
            self.episodes.clear()
        self.stage,self.prior_fraction=stage,prior_fraction

    def add(self, episode):
        # Demonstrations and DAgger corrections must have the same replay schema.
        # Ordinary demonstrations execute their expert label; corrections retain
        # the separately queried label and their actual executed action history.
        episode=dict(episode)
        episode.setdefault('expert_actions',episode['actions'])
        length=len(episode['actions'])
        if length<1 or any(len(episode[key])!=length+1 for key in ('observations','states','masks','alive')) or any(len(episode[key])!=length for key in ('rewards','terminated')):
            raise ValueError('Episode must include its actual final decision state')
        if not all(np.isfinite(np.asarray(v)).all() for v in episode.values()):
            raise ValueError('Replay cannot contain non-finite data')
        actions=np.asarray(episode['actions'])
        labels=np.asarray(episode['expert_actions'])
        masks=np.asarray(episode['masks'],dtype=bool)
        if labels.shape!=actions.shape or (labels<0).any() or (labels>=masks.shape[-1]).any():
            raise ValueError('Replay expert labels are invalid')
        if (actions<0).any() or (actions>=masks.shape[-1]).any() or not masks.any(-1).all():
            raise ValueError('Replay actions/masks are invalid')
        if not np.take_along_axis(masks[:-1],actions[...,None],-1).all():
            raise ValueError('Replay action was illegal at its decision state')
        self.episodes.append({k:np.asarray(v).copy() for k,v in episode.items()})

    def add_representative(self, episodes, groups=None):
        """Retain a spread of a finite demonstration batch instead of its tail.

        Call set_stage first to retain a separate prior-stage reservoir. This
        method changes storage only; cloning can still train on every episode.
        """
        if groups is not None and len(groups)!=len(episodes):
            raise ValueError('Representative group labels must match episodes')
        if not episodes:
            return
        count=min(len(episodes),self.episodes.maxlen)
        if groups is None:
            selected=np.linspace(0,len(episodes)-1,count,dtype=int)
        else:
            members={}
            for index,group in enumerate(groups):members.setdefault(group,[]).append(index)
            if len(members)>count:
                raise ValueError('Replay capacity cannot retain every demonstration family')
            quotas={group:0 for group in members}
            remaining=count
            while remaining:
                for group,indices in members.items():
                    if quotas[group]<len(indices) and remaining:
                        quotas[group]+=1;remaining-=1
            # Evenly spread each family's quota across roles and DAgger rounds.
            selected=[]
            for group,indices in members.items():
                selected.extend(indices[int(i)] for i in np.linspace(0,len(indices)-1,quotas[group],dtype=int))
        for index in selected:
            self.add(episodes[int(index)])

    def sample(self, count, burn, unroll, rng):
        if not self.episodes:
            return None
        result=[]
        for _ in range(count):
            pool=self.prior_episodes if self.prior_episodes and rng.random()<self.prior_fraction else self.episodes
            episode=pool[int(rng.integers(len(pool)))]
            length=len(episode['actions'])
            start=int(rng.integers(length))
            indices=list(range(start-burn,start+unroll+1))
            rows={}
            for key,array in episode.items():
                is_state=key in ('observations','states','masks','alive')
                template=array[0]
                rows[key]=np.stack([array[t] if 0<=t<len(array) else np.zeros_like(template)
                                    for t in (indices if is_state else indices[:-1])])
            valid=np.asarray([0<=t<=length for t in indices])
            transition=np.asarray([0<=t<length for t in indices[:-1]])
            # Padded frames must not alter recurrent memory. Their availability set
            # still needs a legal no-op so max/argmax never produces NaN targets.
            rows['masks'][~valid,:,4]=True
            rows['valid']=valid
            rows['loss_mask']=transition
            result.append(rows)
        return {key:np.stack([r[key] for r in result]) for key in result[0]}


class RecurrentLearner:
    def __init__(self, algorithm, num_agents, obs_dim, state_dim, action_dim=ACTION_DIM, config=None, seed=11, device='cpu'):
        if algorithm not in ('iql','vdn','qmix','dqn'):
            raise ValueError('Unknown learning algorithm')
        self.algorithm=algorithm
        self.num_agents,self.obs_dim,self.state_dim,self.action_dim=num_agents,obs_dim,state_dim,action_dim
        self.config=config if isinstance(config,LearnerConfig) else LearnerConfig(**(config or {}))
        self.device=torch.device(device)
        self.rng=np.random.default_rng(seed)
        self.actor=SharedAgent(obs_dim,action_dim,self.config.hidden).to(self.device)
        self.target_actor=deepcopy(self.actor)
        self.actor.gru.flatten_parameters()
        self.target_actor.gru.flatten_parameters()
        self.mixer=MonotonicMixer(num_agents,state_dim,self.config.mixer_hidden).to(self.device) if algorithm=='qmix' else None
        self.target_mixer=deepcopy(self.mixer)
        self.parameters=list(self.actor.parameters())+(list(self.mixer.parameters()) if self.mixer else [])
        self.optimizer=torch.optim.Adam(self.parameters,lr=self.config.learning_rate)
        self.replay=SequenceReplay(self.config.replay_episodes)
        self.demonstrations=SequenceReplay(self.config.replay_episodes)
        self.updates,self.environment_steps=0,0
        self.epsilon=0.0
        self.training_kind='random_initialization'
        self.reset_hidden_states()

    def reset_hidden_states(self):
        self.hidden=torch.zeros(self.num_agents,self.config.hidden,device=self.device)
        self.previous=torch.zeros(self.num_agents,self.action_dim,device=self.device)
        self.last_entropy=np.zeros(self.num_agents,dtype=np.float32)

    def act(self, observations, action_masks=None):
        obs=torch.as_tensor(np.asarray(observations),dtype=torch.float32,device=self.device)
        if obs.shape!=(self.num_agents,self.obs_dim) or not bool(torch.isfinite(obs).all()):
            raise ValueError('Incompatible or non-finite observation batch')
        masks=np.ones((self.num_agents,self.action_dim),dtype=bool) if action_masks is None else np.asarray(action_masks,dtype=bool)
        if masks.shape!=(self.num_agents,self.action_dim) or not masks.any(axis=1).all():
            raise ValueError('Invalid action mask batch')
        with torch.no_grad():
            q,self.hidden=self.actor(obs,self.hidden,self.previous)
            if not bool(torch.isfinite(q).all()):raise FloatingPointError('Non-finite policy values')
            q=q.masked_fill(~torch.as_tensor(masks,device=self.device),-torch.inf)
            probs=F.softmax(q,dim=-1)
            self.last_entropy=(-(probs*probs.clamp_min(1e-30).log()).sum(-1)).cpu().numpy()
            actions=q.argmax(-1).cpu().tolist()
        for i in range(self.num_agents):
            if self.epsilon>0 and self.rng.random()<self.epsilon:
                actions[i]=int(self.rng.choice(np.flatnonzero(masks[i])))
        self.previous=F.one_hot(torch.as_tensor(actions,device=self.device),self.action_dim).float()
        return actions

    def _roll(self, network, obs, actions, valid, gradient):
        batch,length,n,_=obs.shape
        burn=self.config.burn_in
        previous=torch.zeros(batch,length,n,self.action_dim,device=self.device)
        previous[:,1:]=F.one_hot(actions,self.action_dim).float()*valid[:,:-1,None,None]
        inputs=torch.cat((obs,previous),dim=-1).permute(0,2,1,3).reshape(batch*n,length,-1)
        hidden=torch.zeros(1,batch*n,self.config.hidden,device=self.device)
        if burn:
            # Replay has contiguous valid frames, with left padding before burn-in
            # and right padding after the real final state. Compact only the burn
            # prefix so fictitious left padding never updates recurrent memory.
            with torch.no_grad():
                counts=valid[:,:burn].sum(1).repeat_interleave(n)
                offsets=burn-counts
                positions=(offsets[:,None]+torch.arange(burn,device=self.device)[None,:]).clamp_max(burn-1)
                prefix=inputs[:,:burn].gather(1,positions[...,None].expand(-1,-1,inputs.shape[-1]))
                encoded=network.encoder(prefix)
                packed=pack_padded_sequence(encoded,counts.clamp_min(1).cpu(),batch_first=True,enforce_sorted=False)
                _,hidden=network.gru(packed,hidden)
                hidden=hidden*(counts>0)[None,:,None]
        with torch.set_grad_enabled(gradient):
            encoded=network.encoder(inputs[:,burn:])
            outputs,_=network.gru(encoded,hidden.detach())
            values=network.head(outputs).reshape(batch,n,length-burn,self.action_dim).permute(0,2,1,3)
        # Burn-in outputs are not used in targets or losses. Tail padding follows
        # every real frame and cannot affect earlier outputs in this causal GRU.
        return torch.cat((torch.zeros(batch,burn,n,self.action_dim,device=self.device),values),dim=1)

    def train_step(self, batch=None):
        if batch is None:
            batch=self.replay.sample(self.config.batch_size,self.config.burn_in,self.config.unroll,self.rng)
        if batch is None:
            return None
        tensors={key:torch.as_tensor(value,device=self.device) for key,value in batch.items()}
        obs=tensors['observations'].float()
        actions=tensors['actions'].long()
        valid=tensors['valid'].bool()
        q=self._roll(self.actor,obs,actions,valid,True)
        with torch.no_grad():
            tq=self._roll(self.target_actor,obs,actions,valid,False)
        begin=self.config.burn_in
        chosen=q[:,begin:-1].gather(-1,actions[:,begin:].unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            bootstrap=masked_double_q(q[:,begin+1:].detach(),tq[:,begin+1:],tensors['masks'][:,begin+1:])
        eligible=learning_eligibility(tensors['alive'].bool(),tensors['masks'].bool())
        alive=eligible[:,begin:-1].float()
        next_alive=eligible[:,begin+1:].float()
        loss_mask=tensors['loss_mask'][:,begin:].float()
        rewards=tensors['rewards'][:,begin:].float()
        terminated=tensors['terminated'][:,begin:].bool()
        if self.algorithm in ('iql','dqn'):
            targets=rewards.unsqueeze(-1)+self.config.gamma*torch.where(terminated.unsqueeze(-1),0.0,bootstrap*next_alive)
            weight=loss_mask.unsqueeze(-1)*alive
        else:
            if self.mixer is not None:
                chosen=self.mixer(chosen*alive,tensors['states'][:,begin:-1].float())
                with torch.no_grad():
                    bootstrap=self.target_mixer(bootstrap*next_alive,tensors['states'][:,begin+1:].float())
            else:
                chosen=(chosen*alive).sum(-1)
                bootstrap=(bootstrap*next_alive).sum(-1)
            targets=rewards+self.config.gamma*torch.where(terminated,0.0,bootstrap)
            weight=loss_mask
        loss=(F.smooth_l1_loss(chosen,targets,reduction='none')*weight).sum()/weight.sum().clamp_min(1)
        if self.config.bc_aux_weight and self.demonstrations.episodes:
            demo=self.demonstrations.sample(self.config.batch_size,self.config.burn_in,self.config.unroll,self.rng)
            demonstration={key:torch.as_tensor(value,device=self.device) for key,value in demo.items()}
            utilities=self._roll(self.actor,demonstration['observations'].float(),demonstration['actions'].long(),demonstration['valid'].bool(),True)[:,begin:-1]
            labels=demonstration.get('expert_actions',demonstration['actions'])[:,begin:].long()
            eligible=demonstration['loss_mask'][:,begin:,None].float()*learning_eligibility(demonstration['alive'].bool(),demonstration['masks'].bool())[:,begin:-1].float()
            imitation=(F.cross_entropy(utilities.reshape(-1,self.action_dim),labels.reshape(-1),reduction='none').reshape(labels.shape)*eligible).sum()/eligible.sum().clamp_min(1)
            loss=0.2*loss+(self.config.bc_aux_weight+1.0)*imitation
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Non-finite TD loss')
        self.optimizer.zero_grad()
        loss.backward()
        actor_norm=nn.utils.clip_grad_norm_(self.actor.parameters(),self.config.gradient_clip,error_if_nonfinite=True)
        mixer_norm=nn.utils.clip_grad_norm_(self.mixer.parameters(),self.config.gradient_clip,error_if_nonfinite=True) if self.mixer is not None else torch.tensor(0.0)
        norm=float(torch.hypot(torch.as_tensor(actor_norm),torch.as_tensor(mixer_norm)).item())
        self.optimizer.step()
        self.training_kind=('bc_regularized_rl' if self.config.bc_aux_weight else 'bc_then_rl') if self.training_kind.startswith('bc') else 'rl'
        self.updates+=1
        if self.updates%self.config.target_every==0:
            self.update_targets()
        return dict(loss=float(loss.item()),gradient_norm=float(norm),updates=self.updates)

    def update_targets(self):
        self.target_actor.load_state_dict(self.actor.state_dict())
        if self.mixer:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def behavior_clone(self, episodes, epochs=3):
        if not episodes:
            raise ValueError('Expert demonstrations are empty')
        self.training_kind='bc_only'
        losses=[]
        counts=np.bincount(np.concatenate([e.get('expert_actions',e['actions'])[learning_eligibility(e['alive'].astype(bool),e['masks'].astype(bool))[:-1]] for e in episodes]),minlength=self.action_dim)
        weights=1/np.maximum(counts,1)
        weights=torch.as_tensor(weights/weights.mean(),dtype=torch.float32,device=self.device)
        for _ in range(epochs):
            for episode in episodes:
                obs=torch.as_tensor(episode['observations'][:-1],dtype=torch.float32,device=self.device)
                labels=torch.as_tensor(episode.get('expert_actions',episode['actions']),dtype=torch.long,device=self.device)
                executed=torch.as_tensor(episode['actions'],dtype=torch.long,device=self.device)
                alive=torch.as_tensor(learning_eligibility(episode['alive'][:-1].astype(bool),episode['masks'][:-1].astype(bool)),dtype=torch.float32,device=self.device)
                hidden=torch.zeros(self.num_agents,self.config.hidden,device=self.device)
                previous=torch.zeros(len(obs),self.num_agents,self.action_dim,device=self.device)
                previous[1:]=F.one_hot(executed[:-1],self.action_dim).float()
                for first in range(0,len(obs),self.config.unroll):
                    end=min(len(obs),first+self.config.unroll)
                    inputs=torch.cat((obs[first:end],previous[first:end]),dim=-1).transpose(0,1)
                    encoded=self.actor.encoder(inputs)
                    output,next_hidden=self.actor.gru(encoded,hidden.unsqueeze(0))
                    q=self.actor.head(output).transpose(0,1)
                    loss=weighted_imitation_loss(q,labels[first:end],alive[first:end],weights)
                    self.optimizer.zero_grad(set_to_none=True);loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(),self.config.gradient_clip,error_if_nonfinite=True)
                    self.optimizer.step()
                    losses.append(float(loss.item()))
                    hidden=next_hidden.squeeze(0).detach()
        self.update_targets()
        if self.mixer is not None:
            td_episodes=[ep for ep in episodes if 'rewards' in ep and 'states' in ep and 'terminated' in ep]
            for _ in range(max(1,min(epochs,2))):
                for episode in td_episodes:
                    obs_all=torch.as_tensor(episode['observations'],dtype=torch.float32,device=self.device)
                    executed=torch.as_tensor(episode['actions'],dtype=torch.long,device=self.device)
                    masks_all=torch.as_tensor(episode['masks'],dtype=torch.bool,device=self.device)
                    alive_all=torch.as_tensor(learning_eligibility(episode['alive'].astype(bool),episode['masks'].astype(bool)),dtype=torch.float32,device=self.device)
                    states_all=torch.as_tensor(episode['states'],dtype=torch.float32,device=self.device)
                    rewards_all=torch.as_tensor(episode['rewards'],dtype=torch.float32,device=self.device)
                    terminated_all=torch.as_tensor(episode['terminated'],dtype=torch.bool,device=self.device)
                    previous_all=torch.zeros(len(obs_all),self.num_agents,self.action_dim,device=self.device)
                    previous_all[1:]=F.one_hot(executed,self.action_dim).float()
                    with torch.no_grad():
                        full_in=torch.cat((obs_all,previous_all),dim=-1).transpose(0,1)
                        q_out,_=self.actor.gru(self.actor.encoder(full_in),torch.zeros(1,self.num_agents,self.config.hidden,device=self.device))
                        q_all=self.actor.head(q_out).transpose(0,1)
                        tq_out,_=self.target_actor.gru(self.target_actor.encoder(full_in),torch.zeros(1,self.num_agents,self.config.hidden,device=self.device))
                        tq_all=self.target_actor.head(tq_out).transpose(0,1)
                        chosen_all=q_all[:-1].gather(-1,executed.unsqueeze(-1)).squeeze(-1)*alive_all[:-1]
                        bootstrap_all=masked_double_q(q_all[1:],tq_all[1:],masks_all[1:])*alive_all[1:]
                    for first in range(0,len(executed),self.config.unroll):
                        end=min(len(executed),first+self.config.unroll)
                        chosen_tot=self.mixer(chosen_all[first:end],states_all[first:end])
                        with torch.no_grad():
                            bootstrap_tot=self.target_mixer(bootstrap_all[first:end],states_all[first+1:end+1])
                            targets=rewards_all[first:end]+self.config.gamma*torch.where(terminated_all[first:end],0.0,bootstrap_tot)
                        td_loss=F.smooth_l1_loss(chosen_tot,targets)
                        self.optimizer.zero_grad(set_to_none=True);td_loss.backward()
                        nn.utils.clip_grad_norm_(self.mixer.parameters(),self.config.gradient_clip,error_if_nonfinite=True)
                        self.optimizer.step()
                self.target_mixer.load_state_dict(self.mixer.state_dict())
        return float(np.mean(losses))

    def checkpoint(self, metadata=None, include_replay=False):
        payload=dict(environment_version=ENV_VERSION,observation_version=OBS_VERSION,algorithm=self.algorithm,
            recurrent_backend='batched_gru_v2',
            num_agents=self.num_agents,obs_dim=self.obs_dim,state_dim=self.state_dim,action_dim=self.action_dim,
            config=asdict(self.config),actor=self.actor.state_dict(),target_actor=self.target_actor.state_dict(),
            mixer=self.mixer.state_dict() if self.mixer else None,target_mixer=self.target_mixer.state_dict() if self.target_mixer else None,
            optimizer=self.optimizer.state_dict(),updates=self.updates,environment_steps=self.environment_steps,
            training_kind=self.training_kind,metadata=metadata or {},rng_state=self.rng.bit_generator.state,
            torch_rng=torch.get_rng_state(),python_rng=random.getstate())
        if self.device.type=='cuda':payload['cuda_rng']=[state.cpu() for state in torch.cuda.get_rng_state_all()]
        if include_replay:
            payload['replay']=[{k:torch.from_numpy(v) for k,v in episode.items()} for episode in self.replay.episodes]
            payload['prior_replay']=[{k:torch.from_numpy(v) for k,v in episode.items()} for episode in self.replay.prior_episodes]
            payload['replay_stage']=self.replay.stage
            payload['prior_fraction']=self.replay.prior_fraction
            payload['demonstrations']=[{k:torch.from_numpy(v) for k,v in episode.items()} for episode in self.demonstrations.episodes]
        return payload

    @classmethod
    def from_checkpoint(cls, payload, device='cpu', resume=False):
        if payload.get('environment_version')!=ENV_VERSION or payload.get('observation_version')!=OBS_VERSION:
            raise ValueError('Checkpoint environment/observation version is incompatible')
        model=cls(payload['algorithm'],payload['num_agents'],payload['obs_dim'],payload['state_dim'],payload['action_dim'],payload['config'],device=device)
        def recurrent_state(state):
            # GRUCell and a one-layer GRU use the same gate equations and tensor
            # layout; only parameter names differ. Preserve historical policies.
            old={'gru.weight_ih','gru.weight_hh','gru.bias_ih','gru.bias_hh'}
            return {key+'_l0' if key in old else key:value for key,value in state.items()}
        model.actor.load_state_dict(recurrent_state(payload['actor']))
        model.target_actor.load_state_dict(recurrent_state(payload['target_actor']))
        if model.mixer:
            model.mixer.load_state_dict(payload['mixer']);model.target_mixer.load_state_dict(payload['target_mixer'])
        model.updates=payload['updates'];model.environment_steps=payload['environment_steps'];model.training_kind=payload['training_kind']
        model.metadata=payload.get('metadata',{})
        if resume:
            model.optimizer.load_state_dict(payload['optimizer'])
            model.rng.bit_generator.state=payload['rng_state']
            torch.set_rng_state(payload['torch_rng'].cpu());random.setstate(payload['python_rng'])
            if model.device.type=='cuda' and payload.get('cuda_rng') is not None:
                torch.cuda.set_rng_state_all([state.cpu() for state in payload['cuda_rng']])
            for episode in payload.get('replay',[]):
                model.replay.add({k:v.cpu().numpy() for k,v in episode.items()})
            model.replay.prior_episodes=[{k:v.cpu().numpy() for k,v in episode.items()} for episode in payload.get('prior_replay',[])]
            model.replay.stage=payload.get('replay_stage')
            model.replay.prior_fraction=payload.get('prior_fraction',0.0)
            for episode in payload.get('demonstrations',[]):model.demonstrations.add({k:v.cpu().numpy() for k,v in episode.items()})
        return model
