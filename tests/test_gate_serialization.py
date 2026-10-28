"""Public reporting/restore boundaries must work with real statistics and replay."""
import json
from types import SimpleNamespace
import pytest
from src.evaluation.statistics import family_gate
from src.training import runner


@pytest.mark.parametrize('completion',[0.,1.])
def test_twenty_case_gate_is_json_serializable(completion):
    rows=[dict(family='C0',completion_rate=completion,executed_collisions=0,avoidable_depletions=0) for _ in range(20)]
    result=family_gate(rows)
    restored=json.loads(json.dumps(result))
    assert restored['passed']==bool(completion)
    assert type(result['families'][0]['passed']) is bool
    assert type(result['families'][0]['wilson_lower95']) is float


def test_failed_curriculum_gate_writes_its_result_instead_of_crashing(tmp_path,monkeypatch):
    import scripts.train_curriculum as curriculum
    args=SimpleNamespace(algorithm='qmix',num_agents=5,seed=11,horizon=100,bc_episodes=0,
        bc_epochs=1,bc_aux_weight=0,dagger_rounds=0,eval_episodes=20,stage_steps=100,
        device='cuda',train_every=16,eval_every=50,epsilon_steps=None,skill_episodes=20)
    monkeypatch.setattr(curriculum,'ROOT',tmp_path)
    def train(args,evaluation_callback):
        model=SimpleNamespace(environment_steps=100)
        evaluation_callback(model,args.stage,50)
        evaluation_callback(model,args.stage,100)
        return model,dict(run_id='test'),[]
    monkeypatch.setattr(curriculum,'train',train)
    rows=[dict(family='C0',completion_rate=0.,executed_collisions=0,avoidable_depletions=0) for _ in range(20)]
    monkeypatch.setattr(curriculum,'evaluate_curriculum_gate',lambda *a,**k:family_gate(rows))
    assert curriculum.run(args) is False
    result=json.loads((tmp_path/'outputs/curriculum/qmix_N5_seed11.json').read_text(encoding='utf-8'))
    assert result['status']=='not_ready' and result['stages'][0]['gate']['passed'] is False
    assert [c['environment_steps'] for c in result['checks']]==[50,100]


def test_curriculum_warm_starts_bridge_and_stops_before_five_robot_stage(tmp_path,monkeypatch):
    import scripts.train_curriculum as curriculum
    args=SimpleNamespace(algorithm='qmix',num_agents=5,seed=11,horizon=100,bc_episodes=2,
        bc_epochs=1,bc_aux_weight=0,dagger_rounds=0,eval_episodes=20,stage_steps=100,
        device='cuda',train_every=16,eval_every=50,epsilon_steps=None,skill_episodes=20)
    monkeypatch.setattr(curriculum,'ROOT',tmp_path)
    calls=[];retention=[]
    def train(base,evaluation_callback):
        calls.append(dict(stage=base.stage,resume=base.resume,bc=base.bc_episodes,
            device=base.device,epsilon_steps=base.epsilon_steps))
        model=SimpleNamespace(environment_steps=base.steps)
        evaluation_callback(model,base.stage,base.steps)
        return model,dict(run_id=base.stage),[]
    def gate(model,stages,*args):
        retention.append(tuple(stages))
        return dict(passed=stages[-1]=='C0')
    monkeypatch.setattr(curriculum,'train',train)
    monkeypatch.setattr(curriculum,'evaluate_curriculum_gate',gate)
    assert curriculum.run(args) is False
    assert retention==[('C0',),('C0','C0.5')]
    assert calls[0]['resume'] is None and calls[0]['bc']==2
    assert calls[1]['resume']==tmp_path/'outputs/runs/C0/checkpoint.pt' and calls[1]['bc']==2
    assert all(c['device']=='cuda' and c['epsilon_steps']==50 for c in calls)


def test_resume_deserializes_replay_on_cpu_before_restoring_device(monkeypatch,tmp_path):
    args=runner.parser().parse_args(['--stage','C0','--tiny','--num-agents','1','--device','cuda','--resume',str(tmp_path/'checkpoint.pt')])
    class RestoreBoundary(Exception):pass
    payload={'sentinel':True}
    def load(path,**kwargs):
        assert kwargs['map_location']=='cpu'
        assert kwargs['weights_only'] is True
        return payload
    def restore(actual,*args):
        assert actual is payload
        raise RestoreBoundary()
    monkeypatch.setattr(runner,'RecurrentLearner',lambda *a,**k:None)
    monkeypatch.setattr(runner.torch,'load',load)
    monkeypatch.setattr(runner,'resume_learner',restore)
    with pytest.raises(RestoreBoundary):runner.train(args)
