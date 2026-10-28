from pathlib import Path
import json
import yaml
from scripts.train_research_protocol import atomic_json, resume_progress, stage_profile, stage_demonstration_count


def test_full_curriculum_profiles_and_budget():
    path=Path(__file__).parents[1]/'configs/council_full_curriculum.yaml'
    config=yaml.safe_load(path.read_text(encoding='utf-8'))
    assert list(config['stages'])==['C0','C0.5','C0.75','C1','C2','C3','C4','C5']
    assert sum(config['stages'].values())==2730000
    assert stage_profile(config,'C2')=='core'
    assert stage_profile(config,'C3')=='robust'
    assert config['balanced_demonstrations']
    assert config['promotion']['consecutive_checks']==2
    assert config['conditions']['assisted_unshaped']['overrides']['potential_weight']==0
    assert config['conditions']['pure_rl_potential']['overrides']['bc_episodes']==0
    assert config['conditions']['pure_rl_potential']['overrides']['epsilon_start']==1


def test_protocol_resume_keeps_only_promoted_stages():
    stages=[dict(stage='C0',steps=12000,promoted=True),dict(stage='C0.5',steps=262000,promoted=False)]
    completed,total,checkpoint=resume_progress(dict(stages=stages),Path('latest.pt'))
    assert completed==stages[:1]
    assert total==12000
    assert checkpoint==Path('latest.pt')


def test_atomic_protocol_report_replaces_and_leaves_no_temporary(tmp_path):
    path=tmp_path/'result.json'
    atomic_json(path,dict(status='initializing'))
    atomic_json(path,dict(status='training',steps=5000))
    assert json.loads(path.read_text(encoding='utf-8'))==dict(status='training',steps=5000)
    assert not path.with_suffix('.json.tmp').exists()


def test_same_stage_resume_skips_expert_collection_but_new_stage_restores_it():
    config=dict(bc_episodes=80)
    assert stage_demonstration_count(config,'C0.5','C0.5')==0
    assert stage_demonstration_count(config,'C0.75','C0.5')==80
    assert stage_demonstration_count(config,'C0.5',None)==80
