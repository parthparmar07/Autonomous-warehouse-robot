from scripts import launch_training_study as study


def test_status_requires_fresh_heartbeat_and_same_live_process_identity(monkeypatch):
    matrix=dict(status='running',heartbeat_unix=100,supervisor_pid=42,supervisor_birth='original')
    monkeypatch.setattr(study,'process_identity',lambda pid:'original')
    assert study.effective_status(matrix,110)=='running'
    assert study.effective_status(matrix,200)=='stale_heartbeat'
    monkeypatch.setattr(study,'process_identity',lambda pid:'reused')
    assert study.effective_status(matrix,110)=='supervisor_not_alive'
    monkeypatch.setattr(study,'process_identity',lambda pid:None)
    assert study.effective_status(matrix,110)=='supervisor_not_alive'
    matrix['supervisor_birth']=None
    assert study.effective_status(matrix,110)=='supervisor_not_alive'
    matrix['status']='interrupted'
    assert study.effective_status(matrix,110)=='interrupted'


def test_status_report_distinguishes_execution_error_from_finished_study(tmp_path):
    matrix=dict(status='finished',heartbeat_at='test',jobs=[dict(algorithm='qmix',seed=11,status='execution_error')])
    study.status_report(tmp_path,matrix)
    report=(tmp_path/'status.md').read_text(encoding='utf-8')
    assert 'execution_error' in report and 'completion is not a competence claim' in report
    assert not (tmp_path/'status.md.tmp').exists()


def test_resume_rejects_live_supervisor_even_with_stale_heartbeat(tmp_path,monkeypatch):
    import json
    import yaml
    import pytest
    config=dict(study='test',seeds=[11],algorithms=['qmix'])
    path=tmp_path/'protocol.yaml';path.write_text(yaml.safe_dump(config),encoding='utf-8')
    matrix=dict(status='running',heartbeat_unix=0,supervisor_pid=42,supervisor_birth='original')
    (tmp_path/'matrix.json').write_text(json.dumps(matrix),encoding='utf-8')
    monkeypatch.setattr(study,'process_identity',lambda pid:'original')
    with pytest.raises(ValueError,match='already alive'):
        study.run(path,tmp_path,resume=True)


def test_controls_are_explicit_and_do_not_mutate_main_configuration():
    config=dict(study='council',seeds=[11,23],algorithms=['qmix','iql'],bc_episodes=80,
        potential_weight=2,conditions=dict(main=dict(seeds=[11,23]),
        pure=dict(seeds=[11],overrides=dict(bc_episodes=0))))
    jobs=study.study_jobs(config)
    assert len(jobs)==6
    child=study.job_config(config,jobs[-1])
    assert child['bc_episodes']==0 and child['study']=='council_pure'
    assert config['bc_episodes']==80
