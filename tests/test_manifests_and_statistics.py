import json
import pytest
from src.evaluation.manifests import generate_evaluation_manifest,environment_from_manifest,write_manifest
from src.evaluation.statistics import paired_hierarchical_interval,family_gate
from src.environment.spec import OBS_VERSION
from src.baselines.nearest_task import NearestTaskDispatcher

def test_materialized_energy_fault_and_sensor_scenarios_replay_exactly(tmp_path):
    manifest=json.loads(json.dumps(generate_evaluation_manifest(5,300005)))
    a=environment_from_manifest(manifest);b=environment_from_manifest(manifest)
    pa,pb=NearestTaskDispatcher(a),NearestTaskDispatcher(b)
    for _ in range(80):
        assert a.step(pa.act(a))==b.step(pb.act(b))
        assert a.event_log==b.event_log
    path=tmp_path/'scenario.json';write_manifest(path,manifest)
    other=generate_evaluation_manifest(5,300006)
    with pytest.raises(FileExistsError):write_manifest(path,other)

def test_uncertainty_rejects_unpaired_or_duplicate_scenarios():
    row=dict(training_seed=11,scenario_hash='x',throughput=0)
    with pytest.raises(ValueError):paired_hierarchical_interval([row,row],[row])
    with pytest.raises(ValueError):paired_hierarchical_interval([row],[dict(row,scenario_hash='y')])
    assert paired_hierarchical_interval([row],[row])['ci95'] is None
    assert not family_gate([])['passed']
    wait=[dict(family='positive_demand',completion_rate=0,executed_collisions=0,avoidable_depletions=0)]*20
    assert not family_gate(wait)['passed']
