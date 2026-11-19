from concurrent.futures import ThreadPoolExecutor
import time
import pytest
from fastapi.testclient import TestClient
from src.deployment.api import app,_sessions,SESSION_TTL
from src.environment.spec import OBS_DIM

def test_sensor_configuration_shapes_and_strict_actions():
    with TestClient(app) as client:
        sid='sensor-contract'
        try:
            response=client.post(f'/reset?session_id={sid}',json=dict(num_agents=2,sensor_front_fov=90,sensor_rear_fov=90,initial_soc=.7))
            assert response.status_code==200
            assert response.json()['robots'][0]['energy_eu']==70
            view=client.get(f'/sensor?session_id={sid}&agent_id=0').json()
            assert len(view['frame'])==5 and len(view['frame'][0][0])==12
            assert len(view['observation'])==OBS_DIM and len(view['action_mask'])==7
            for actions in ([True,4],[1.1,4],[7,4],[4]):
                assert client.post(f'/step?session_id={sid}',json=dict(actions=actions)).status_code==422
            assert client.get(f'/sensor?session_id={sid}&agent_id=2').status_code==422
        finally:client.delete(f'/session?session_id={sid}')

def test_concurrent_steps_are_serialized_and_failed_reset_preserves_session():
    with TestClient(app) as client:
        sid='concurrent-contract'
        try:
            assert client.post(f'/reset?session_id={sid}',json=dict(num_agents=1)).status_code==200
            with ThreadPoolExecutor(max_workers=4) as pool:
                results=list(pool.map(lambda _:client.post(f'/step?session_id={sid}',json=dict(actions=[4])),range(12)))
            assert all(r.status_code==200 for r in results)
            assert client.get(f'/state?session_id={sid}').json()['metrics']['step_count']==12
            assert client.post(f'/reset?session_id={sid}',json=dict(algorithm='qmix',num_agents=39)).status_code==409
            assert client.get(f'/state?session_id={sid}').json()['metrics']['step_count']==12
        finally:client.delete(f'/session?session_id={sid}')

def test_session_ttl_and_identifier_validation():
    with TestClient(app) as client:
        assert client.post('/reset?session_id=bad/id',json={}).status_code==422
        client.post('/reset?session_id=expired',json={})
        # monotonic() counts from boot; 0 is not "old" on a runner up < SESSION_TTL.
        _sessions['expired'].last_access=time.monotonic()-SESSION_TTL-1
        assert client.get('/state?session_id=expired').status_code==404

def test_algorithm_availability_follows_workload():
    with TestClient(app) as client:
        trained = lambda wl: {a['name']: a['trained'] for a in client.get(f'/algorithms?num_agents=10&workload={wl}').json()}
        assert trained('mixed')['qmix'] and trained('mixed')['iql']
        assert not trained('outbound')['qmix']
        assert client.get('/algorithms?num_agents=10&workload=bogus').status_code == 422
