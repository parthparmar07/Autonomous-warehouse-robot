from concurrent.futures import ThreadPoolExecutor
import pytest
from fastapi.testclient import TestClient
from src.deployment.api import app,_sessions
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
        _sessions['expired'].last_access=0
        assert client.get('/state?session_id=expired').status_code==404
