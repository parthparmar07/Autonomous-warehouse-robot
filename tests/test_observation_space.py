import numpy as np
from src.environment.warehouse_env import WarehouseEnv
from src.environment.spec import OBS_DIM, SELF_FIELDS, SENSOR_CHANNELS

def test_local_observation_versioned_shape_and_finite_proprioception():
    env=WarehouseEnv(grid_size=(8,8),num_agents=2,seed=21)
    obs=env.get_local_observation(0)
    assert obs.shape==(OBS_DIM,) and np.isfinite(obs).all()
    offset=25*len(SENSOR_CHANNELS)
    s=offset+SELF_FIELDS.index('heading_sin');c=offset+SELF_FIELDS.index('heading_cos')
    assert np.isclose(obs[s]**2+obs[c]**2,1)

def test_local_observation_heading_and_ego_goal_rotate_together():
    env=WarehouseEnv(grid_size=(8,8),num_agents=1,seed=3)
    env.robots[0]['target']=(4,1)
    offset=25*len(SENSOR_CHANNELS)
    env.robots[0]['orientation']=0
    north=env.get_local_observation(0)
    env.robots[0]['orientation']=2
    south=env.get_local_observation(0)
    np.testing.assert_allclose(north[offset:offset+2],-south[offset:offset+2])
    assert not np.allclose(north[offset+4:offset+6],south[offset+4:offset+6])
