"""Versioned contracts. Changing these invalidates saved policies/manifests."""
ENV_VERSION = "warehouse_v6_research"
OBS_VERSION = "egocentric_range_v3_pose_feedback"
ACTION_NAMES = ("FORWARD", "BACKWARD", "LEFT", "RIGHT", "WAIT", "SERVICE", "REQUEST_CHARGE")
ACTION_DIM = len(ACTION_NAMES)
MODES = ("IDLE", "TO_PICKUP", "PICKING", "TO_DROP", "DROPPING", "TO_CHARGER", "QUEUED_CHARGE", "CHARGING", "TO_PARK", "FAULTED", "DEPLETED", "BLOCKED_RECOVERY")
SENSOR_CHANNELS = ("visible", "outside", "shelf", "corridor", "pickup", "drop", "charger", "robot", "neighbor_heading_sin", "neighbor_heading_cos", "neighbor_load", "neighbor_immobile")
SELF_FIELDS = ("goal_right", "goal_forward", "goal_distance", "load", "heading_sin", "heading_cos", "soc", "capacity", "energy_cost", "service_remaining", "task_age", "deadline_slack", "task_present", "energy_margin", "sensor_age", "sensor_valid", "own_x", "own_y")
SENSOR_CONFIG_FIELDS = ("front_fov", "rear_fov", "dropout", "configured_latency")
ACTION_HISTORY_LENGTH = 2
DEFAULT_RADIUS = 2
OBS_DIM = ((2 * DEFAULT_RADIUS + 1) ** 2 * len(SENSOR_CHANNELS) + len(SELF_FIELDS)
           + len(MODES) + len(SENSOR_CONFIG_FIELDS) + ACTION_HISTORY_LENGTH * ACTION_DIM)
