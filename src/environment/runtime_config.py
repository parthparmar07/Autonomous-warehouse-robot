from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class RuntimeConfig:
    horizon: int = 2000
    capacity: float = 100.0
    initial_soc: float = 1.0
    move_cost: float = 0.2
    loaded_move_cost: float = 0.3
    turn_cost: float = 0.05
    wait_cost: float = 0.005
    service_cost: float = 0.1
    charge_rate: float = 2.0
    taper_rate: float = 0.5
    taper_soc: float = 0.8
    charge_trigger: float = 0.3
    charge_release: float = 0.9
    reserve: float = 5.0
    service_ticks: int = 3
    lease_ticks: int = 100
    heartbeat_timeout: int = 15
    queue_capacity: int = 4
    max_orders: int = 128
    deadlock_ticks: int = 50
    sensor_radius: int = 2
    front_fov: float = 360.0
    rear_fov: float = 0.0
    sensor_dropout: float = 0.0
    sensor_latency: int = 0
    shuffle_roles: bool = False
    finite_batch: bool = True
    delivery_reward: float = 10.0
    backlog_cost: float = .001
    energy_reward_cost: float = .01
    conflict_cost: float = .05
    lateness_cost: float = .002
    depletion_cost: float = 10.0
    idle_bay_cost: float = .02
    potential_weight: float = 2.0
    potential_gamma: float = .99
    potential_distance_scale: float = 20.0

    def __post_init__(self):
        if any(not math.isfinite(v) for v in asdict(self).values() if isinstance(v, (int, float))):
            raise ValueError("Configuration must be finite")
        for name in ("capacity", "move_cost", "loaded_move_cost", "turn_cost", "wait_cost", "service_cost", "charge_rate", "taper_rate", "reserve"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("horizon", "service_ticks", "lease_ticks", "heartbeat_timeout", "queue_capacity", "max_orders", "deadlock_ticks"):
            if isinstance(getattr(self,name),bool) or not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= self.initial_soc <= 1 or not 0 < self.charge_trigger < self.charge_release <= 1:
            raise ValueError("Invalid SOC / charge hysteresis")
        if self.loaded_move_cost < self.move_cost or self.taper_rate > self.charge_rate or self.taper_rate <= self.wait_cost or not 0 < self.taper_soc < 1:
            raise ValueError("Invalid energy/charging profile")
        if not isinstance(self.sensor_radius, int) or not 1 <= self.sensor_radius <= 8 or not 0 <= self.sensor_dropout <= 1:
            raise ValueError("Invalid sensor radius/dropout")
        if not isinstance(self.sensor_latency, int) or not 0 <= self.sensor_latency <= 50:
            raise ValueError("Invalid sensor latency")
        if not all(0 <= f <= 360 for f in (self.front_fov, self.rear_fov)):
            raise ValueError("Invalid sensor field of view")
        for name in ('delivery_reward','backlog_cost','energy_reward_cost','conflict_cost',
                     'lateness_cost','depletion_cost','idle_bay_cost','potential_weight'):
            if getattr(self,name)<0:raise ValueError(f'{name} must be nonnegative')
        if not 0<=self.potential_gamma<=1 or self.potential_distance_scale<=0:
            raise ValueError('Invalid potential reward configuration')

    def to_dict(self):
        return asdict(self)
