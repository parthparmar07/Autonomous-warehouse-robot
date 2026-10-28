"""Receiving, finite shelf storage and stock-backed shipping on the fixed map."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import numpy as np
from src.environment.core import WarehouseEnv
from src.environment.layout import default_layout
from src.environment.runtime_config import RuntimeConfig
from src.environment.task_generator import schedule_order
from src.utils.hashing import sha256_digest


class MixedWarehouseEnv(WarehouseEnv):
    workload_version = 'mixed_stock_v1'

    def __init__(self, num_agents=5, seed=0, config=None, shelf_capacity=4):
        if isinstance(shelf_capacity, bool) or not isinstance(shelf_capacity, int) or shelf_capacity < 1:
            raise ValueError('Shelf capacity must be a positive integer')
        self.shelf_capacity = shelf_capacity
        self.inventory = {}
        self.receiving_bays = [(2, 2), (10, 2)]
        layout = default_layout()
        layout['staging_cells'] = [c for c in layout['staging_cells'] if c not in self.receiving_bays]
        super().__init__(num_agents=num_agents, seed=seed, config=config, **layout)
        for x, y in self.receiving_bays:
            self.grid[y, x] = self._base_grid[y, x] = 3  # Locally sensed package service station.

    def reset(self, seed=None):
        self.inventory = {}
        return super().reset(seed)

    def _normalize_tasks(self):
        identities={}
        locations={}
        for task in self.task_queue+self.finished_tasks:
            if task.get('flow') not in ('inbound','outbound'):
                raise ValueError('Mixed logistics requires a valid package flow')
            identity=(task['flow'],task['package_id'])
            if identity in identities and identities[identity]!=task['task_id']:
                raise ValueError('A physical package can have only one job per flow')
            identities[identity]=task['task_id']
            bay=tuple(task['storage_bay'])
            if task['package_id'] in locations and locations[task['package_id']]!=bay:
                raise ValueError('Incoming and outgoing jobs disagree on storage location')
            locations[task['package_id']]=bay
        super()._normalize_tasks()
        for task in self.task_queue:
            if 'flow' not in task:
                raise ValueError('Mixed logistics requires stock-backed orders')
            stock = task['package_id']
            bay = tuple(task['storage_bay'])
            if bay not in self.pickup_bays or task['flow'] not in ('inbound', 'outbound'):
                raise ValueError('Invalid stock order')
            if task['flow'] == 'inbound' and stock not in self.inventory:
                if tuple(task['pickup']) not in self.receiving_bays or tuple(task['delivery']) != bay:
                    raise ValueError('Inbound must connect receiving and storage')
                self.inventory[stock] = dict(storage_bay=bay, state='receiving', reserved=False)
                task['package_location'] = dict(kind='receiving', cell=list(task['pickup']))
            if task['flow'] == 'outbound':
                if tuple(task['pickup']) != bay or tuple(task['delivery']) not in self.drop_bays:
                    raise ValueError('Outbound must connect storage and shipping')
        # A slot remains reserved during the incoming journey. It is freed only
        # when the outgoing robot physically picks the package off the shelf.
        occupancy = Counter(tuple(s['storage_bay']) for s in self.inventory.values() if s['reserved'])
        for task in self.task_queue:
            if task.get('owner') is not None or task['status'] in ('picked', 'blocked_recovery'):
                continue
            stock = self.inventory.get(task['package_id'])
            bay = tuple(task['storage_bay'])
            if task['flow'] == 'inbound':
                if stock['state'] != 'receiving':
                    continue
                if not stock['reserved'] and occupancy[bay] < self.shelf_capacity:
                    stock['reserved'] = True
                    occupancy[bay] += 1
                task['status'] = 'pending' if stock['reserved'] else 'waiting_capacity'
            else:
                ready = stock is not None and stock['state'] == 'shelf'
                task['status'] = 'pending' if ready else 'waiting_stock'
                task['package_location'] = dict(kind='shelf' if ready else 'awaiting_stock', cell=list(bay))

    def _service(self, robot):
        task = self._task(robot['current_task'])
        kind = self._service_kind(robot)
        before = None if task is None else task['status']
        result = super()._service(robot)
        if task is not None and kind in ('pickup', 'drop') and task['status'] != before:
            stock = self.inventory[task['package_id']]
            if kind == 'pickup':
                stock['state'] = 'inbound_transit' if task['flow'] == 'inbound' else 'outbound_transit'
                if task['flow'] == 'outbound':
                    stock['reserved'] = False
            else:
                stock['state'] = 'shelf' if task['flow'] == 'inbound' else 'shipped'
                location = dict(kind=stock['state'], cell=list(task['delivery']))
                task['package_location'] = location
                self.finished_tasks[-1]['package_location'] = deepcopy(location)
                self._event('stored' if task['flow'] == 'inbound' else 'shipped',
                            task_id=task['task_id'], package_id=task['package_id'], event_tick=self.tick+1)
        return result

    def cancel_task(self, task_id):
        # Returns/disposal require their own physical workflow; never erase stock.
        raise ValueError('Mixed-stock cancellation/returns are not implemented')

    def assert_invariants(self):
        super().assert_invariants()
        occupancy = Counter(tuple(s['storage_bay']) for s in self.inventory.values() if s['reserved'])
        if any(n > self.shelf_capacity for n in occupancy.values()):
            raise AssertionError('Shelf capacity exceeded')
        for stock in self.inventory.values():
            if stock['state'] in ('shelf', 'inbound_transit') and not stock['reserved']:
                raise AssertionError('Stored/incoming cargo has no shelf reservation')
        for r in self.robots:
            if r['package_id'] is not None:
                expected = 'inbound_transit' if self._task(r['current_task'])['flow'] == 'inbound' else 'outbound_transit'
                if self.inventory[r['package_id']]['state'] != expected:
                    raise AssertionError('Physical cargo and inventory disagree')

    def get_training_state(self):
        state = list(super().get_training_state())
        for bay in self.pickup_bays:
            items = [s for s in self.inventory.values() if tuple(s['storage_bay']) == bay]
            state.extend([sum(s['reserved'] for s in items)/self.shelf_capacity,
                          sum(s['state']=='shelf' for s in items)/self.shelf_capacity])
        for task in sorted(self.task_queue, key=lambda t: t['task_id']):
            state.extend([float(task['flow']=='inbound'), float(task['status']=='waiting_stock'),
                          float(task['status']=='waiting_capacity')])
        state.extend([0.] * (3*(self.config.max_orders-len(self.task_queue))))
        return np.asarray(state, dtype=np.float32)

    def get_layout_spec(self):
        return dict(super().get_layout_spec(), receiving_bays=[list(c) for c in self.receiving_bays])

    def get_global_state(self):
        return dict(super().get_global_state(), workload_version=self.workload_version,
                    inventory=deepcopy(self.inventory), shelf_capacity=self.shelf_capacity)


def make_mixed_scenario(seed, interval=20, horizon=600, num_agents=5, split='train', runtime_overrides=None):
    """Each external arrival has one put-away job and a later shipping request.

    Requests occur independently of policy speed. A shipping request waits for
    actual storage; it never conjures a second package. Arrival window is first
    two-thirds of the shift, followed by a fixed drain period for every policy.
    """
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        raise ValueError('Arrival interval must be a positive integer')
    config = RuntimeConfig(horizon=horizon, finite_batch=False, shuffle_roles=True,
                           sensor_dropout=.05, sensor_latency=1)
    if runtime_overrides:
        config = replace(config, **runtime_overrides)
    env = MixedWarehouseEnv(num_agents=num_agents, seed=seed, config=config)
    rng = np.random.default_rng(seed)
    for r in env.robots:
        r['capacity_eu'] *= float(rng.uniform(.7, 1.1))
        sampled_soc=float(rng.uniform(.35, 1))
        r['energy_eu'] = r['capacity_eu'] * (config.initial_soc if runtime_overrides and 'initial_soc' in runtime_overrides else sampled_soc)
        r['energy_factor'] = float(rng.uniform(.8, 1.2))
    events=[]
    arrival_ticks=list(range(0, horizon*2//3, interval))
    if 2*len(arrival_ticks)>config.max_orders:
        raise ValueError('Arrival stream can exceed the bounded order state; increase interval or max_orders')
    for index, tick in enumerate(arrival_ticks):
        bay = env.pickup_bays[int(rng.integers(len(env.pickup_bays)))]
        dock = env.receiving_bays[int(rng.integers(len(env.receiving_bays)))]
        shipping = env.drop_bays[int(rng.integers(len(env.drop_bays)))]
        stock=index
        for offset, flow, pickup, delivery, release in (
            (0,'inbound',dock,bay,tick), (1,'outbound',bay,shipping,tick+40)):
            order=dict(task_id=2*index+offset, package_id=stock, flow=flow, storage_bay=bay,
                       pickup=pickup, delivery=delivery, creation_timestep=0, priority=1,
                       deadline=240, status='pending')
            order=schedule_order(order, release)
            events.append(dict(id=f'{flow}-{index}', type='order', tick=release, order=order))
    env.set_events(events)
    env._sensor_history.clear();env._capture_sensor_snapshot()
    manifest=dict(seed=seed,split=split,workload_version=env.workload_version,arrival_interval=interval,
                  expected_packages=len(arrival_ticks),horizon=horizon,layout=env.get_layout_spec(),events=events,
                  initial_robots=deepcopy(env.robots),config=env.config.to_dict())
    manifest['scenario_hash']=sha256_digest(manifest)
    return env,manifest
