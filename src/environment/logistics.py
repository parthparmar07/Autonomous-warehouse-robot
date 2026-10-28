from __future__ import annotations
from copy import deepcopy
from collections import deque
import math


class LogisticsMixin:
    def _task(self, task_id):
        return next((t for t in self.task_queue if t['task_id'] == task_id), None)

    def _normalize_tasks(self):
        if len(self.task_queue) > self.config.max_orders:
            raise ValueError('Task queue exceeds configured state capacity')
        ids = [t['task_id'] for t in self.task_queue]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate order ID')
        for task in self.task_queue:
            tid = task['task_id']
            if tid not in self._ledger:
                if any(t['task_id'] == tid for t in self.finished_tasks):
                    raise ValueError('Completed order ID reused')
                for key, value in dict(owner=None, status='pending', creation_timestep=self.tick,
                                      deadline=self.tick+self.config.horizon, priority=1, package_id=tid).items():
                    task.setdefault(key, value)
                task.setdefault('package_location', dict(kind='shelf', cell=list(task.get('shelf', task['pickup']))))
                for endpoint in ('pickup', 'delivery'):
                    x, y = task[endpoint]
                    if any(isinstance(v,bool) or not isinstance(v,int) for v in (x,y)) or not self._is_in_bounds(x, y) or self.grid[y, x] in (1, 2):
                        raise ValueError('Invalid order endpoint')
                if task['creation_timestep']>self.tick:raise ValueError('Unreleased orders must use the event stream')
                if not all(math.isfinite(task[k]) for k in ('creation_timestep','deadline','priority')) or task['priority']<=0:
                    raise ValueError('Invalid order timing/priority')
                self._ledger[tid] = task
        for robot in self.robots:
            if robot['current_task'] is not None and not self._task(robot['current_task']):
                if robot['package_id'] is not None:
                    raise ValueError('Carried package lost its order')
                robot.update(current_task=None, target=None, mode='IDLE')

    def path(self, start, target, occupied=()):
        start, target = tuple(start), tuple(target)
        forbidden = frozenset(occupied)
        key = (start, target, forbidden)
        if key in self._paths:
            return self._paths[key]
        if not self._is_in_bounds(*target) or self.grid[target[1], target[0]] in (1, 2):
            return None
        queue, previous = deque([start]), {start: None}
        while queue:
            cell = queue.popleft()
            if cell == target:
                result = []
                while cell is not None:
                    result.append(cell)
                    cell = previous[cell]
                result.reverse()
                if len(self._paths) > 20000:
                    static = {k: v for k, v in self._paths.items() if isinstance(k, tuple) and len(k) == 3 and not k[2]}
                    self._paths.clear()
                    self._paths.update(static)
                self._paths[key] = result
                return result
            for dx, dy in ((0,-1),(1,0),(0,1),(-1,0)):
                nxt = (cell[0]+dx, cell[1]+dy)
                if nxt not in previous and nxt not in forbidden and self._is_in_bounds(*nxt) and self.grid[nxt[1],nxt[0]] not in (1,2):
                    previous[nxt] = cell
                    queue.append(nxt)
        return None

    def route_energy(self, start, target, loaded=False, factor=1.0):
        route = self.path(start, target)
        if route is None:
            return math.inf
        # Conservative bound: at most one quarter turn per step (backward is allowed).
        cost = self.config.loaded_move_cost if loaded else self.config.move_cost
        return (len(route)-1)*(cost+self.config.turn_cost)*factor

    def mission_energy(self, robot, task):
        cost = self.route_energy((robot['x'],robot['y']), task['pickup'], False, robot['energy_factor'])
        cost += self.route_energy(task['pickup'], task['delivery'], True, robot['energy_factor'])
        cost += 2*self.config.service_ticks*self.config.service_cost*robot['energy_factor']
        if self.chargers:
            cost += min((self.route_energy(task['delivery'], c['position'], False, robot['energy_factor'])
                         for c in self.chargers if c['online']), default=math.inf)
        return cost+self.config.reserve

    def _clear_interrupted_service(self, robot):
        robot.update(service_remaining=0, service_key=None)
        if robot.get('mode') == 'PICKING':
            robot['mode'] = 'TO_PICKUP' if robot.get('current_task') is not None else 'IDLE'
        elif robot.get('mode') == 'DROPPING':
            robot['mode'] = 'TO_DROP' if robot.get('current_task') is not None else 'IDLE'

    def _release_task(self, robot, reason):
        task = self._task(robot['current_task'])
        if task is not None and robot['package_id'] is None:
            task.update(owner=None, status='pending')
            self._event('task_released', robot_id=robot['id'], task_id=task['task_id'], reason=reason)
            robot.update(current_task=None, target=None, service_remaining=0, service_key=None)
            if robot.get('mode') in ('TO_PICKUP', 'PICKING', 'TO_DROP', 'DROPPING'):
                robot['mode'] = 'IDLE' if robot.get('active') else robot.get('mode')

    def dispatch(self):
        self._normalize_tasks()
        candidates=[]
        for robot in self.robots:
            if self._transfer_sender(robot) is not None:
                continue
            if not robot['active']:
                if robot['package_id'] is None and self.tick-self._faults.get(robot['id'],self.tick) >= self.config.heartbeat_timeout:
                    self._release_task(robot, 'heartbeat_timeout')
                continue
            if robot['current_task'] is not None:
                robot['lease_tick'] = self.tick
                if robot['package_id'] is None and self.tick-robot['last_progress_tick']>=self.config.lease_ticks:
                    self._release_task(robot,'stalled_lease')
                    robot['lease_excluded_until']=self.tick+max(1,self.config.lease_ticks//2)
            if robot['charger_id'] is not None:
                continue
            if robot['current_task'] is not None:
                continue
            # Low SOC makes a charging decision available to the actor. Do not
            # pre-empt it with an automatic reservation or another new order.
            if self.chargers and robot['energy_eu']/robot['capacity_eu']<=self.config.charge_trigger:
                robot.update(target=None,mode='IDLE',wait_reason='AWAITING_CHARGE_REQUEST')
                continue
            if self.tick<robot.get('lease_excluded_until',0):
                robot.update(target=robot['parking'],mode='TO_PARK')
                continue
            if robot['package_id'] is None:candidates.append(robot)
        available=[t for t in self.task_queue if t.get('owner') is None and t['status']=='pending' and t['creation_timestep']<=self.tick]
        # Compare all feasible robot/order pairs before assigning any of them.
        # Cost dominates robot ID. Exact ties rotate with seed/tick, reproducibly.
        pairs=[]
        for robot in candidates:
            for task in available:
                if self.mission_energy(robot,task)<=robot['energy_eu']+1e-9:
                    pairs.append(((-task['priority'],task['deadline'],
                        self.route_energy((robot['x'],robot['y']),task['pickup']),
                        (robot['id']-self.tick-self.seed)%self.num_agents,task['task_id']),robot,task))
        assigned=set()
        for _,robot,task in sorted(pairs,key=lambda row:row[0]):
            if robot['id'] in assigned or task.get('owner') is not None:continue
            task['owner']=robot['id'];assigned.add(robot['id'])
            robot.update(current_task=task['task_id'],target=tuple(task['pickup']),mode='TO_PICKUP',lease_tick=self.tick,
                         last_progress_tick=self.tick,service_remaining=0,service_key=None)
            self._event('assigned',task_id=task['task_id'],robot_id=robot['id'])
        for robot in candidates:
            if robot['id'] in assigned:continue
            if robot['load_status']==0:
                robot.update(target=robot['parking'],mode='TO_PARK' if (robot['x'],robot['y'])!=robot['parking'] else 'IDLE')

    def voluntary_charge_allowed(self, agent_id):
        """Own-state guard, shared by the action mask and request transition.

        An exploratory request cannot abandon an assigned order. Below the SOC
        trigger, an idle robot chooses when to request instead of auto-charging.
        This check deliberately does not expose remote queues/outages in masks.
        """
        robot=self.robots[agent_id]
        return (robot['active'] and robot['current_task'] is None and robot['package_id'] is None
            and robot['charger_id'] is None and bool(self.chargers)
            and self.tick>=robot.get('charge_retry_after',0)
            and robot['energy_eu']/robot['capacity_eu']<=self.config.charge_trigger)

    def request_charge(self, agent_id, *, automatic=False):
        # automatic=True is a privileged fixture/controller hook, never used by
        # normal dispatch or exposed as a policy action/API request flag.
        robot = self.robots[agent_id]
        if not automatic and not self.voluntary_charge_allowed(agent_id):return False
        if not robot['active'] or not self.chargers:
            return False
        if self.tick<robot.get('charge_retry_after',0):return False
        if robot['energy_eu']/robot['capacity_eu']>=self.config.charge_release:return False
        if robot['charger_id'] is not None:
            return True
        online_open = [c for c in self.chargers if c['online'] and len(c['queue'])<min(self.config.queue_capacity,len(c['queue_cells']))]
        choices = [c for c in online_open
                   if self.route_energy((robot['x'],robot['y']),c['position'],bool(robot['load_status']),robot['energy_factor'])<=robot['energy_eu']+1e-9]
        if not choices and robot['package_id'] is None and robot['energy_eu']>0:
            choices = online_open
        if not choices:
            self.episode_metrics['charger_rejections'] += 1
            if not automatic:
                robot['charge_retry_after'] = max(robot.get('charge_retry_after',0), self.tick+5)
            return False
        station = min(choices,key=lambda c:(len(c['queue'])+(c['owner'] is not None),self.route_energy((robot['x'],robot['y']),c['position'])))
        if robot['package_id'] is None:
            self._release_task(robot,'charging')
        station['queue'].append(agent_id)
        station['requested'][agent_id] = self.tick
        robot['charger_id'] = station['id']
        robot['last_progress_tick']=self.tick
        robot['charger_approach_tick']=self.tick
        self._update_chargers()
        self._event('charge_requested',robot_id=agent_id,charger_id=station['id'],
                    source='dispatcher' if automatic else 'policy')
        return True

    def _leave_charger(self, robot):
        if robot['charger_id'] is not None:
            station = self.chargers[robot['charger_id']]
            if station['owner']==robot['id']:
                station['owner'] = None
            station['queue'] = [i for i in station['queue'] if i!=robot['id']]
            station['requested'].pop(robot['id'],None)
        robot['charger_id'] = None
        robot.pop('charge_started_tick',None)
        robot.pop('charger_approach_tick',None)

    def _update_chargers(self):
        for station in self.chargers:
            if station['owner'] is not None:
                robot = self.robots[station['owner']]
                if (robot['x'],robot['y'])!=station['position'] and self.tick-robot.get('charger_approach_tick',self.tick)>=self.config.lease_ticks:
                    self._event('charger_lease_expired',robot_id=robot['id'],charger_id=station['id'])
                    self._leave_charger(robot)
                    robot['charge_retry_after']=self.tick+max(1,self.config.lease_ticks//2)
            if station['owner'] is not None:
                robot = self.robots[station['owner']]
                connected = robot['health']=='OK' and (robot['x'],robot['y'])==station['position']
                if not station['online'] or (not robot['active'] and not connected) or robot['energy_eu']/robot['capacity_eu']>=self.config.charge_release:
                    if robot['energy_eu']/robot['capacity_eu']>=self.config.charge_release and robot.get('charge_started_tick') is not None:
                        self._event('charge_completed',robot_id=robot['id'],charger_id=station['id'])
                    self._leave_charger(robot)
                    if robot['active']:
                        task = self._task(robot['current_task'])
                        robot.update(mode='TO_DROP' if task and robot['package_id'] is not None else 'IDLE',target=tuple(task['delivery']) if task else None)
                    station['services'] += 1
            if station['online'] and station['owner'] is None and station['queue']:
                old = [i for i in station['queue'] if self.tick-station['requested'][i]>=100]
                agent = min(old,key=lambda i:station['requested'][i]) if old else min(station['queue'],key=lambda i:(self.robots[i]['energy_eu']/self.robots[i]['capacity_eu'],station['requested'][i],(i-self.tick)%self.num_agents))
                station['queue'].remove(agent)
                station['owner'] = agent
                self.robots[agent]['charger_approach_tick']=self.tick
                self.robots[agent]['last_progress_tick']=self.tick
            if station['owner'] is not None:
                owner=self.robots[station['owner']]
                connected=(owner['x'],owner['y'])==station['position'] and owner.get('charge_started_tick') is not None
                owner.update(target=station['position'],mode='CHARGING' if connected else 'TO_CHARGER')
            for index, agent in enumerate(station['queue']):
                self.robots[agent].update(target=station['queue_cells'][index],mode='QUEUED_CHARGE')

    def cancel_task(self, task_id):
        task = self._task(task_id)
        if task is None:
            return
        robot = self.robots[task['owner']] if task.get('owner') is not None else None
        if robot is not None and robot['package_id'] is not None:
            task['cancel_requested'] = True
            target = self.transfer_bays[0] if self.transfer_bays else task['pickup']
            task['delivery'] = tuple(target)
            robot.update(target=tuple(target),mode='TO_DROP')
        else:
            if robot:
                robot.update(current_task=None,target=None,mode='IDLE')
            task.update(status='cancelled',owner=None,completed_tick=self.tick)
            self.task_queue.remove(task)
            self.finished_tasks.append(deepcopy(task))
            self.episode_metrics['cancellations'] += 1

    def transfer_package(self, sender_id, receiver_id, authorized=False):
        """Authorize a physical handover; receiver must SERVICE for distinct ticks.

        Repeated controller/API calls cannot advance service or move cargo. A
        receiver movement, interrupted service, or lost adjacency resets progress.
        """
        sender, receiver = self.robots[sender_id], self.robots[receiver_id]
        if (not authorized or sender_id==receiver_id or sender['package_id'] is None or receiver['package_id'] is not None
            or receiver['current_task'] is not None or not receiver['active'] or (sender['x'],sender['y']) not in self.transfer_bays
            or abs(sender['x']-receiver['x'])+abs(sender['y']-receiver['y'])!=1):
            raise ValueError('Physical transfer requires an authorized dock and adjacent empty receiver')
        if receiver['energy_eu']<self.config.service_cost*receiver['energy_factor']:
            raise ValueError('Receiver cannot fund transfer service')
        receiver['transfer_from'] = sender_id
        return True

    def _transfer_sender(self, receiver):
        sender_id = receiver.get('transfer_from')
        if sender_id is None:
            return None
        sender = self.robots[sender_id]
        if (sender['package_id'] is None or receiver['package_id'] is not None
            or receiver['current_task'] is not None or receiver['charger_id'] is not None
            or (sender['x'],sender['y']) not in self.transfer_bays
            or abs(sender['x']-receiver['x'])+abs(sender['y']-receiver['y'])!=1):
            receiver.pop('transfer_from',None)
            receiver.update(service_remaining=0,service_key=None)
            return None
        return sender

    def _transfer_service(self, receiver):
        sender = self._transfer_sender(receiver)
        if sender is None:
            return False
        sender_id, receiver_id = sender['id'], receiver['id']
        key = ('transfer',sender_id,receiver_id)
        receiver['service_remaining'] = receiver['service_remaining']-1 if receiver['service_key']==key else self.config.service_ticks-1
        receiver['service_key'] = key
        if receiver['service_remaining']:
            return True
        task = self._task(sender['current_task'])
        task.update(owner=receiver_id,status='picked',package_location=dict(kind='robot',robot_id=receiver_id))
        receiver.update(current_task=sender['current_task'],package_id=sender['package_id'],load_status=1,target=tuple(task['delivery']),mode='TO_DROP',service_key=None)
        sender.update(current_task=None,package_id=None,load_status=0,target=None,service_remaining=0,service_key=None,
            mode='IDLE' if sender['active'] else 'FAULTED' if sender['health']!='OK' else 'DEPLETED')
        receiver.pop('transfer_from',None)
        self._event('physical_transfer',task_id=task['task_id'],sender=sender_id,receiver=receiver_id,event_tick=self.tick+1)
        self.assert_invariants()
        return True

    def _service_kind(self, robot):
        position = (robot['x'],robot['y'])
        if self._transfer_sender(robot) is not None:
            return 'transfer'
        if robot['charger_id'] is not None:
            station = self.chargers[robot['charger_id']]
            return 'charge' if station['online'] and station['owner']==robot['id'] and position==station['position'] else None
        task = self._task(robot['current_task'])
        if task and task.get('owner')==robot['id']:
            if task['status']=='pending' and position==tuple(task['pickup']):
                return 'pickup'
            if task['status']=='picked' and position==tuple(task['delivery']):
                return 'drop'
        return None

    def _service(self, robot):
        kind = self._service_kind(robot)
        if kind=='transfer':
            return self._transfer_service(robot)
        if kind=='charge':
            if robot.get('charge_started_tick') is None:
                robot['charge_started_tick']=self.tick
                station=self.chargers[robot['charger_id']]
                self._event('charge_started',robot_id=robot['id'],charger_id=station['id'],queue_wait_ticks=self.tick-station['requested'][robot['id']])
            duration, incoming = 1.0, 0.0
            while duration>1e-12 and robot['energy_eu']<robot['capacity_eu']:
                boundary = robot['capacity_eu']*self.config.taper_soc
                bulk = robot['energy_eu']<boundary-1e-9
                rate = self.config.charge_rate if bulk else self.config.taper_rate
                stop = boundary if bulk else robot['capacity_eu']
                dt = min(duration,(stop-robot['energy_eu'])/rate)
                robot['energy_eu'] += dt*rate
                incoming += dt*rate
                duration -= dt
            self.episode_metrics['energy_charged'] += incoming
            robot.update(mode='CHARGING',wait_reason='CHARGING')
            return incoming>0
        task = self._task(robot['current_task'])
        if kind not in ('pickup','drop') or task is None:
            self._clear_interrupted_service(robot)
            return False
        key = (task['task_id'],kind)
        robot['service_remaining'] = robot['service_remaining']-1 if robot['service_key']==key else self.config.service_ticks-1
        robot['service_key'] = key
        robot['mode'] = 'PICKING' if kind=='pickup' else 'DROPPING'
        robot['wait_reason'] = 'SERVICE'
        if robot['service_remaining']:
            return True
        robot['service_key'] = None
        if kind=='pickup':
            task.update(status='picked',picked_tick=self.tick+1,package_location=dict(kind='robot',robot_id=robot['id']))
            robot.update(load_status=1,package_id=task['package_id'],target=tuple(task['delivery']),mode='TO_DROP')
            self._event('picked',robot_id=robot['id'],task_id=task['task_id'],event_tick=self.tick+1)
        else:
            cancelled = task.get('cancel_requested',False)
            task.update(status='cancelled' if cancelled else 'delivered',completed_tick=self.tick+1,owner=None,
                        package_location=dict(kind='return' if cancelled else 'outbound',cell=list(task['delivery'])))
            self.task_queue.remove(task)
            self.finished_tasks.append(deepcopy(task))
            self.completed_tasks += int(not cancelled)
            self.episode_metrics['cancellations' if cancelled else 'delivered'] += 1
            robot.update(load_status=0,package_id=None,current_task=None,target=None,mode='IDLE')
            self._event('cancel_returned' if cancelled else 'delivered',robot_id=robot['id'],task_id=task['task_id'],event_tick=self.tick+1)
        return True
