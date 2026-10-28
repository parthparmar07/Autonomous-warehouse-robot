from copy import deepcopy
import math
import numpy as np
from src.environment.movement import resolve_moves
from src.environment.spec import ACTION_DIM
from src.safety.action_mask import orientation_delta


class PhysicsMixin:
    def _event(self, kind, **fields):
        stamp=fields.pop('event_tick',self.tick)
        self.event_log.append(dict(tick=stamp,type=kind,**fields))

    def set_events(self, events):
        events = deepcopy(list(events))
        ids = [e['id'] for e in events]
        if len(ids)!=len(set(ids)):
            raise ValueError('Duplicate event IDs')
        allowed = {'order','fault','repair','charger_outage','charger_online','block','unblock','cancel'}
        if any(e.get('type') not in allowed or not isinstance(e.get('tick'),int) or e['tick']<self.tick for e in events):
            raise ValueError('Invalid scheduled event')
        for event in events:
            if event['type'] in ('fault','repair') and (isinstance(event.get('robot_id'),bool) or not isinstance(event.get('robot_id'),int) or not 0<=event['robot_id']<self.num_agents):
                raise ValueError('Invalid event robot ID')
            if event['type'] in ('charger_outage','charger_online') and (not isinstance(event.get('charger_id'),int) or not 0<=event['charger_id']<len(self.chargers)):
                raise ValueError('Invalid event charger ID')
        self._events = sorted(events,key=lambda e:(e['tick'],str(e['id'])))
        self.prepare_decision()

    def _process_events(self):
        for event in self._events:
            if event['id'] in self._consumed_events or event['tick']>self.tick:
                continue
            self._consumed_events.add(event['id'])
            kind = event['type']
            if kind=='order':
                self.task_queue.append(deepcopy(event['order']))
            elif kind=='fault':
                robot = self.robots[event['robot_id']]
                robot.update(active=False,health='IMMOBILIZED',mode='FAULTED')
                self._faults[robot['id']] = self.tick
                if robot['package_id'] is not None:
                    task = self._task(robot['current_task'])
                    if task:
                        task['status'] = 'blocked_recovery'
                        robot['mode'] = 'BLOCKED_RECOVERY'
                self._leave_charger(robot)
            elif kind=='repair':
                robot = self.robots[event['robot_id']]
                robot.update(active=robot['energy_eu']>0,health='OK')
                self._faults.pop(robot['id'],None)
                task = self._task(robot['current_task'])
                if task and robot['package_id'] is not None:
                    task['status'] = 'picked'
                    robot.update(target=tuple(task['delivery']),mode='TO_DROP')
                else:
                    robot['mode'] = 'IDLE' if robot['active'] else 'DEPLETED'
            elif kind in ('charger_outage','charger_online'):
                charger = self.chargers[event['charger_id']]
                charger['online'] = kind=='charger_online'
                if not charger['online']:
                    for robot in self.robots:
                        if robot['charger_id']==charger['id']:
                            self._leave_charger(robot)
                            task=self._task(robot['current_task'])
                            robot.update(target=tuple(task['delivery']) if task and robot['load_status'] else None,mode='TO_DROP' if task and robot['load_status'] else 'IDLE')
            elif kind in ('block','unblock'):
                cell = tuple(event['cell'])
                if not self._is_in_bounds(*cell):
                    raise ValueError('Blockage outside layout')
                if kind=='block':
                    if cell in {(r['x'],r['y']) for r in self.robots} or self._base_grid[cell[1],cell[0]]!=0:
                        self._event('blockage_rejected',cell=cell)
                        continue
                    self.grid[cell[1],cell[0]] = 1
                else:
                    self.grid[cell[1],cell[0]] = self._base_grid[cell[1],cell[0]]
                self._paths.clear()
            elif kind=='cancel':
                self.cancel_task(event['task_id'])
            affected=self.robots[event['robot_id']]['current_task'] if kind=='fault' else None
            self._event(kind,event_id=event['id'],robot_id=event.get('robot_id'),task_id=affected)

    def prepare_decision(self):
        self._process_events()
        self._update_chargers()
        self.dispatch()
        return self.get_global_state()

    def _debit(self, robot, amount):
        debit = min(robot['energy_eu'],max(0.0,float(amount)))
        robot['energy_eu'] = max(0.0,robot['energy_eu']-debit)
        self.episode_metrics['energy_used'] += debit
        if robot['energy_eu']<=1e-9:
            robot['energy_eu'] = 0.0
            connected = self._service_kind(robot)=='charge' and robot['health']=='OK'
            if robot['active'] and not connected:
                self.episode_metrics['avoidable_depletions'] += 1
                self._event('depleted',robot_id=robot['id'],event_tick=self.tick+1)
            robot.update(active=False,mode='DEPLETED')
            if robot['package_id'] is not None and not connected:
                self._task(robot['current_task'])['status']='blocked_recovery'
                robot['mode']='BLOCKED_RECOVERY'
            self._faults.setdefault(robot['id'],self.tick)
        return debit

    def _action_cost(self, robot, action):
        if action in (0,1):
            base = self.config.loaded_move_cost if robot['load_status'] else self.config.move_cost
        elif action in (2,3):
            base = self.config.turn_cost
        elif action==5 and self._service_kind(robot) in ('pickup','drop','transfer'):
            base = self.config.service_cost
        else:
            base = self.config.wait_cost
        return base*robot['energy_factor']

    def step(self, actions):
        if len(actions)!=self.num_agents or any(isinstance(a,bool) or not isinstance(a,(int,np.integer)) or not 0<=a<ACTION_DIM for a in actions):
            raise ValueError('One integer action in 0..6 is required for each robot')
        if self.tick>=self.config.horizon:
            raise RuntimeError('Episode is truncated; reset before stepping')
        self.prepare_decision()
        # Keep this diagnostic separate from actor inactivity and reward. Old
        # resumable environments need not already contain the new metric key.
        locked=self.charge_locked_infeasible_agents()
        self.episode_metrics['charge_locked_infeasible_ticks']=self.episode_metrics.get('charge_locked_infeasible_ticks',0)+locked
        before_potential=self.task_potential() if self.config.potential_weight else 0.0
        before_completed = self.completed_tasks
        before_energy = self.episode_metrics['energy_used']
        before_depleted = self.episode_metrics['avoidable_depletions']
        origins = [(r['x'],r['y']) for r in self.robots]
        previous_targets = [r['target'] for r in self.robots]
        destinations, accepted = list(origins),list(actions)
        executed=[4]*self.num_agents
        for i,(robot,action) in enumerate(zip(self.robots,actions)):
            robot['wait_reason'] = 'NONE'
            if not robot['active'] and not (action==5 and self._service_kind(robot)=='charge' and robot['health']=='OK'):
                accepted[i]=4
                robot['wait_reason']='DEPLETION' if robot['energy_eu']==0 else 'FAULT'
                continue
            if action==6:
                if self.request_charge(i):executed[i]=6
                accepted[i]=4
            elif action==5 and self._service_kind(robot)=='charge':
                # Connected external power can restart a healthy empty battery;
                # the debit is clipped to stored energy before the measured credit.
                pass
            elif action in (0,1):
                dx,dy=orientation_delta(robot['orientation'],action)
                target=(robot['x']+dx,robot['y']+dy)
                if not self._is_in_bounds(*target) or self.grid[target[1],target[0]] in (1,2):
                    if self._is_in_bounds(*target) and self.grid[target[1],target[0]]==2:
                        self.episode_metrics['safety_overrides']+=1
                    accepted[i]=4
                    robot['wait_reason']='BLOCKED'
                elif robot['energy_eu']+1e-9>=self._action_cost(robot,action):
                    destinations[i]=target
                else:
                    accepted[i]=4
                    robot['wait_reason']='DEPLETION'
            elif robot['energy_eu']+1e-9<self._action_cost(robot,action):
                accepted[i]=4
                robot['wait_reason']='DEPLETION'
        priority=sorted(range(self.num_agents),key=lambda i:(i-self.tick)%self.num_agents)
        resolved,conflicts=resolve_moves(origins,destinations,priority)
        self.episode_metrics['collision_count']+=len(conflicts)
        progress=False
        for i,robot in enumerate(self.robots):
            if not robot['active'] and not (accepted[i]==5 and self._service_kind(robot)=='charge' and robot['health']=='OK'):
                continue
            action=accepted[i]
            if destinations[i]!=origins[i] and resolved[i]==origins[i]:
                action=4
                robot['wait_reason']='BLOCKED'
                self.episode_metrics['blocked_ticks']+=1
            if action==5 and self._service_kind(robot) is None:
                action=4
            robot['x'],robot['y']=resolved[i]
            if executed[i]!=6:executed[i]=int(action)
            if action in (2,3):
                robot['orientation']=(robot['orientation']+(-1 if action==2 else 1))%4
            kind=self._service_kind(robot)
            if kind=='charge' and action==5:
                self._debit(robot,self.config.wait_cost*robot['energy_factor'])
                if robot['health']=='OK':
                    robot['active']=True
                progress=self._service(robot) or progress
                self.episode_metrics['charging_ticks']+=1
                robot['last_progress_tick']=self.tick
            else:
                self._debit(robot,self._action_cost(robot,action))
                if action==5 and robot['active']:
                    progress=self._service(robot) or progress
                    if kind is not None:robot['last_progress_tick']=self.tick
                else:
                    self._clear_interrupted_service(robot)
            target=previous_targets[i]
            if target:
                old_p=self.path(origins[i],target)
                new_p=self.path(resolved[i],target)
                if (new_p and old_p and len(new_p)<len(old_p)) or math.dist(resolved[i],target)<math.dist(origins[i],target):
                    progress=True
                    robot['last_progress_tick']=self.tick
                    if robot['charger_id'] is not None:
                        robot['charger_approach_tick']=self.tick
            if action==4 and robot['wait_reason']=='NONE':
                robot['wait_reason']='NO_WORK' if robot['current_task'] is None else 'UNKNOWN'
                if robot['current_task'] is not None:
                    self.episode_metrics['idle_with_assignment_ticks']+=1
        for robot,action in zip(self.robots,executed):
            robot['executed_actions']=[action]+robot['executed_actions'][:-1]
        self.episode_metrics['step_count']+=1
        self.prepare_decision()
        backlog=len(self.task_queue)
        late=sum(t['priority'] for t in self.task_queue if self.tick>t['deadline'])
        self.episode_metrics['backlog_ticks']+=backlog
        self.episode_metrics['charge_queue_ticks']+=sum(len(c['queue']) for c in self.chargers)
        self.episode_metrics['tardiness_ticks']+=late
        future=any(e['id'] not in self._consumed_events and e['type']=='order' for e in self._events)
        terminated=bool(self.config.finite_batch and not self.task_queue and not future)
        potential=0.0 if terminated else self.task_potential() if self.config.potential_weight else 0.0
        self.reward_components=dict(delivery=self.config.delivery_reward*(self.completed_tasks-before_completed),
            backlog=-self.config.backlog_cost*backlog,
            energy=-self.config.energy_reward_cost*(self.episode_metrics['energy_used']-before_energy),
            conflicts=-self.config.conflict_cost*len(conflicts),lateness=-self.config.lateness_cost*late,
            depletion=-self.config.depletion_cost*(self.episode_metrics['avoidable_depletions']-before_depleted),
            idle_bay=-self.config.idle_bay_cost*self.idle_service_bay_occupants(),
            shaping=self.config.potential_weight*(self.config.potential_gamma*potential-before_potential))
        reward=sum(self.reward_components.values())
        self.episode_metrics['reward']+=reward
        if progress or self.completed_tasks>before_completed:
            self._last_progress=self.tick
            self._deadlock_reported=False
        elif backlog and self.tick-self._last_progress>=self.config.deadlock_ticks and not self._deadlock_reported:
            self.episode_metrics['deadlocks']+=1
            self._deadlock_reported=True
        self._capture_sensor_snapshot()
        positions=[(r['x'],r['y']) for r in self.robots]
        self.episode_metrics['executed_collisions']+=len(positions)-len(set(positions))
        self.episode_metrics['executed_safety_violations']+=sum(self.grid[y,x]==2 for x,y in positions)
        self.assert_invariants()
        state=self.get_global_state()
        future=any(e['id'] not in self._consumed_events and e['type']=='order' for e in self._events)
        state.update(reward=reward,reward_components=dict(self.reward_components),terminated=terminated,truncated=self.tick>=self.config.horizon)
        state['done']=state['terminated'] or state['truncated']
        return state

    def get_action_mask(self, agent_id):
        robot=self.robots[agent_id]
        if not robot['active']:
            return [False,False,False,False,True,self._service_kind(robot)=='charge' and robot['health']=='OK',False]
        mask=[True]*ACTION_DIM
        # Static map is public prior knowledge. Live occupancy is resolved jointly;
        # unseen robot locations do not leak through the actor's availability mask.
        for action in (0,1):
            dx,dy=orientation_delta(robot['orientation'],action)
            x,y=robot['x']+dx,robot['y']+dy
            mask[action]=self._is_in_bounds(x,y) and self._base_grid[y,x] not in (1,2)
        for action in range(ACTION_DIM):
            mask[action]=bool(mask[action] and robot['energy_eu']+1e-9>=self._action_cost(robot,action))
        mask[4]=True
        kind=self._service_kind(robot)
        mask[5]=(mask[5] and kind is not None) or (kind=='charge' and robot['health']=='OK')
        mask[6]=mask[6] and self.voluntary_charge_allowed(agent_id)
        return mask
