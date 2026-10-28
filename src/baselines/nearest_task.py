from __future__ import annotations
from src.safety.action_mask import orientation_delta


class NearestTaskDispatcher:
    """Shared energy-aware allocation plus a privileged-map reservation planner.

    This controller is an operational baseline, not a learned policy. All policies
    use the same environment allocator, masks and atomic physical resolver.
    """
    def __init__(self, warehouse_env,cooperative=True):
        self.warehouse_env = warehouse_env
        self.cooperative=cooperative
        self._pair_plan=None
        self._pair_retry={}

    def assign_tasks(self, tasks):
        if tasks is not self.warehouse_env.task_queue:
            self.warehouse_env.task_queue = list(tasks)
        self.warehouse_env.dispatch()

    def assign_pending_tasks(self, tasks):
        self.assign_tasks(tasks)

    def _next_step_towards(self, start, target):
        path = self.warehouse_env.path(start, target)
        return path[1] if path and len(path)>1 else tuple(start)

    @staticmethod
    def _direction_action(robot, next_cell):
        dx, dy = next_cell[0]-robot['x'], next_cell[1]-robot['y']
        if not dx and not dy:
            return 4
        desired = (1 if dx>0 else 3) if dx else (2 if dy>0 else 0)
        delta = (desired-robot['orientation'])%4
        return {0:0, 1:3, 2:1, 3:2}[delta]

    def choose_action(self, robot):
        env = self.warehouse_env
        if env._service_kind(robot) is not None and env.get_action_mask(robot['id'])[5]:
            return 5
        if not robot['active']:
            return 4
        if env.get_action_mask(robot['id'])[6]:
            return 6
        if robot['target'] is None:
            return 4
        action = self._direction_action(robot,self._next_step_towards((robot['x'],robot['y']),robot['target']))
        return action if env.get_action_mask(robot['id'])[action] else 4

    def act(self, env):
        self.warehouse_env = env
        env.prepare_decision()
        if not self.cooperative:return [self.choose_action(r) for r in env.robots]
        actions = [4]*env.num_agents
        positions = [(r['x'],r['y']) for r in env.robots]
        occupied = set(positions)
        reserved = set()
        # Commit short joint plans instead of repeatedly waiting at a head-on
        # blockage. Plans include physical yielding; the normal resolver remains
        # authoritative when other traffic changes a proposed move.
        planned={}
        if self._pair_plan:
            ids,goals,route=self._pair_plan
            current=tuple(positions[i] for i in ids)
            valid=all(env.robots[i]['active'] and tuple(env.robots[i]['target'] or positions[i])==g for i,g in zip(ids,goals))
            while len(route)>1 and current==route[1]:route.pop(0)
            # Other traffic can enter a committed route while a participant is
            # turning. A stale plan must not repeatedly request an illegal swap
            # with that third robot instead of planning with the new blocker.
            outsiders={position for i,position in enumerate(positions) if i not in ids}
            if len(route)>1:
                valid=valid and all(cell not in outsiders and env._is_in_bounds(*cell)
                    and env.grid[cell[1],cell[0]] not in (1,2) for cell in route[1])
            if valid and len(route)>1 and all(p in (old,nxt) for p,old,nxt in zip(current,route[0],route[1])):
                planned=dict(zip(ids,route[1]))
            else:self._pair_plan=None
        if not self._pair_plan:
            from src.baselines.pair_planner import plan_pair
            for i,r in enumerate(env.robots):
                if not r['active'] or r['target'] is None:continue
                static=env.path(positions[i],r['target'])
                if not static or len(static)<2 or env.path(positions[i],r['target'],occupied-{positions[i]}) is not None:continue
                blockers=[j for j in range(env.num_agents) if j!=i and positions[j] in static[1:] and env.robots[j]['active']]
                if not blockers:continue
                j=blockers[0]
                if env.tick<self._pair_retry.get((i,j),0):continue
                route=plan_pair(env,i,j,budget=2000 if env.num_agents<=5 else 200)
                if route and len(route)>1:
                    ids=(i,j);goals=tuple(tuple(env.robots[k]['target'] or positions[k]) for k in ids)
                    self._pair_plan=(ids,goals,route);planned=dict(zip(ids,route[1]));break
                self._pair_retry[(i,j)]=env.tick+10
                break
        for i,cell in planned.items():
            # One participant can reach its bay before its partner completes a
            # joint route. Do not let a planned no-op suppress physical service.
            if env._service_kind(env.robots[i]) is not None and env.get_action_mask(i)[5]:
                actions[i]=5;reserved.add(positions[i])
                continue
            actions[i]=self._direction_action(env.robots[i],cell)
            if not env.get_action_mask(i)[actions[i]]:actions[i]=4
            if actions[i] in (0,1):occupied.discard(positions[i]);reserved.add(cell)
            else:reserved.add(positions[i])
        for i in sorted(range(env.num_agents),key=lambda i:(i-env.tick)%env.num_agents):
            if i in planned:continue
            robot = env.robots[i]
            start = positions[i]
            if env.get_action_mask(i)[6]:
                actions[i]=6;reserved.add(start)
                continue
            if (not robot['active'] and not env.get_action_mask(i)[5]) or robot['target'] is None:
                reserved.add(start)
                continue
            if env._service_kind(robot) is not None and env.get_action_mask(i)[5]:
                actions[i] = 5
                reserved.add(start)
                continue
            blocked = (occupied-{start})|reserved
            path = env.path(start,robot['target'],blocked)
            if path and len(path)>1:
                action = self._direction_action(robot,path[1])
                if env.get_action_mask(i)[action]:
                    actions[i] = action
                    if action in (0,1):
                        dx,dy=orientation_delta(robot['orientation'],action)
                        reserved.add((robot['x']+dx,robot['y']+dy))
                        occupied.discard(start)
                        continue
            reserved.add(start)
        return actions

    @staticmethod
    def _distance(x,y,target):
        return abs(target[0]-x)+abs(target[1]-y)
