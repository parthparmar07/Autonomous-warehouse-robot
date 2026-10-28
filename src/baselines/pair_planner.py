"""Bounded joint search to resolve a blocked pair, including passing-pocket yielding."""
import heapq
from collections import deque
import math

def plan_pair(env,a,b,budget=5000):
    robots=[env.robots[a],env.robots[b]]
    start=tuple((r['x'],r['y']) for r in robots)
    goals=tuple(tuple(r['target'] or p) for r,p in zip(robots,start))
    if goals[0]==goals[1]:return None
    forbidden={(r['x'],r['y']) for r in env.robots if r['id'] not in (a,b)}
    def adjacent(cell):
        return [(cell[0]+dx,cell[1]+dy) for dx,dy in ((0,0),(0,-1),(1,0),(0,1),(-1,0))
                if env._is_in_bounds(cell[0]+dx,cell[1]+dy) and env.grid[cell[1]+dy,cell[0]+dx] not in (1,2)
                and (cell[0]+dx,cell[1]+dy) not in forbidden]
    # Makespan alone permits one robot to take a long unnecessary detour while
    # its partner progresses. That can exhaust a robot already beside a charger.
    # Minimize summed movement energy instead, with a small cost for waiting.
    movement=[(env.config.loaded_move_cost if r['load_status'] else env.config.move_cost)*r['energy_factor'] for r in robots]
    waiting=[env.config.wait_cost*r['energy_factor'] for r in robots]
    cache=getattr(env,'_paths',None)
    cache_key=('pair_plan',start,goals,frozenset(forbidden),tuple(movement),tuple(waiting),budget)
    if isinstance(cache,dict) and cache_key in cache:
        cached=cache[cache_key]
        return list(cached) if cached is not None else None
    distances=[]
    for goal in goals:
        distance={goal:0};queue=deque([goal])
        while queue:
            cell=queue.popleft()
            for neighbor in adjacent(cell):
                if neighbor not in distance:
                    distance[neighbor]=distance[cell]+1;queue.append(neighbor)
        distances.append(distance)
    def heuristic(state):
        return sum(table.get(position,math.inf)*unit for table,position,unit in zip(distances,state,movement))
    if not math.isfinite(heuristic(start)):return None
    fringe=[(heuristic(start),0,start)];cost={start:0.0};previous={start:None}
    expanded=0
    while fringe and expanded<budget:
        _,distance,state=heapq.heappop(fringe)
        if distance!=cost[state]:continue
        expanded+=1
        if state==goals:
            route=[]
            while state is not None:route.append(state);state=previous[state]
            route=route[::-1]
            if isinstance(cache,dict):cache[cache_key]=tuple(route)
            return route
        for first in adjacent(state[0]):
            for second in adjacent(state[1]):
                nxt=(first,second)
                if first==second or (first==state[1] and second==state[0]) or nxt==state:continue
                step=sum(waiting[index] if nxt[index]==state[index] else movement[index] for index in (0,1))
                candidate=distance+step
                if candidate+1e-12<cost.get(nxt,math.inf):
                    cost[nxt]=candidate;previous[nxt]=state
                    heapq.heappush(fringe,(candidate+heuristic(nxt),candidate,nxt))
    if isinstance(cache,dict):cache[cache_key]=None
    return None
