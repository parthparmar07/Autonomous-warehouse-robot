"""Atomic point-robot movement: vertex conflicts, swaps, dependency propagation."""
from collections import defaultdict


def resolve_moves(origins, destinations, priority):
    if len(set(origins)) != len(origins):
        raise ValueError("Initial robot positions overlap")
    ranks = {agent: rank for rank, agent in enumerate(priority)}
    proposed = list(destinations)
    conflicts = set()
    groups = defaultdict(list)
    for i, destination in enumerate(proposed):
        groups[destination].append(i)
    for destination, agents in groups.items():
        if len(agents) < 2:
            continue
        stationary = [i for i in agents if origins[i] == destination]
        winner = stationary[0] if stationary else min(agents, key=ranks.get)
        for i in agents:
            if i != winner:
                conflicts.add(tuple(sorted((i, winner))))
                proposed[i] = origins[i]
    for i in range(len(origins)):
        for j in range(i + 1, len(origins)):
            if proposed[i] != origins[i] and proposed[i] == origins[j] and proposed[j] == origins[i]:
                conflicts.add((i, j))
                proposed[i], proposed[j] = origins[i], origins[j]
    # A losing robot keeps its origin; recursively block followers into that cell.
    while True:
        stationary = {origins[i]: i for i in range(len(origins)) if proposed[i] == origins[i]}
        blocked = [i for i in range(len(origins)) if proposed[i] != origins[i] and proposed[i] in stationary]
        if not blocked:
            break
        for i in blocked:
            conflicts.add(tuple(sorted((i, stationary[proposed[i]]))))
            proposed[i] = origins[i]
    if len(set(proposed)) != len(proposed):
        raise AssertionError("Joint resolution broke occupancy invariant")
    return proposed, conflicts
