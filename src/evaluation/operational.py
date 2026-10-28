from collections import Counter
import numpy as np


def episode_metrics(env, manifest, latencies):
    completed=[t for t in env.finished_tasks if t['status']=='delivered']
    released=[t for t in env._ledger.values() if t['creation_timestep']<=env.tick]
    durations=[t['completed_tick']-t['creation_timestep'] for t in completed]
    unfinished=[dict(task_id=t['task_id'],age=env.tick-t['creation_timestep'],right_censored=True) for t in env.task_queue]
    faults=[e for e in env.event_log if e['type']=='fault']
    recoveries=[]
    for fault in faults:
        if fault.get('task_id') is None:
            continue
        later=next((e for e in env.event_log if e['tick']>=fault['tick'] and e['type']=='delivered' and e.get('task_id')==fault['task_id'] and e.get('robot_id')!=fault.get('robot_id')),None)
        recoveries.append(dict(task_id=fault['task_id'],fault_tick=fault['tick'],latency=None if later is None else later['tick']-fault['tick'],right_censored=later is None))
    n=len(released)
    queue_waits=[e['queue_wait_ticks'] for e in env.event_log if e['type']=='charge_started']
    result = dict(seed=manifest['seed'],scenario_hash=manifest['scenario_hash'],num_agents=env.num_agents,
        completed_tasks=len(completed),released_tasks=n,completion_rate=len(completed)/max(1,n),
        episode_reward=env.episode_metrics['reward'],throughput=len(completed)/max(1,env.tick),
        delivery_time=float(np.mean(durations)) if durations else None,unfinished_orders=unfinished,
        collision_count=env.episode_metrics['collision_count'],executed_collisions=env.episode_metrics['executed_collisions'],
        conflicts_per_robot_tick=env.episode_metrics['collision_count']/max(1,env.tick*env.num_agents),
        deadlocks=env.episode_metrics['deadlocks'],safety_violations=env.episode_metrics['executed_safety_violations'],safety_overrides=env.episode_metrics['safety_overrides'],
        energy_used=env.episode_metrics['energy_used'],energy_per_delivery=env.episode_metrics['energy_used']/len(completed) if completed else None,
        avoidable_depletions=env.episode_metrics['avoidable_depletions'],backlog_ticks=env.episode_metrics['backlog_ticks'],
        charge_queue_wait_p95=float(np.percentile(queue_waits,95)) if queue_waits else None,
        charge_queue_ticks=env.episode_metrics['charge_queue_ticks'],charger_utilization=env.episode_metrics['charging_ticks']/max(1,env.tick*len(env.chargers)),
        idle_with_assignment_ticks=env.episode_metrics['idle_with_assignment_ticks'],
        charge_locked_infeasible_ticks=env.episode_metrics.get('charge_locked_infeasible_ticks',0),
        completed_charge_cycles=sum(e['type']=='charge_completed' for e in env.event_log),
        weighted_tardiness=env.episode_metrics['tardiness_ticks'],recovery=recoveries,
        per_robot_deliveries=dict(Counter(e['robot_id'] for e in env.event_log if e['type']=='delivered')),
        inference_latency_ms=float(np.mean(latencies)) if latencies else None,
        inference_latency_p95_ms=float(np.percentile(latencies,95)) if latencies else None,steps=env.tick)

    if hasattr(env, 'inventory'):
        result.update(workload_version=env.workload_version,arrival_interval=manifest.get('arrival_interval'),
            stored_packages=sum(t.get('flow')=='inbound' for t in completed),
            shipped_packages=sum(t.get('flow')=='outbound' for t in completed),
            incoming_packages=sum(t.get('flow')=='inbound' for t in released),
            shipping_requests=sum(t.get('flow')=='outbound' for t in released),
            waiting_for_stock=sum(t['status']=='waiting_stock' for t in env.task_queue),
            waiting_for_capacity=sum(t['status']=='waiting_capacity' for t in env.task_queue),
            inventory_states=dict(Counter(s['state'] for s in env.inventory.values())))
    return result
