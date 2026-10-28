from dataclasses import dataclass
@dataclass
class MetricSummary:
    completion_rate:float=0
    average_delivery_time:float|None=None
    collision_rate:float=0
    deadlocks:int=0
    throughput:float=0
    safety_violations:int=0
    episode_reward:float=0
    inference_latency_ms:float|None=None
    safety_overrides:int=0

def summarize_metrics(results):
    def mean(key):
        values=[r[key] for r in results if r.get(key) is not None]
        return sum(values)/len(values) if values else None
    if not results:return MetricSummary()
    return MetricSummary(completion_rate=mean('completion_rate'),average_delivery_time=mean('delivery_time'),
        collision_rate=mean('conflicts_per_robot_tick'),deadlocks=sum(r.get('deadlocks',0) for r in results),
        throughput=mean('throughput'),safety_violations=sum(r.get('safety_violations',0) for r in results),
        episode_reward=mean('episode_reward'),inference_latency_ms=mean('inference_latency_ms'),
        safety_overrides=sum(r.get('safety_overrides',0) for r in results))
