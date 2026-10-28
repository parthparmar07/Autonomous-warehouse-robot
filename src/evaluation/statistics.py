"""Paired uncertainty with training runs as the independent outer unit."""
import numpy as np

def paired_hierarchical_interval(a,b,key='throughput',samples=2000,seed=0):
    def index(rows):
        result={}
        for row in rows:
            pair=(row['training_seed'],row['scenario_hash'])
            if pair in result:raise ValueError('Duplicate training/scenario pair')
            value=float(row[key])
            if not np.isfinite(value):raise ValueError('Non-finite outcome')
            result[pair]=value
        return result
    left,right=index(a),index(b)
    if left.keys()!=right.keys() or not left:raise ValueError('Algorithms require the same paired runs and scenarios, including failures')
    runs=sorted({k[0] for k in left})
    groups=[np.array([left[k]-right[k] for k in sorted(left) if k[0]==run]) for run in runs]
    point=float(np.mean([g.mean() for g in groups]))
    if len(runs)<2:
        return dict(mean_difference=point,ci95=None,training_runs=len(runs),status='insufficient_independent_training_runs')
    rng=np.random.default_rng(seed)
    draws=[np.mean([rng.choice(groups[i],len(groups[i]),replace=True).mean() for i in rng.integers(len(groups),size=len(groups))]) for _ in range(samples)]
    return dict(mean_difference=point,ci95=np.quantile(draws,[.025,.975]).tolist(),training_runs=len(runs),status='paired_hierarchical_bootstrap')

def wilson_lower(successes,total,z=1.96):
    if total<1:return 0.0
    p=successes/total
    return float((p+z*z/(2*total)-z*np.sqrt(p*(1-p)/total+z*z/(4*total*total)))/(1+z*z/total))

def family_gate(results,minimum=20,lower=.8):
    records=[]
    for family in sorted({r['family'] for r in results}):
        rows=[r for r in results if r['family']==family]
        successes=int(sum(r['completion_rate']>=.95 and r['executed_collisions']==0 and r['avoidable_depletions']==0 for r in rows))
        bound=wilson_lower(successes,len(rows))
        records.append(dict(family=family,count=len(rows),successes=successes,wilson_lower95=bound,passed=bool(len(rows)>=minimum and bound>=lower)))
    return dict(passed=bool(records) and all(r['passed'] for r in records),families=records)
