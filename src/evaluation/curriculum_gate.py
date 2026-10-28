"""Development promotion: generic retention plus named skill prerequisites."""
from src.evaluation.curriculum import eligible_cases
from src.evaluation.policy_cases import materialize,skill_outcome
from src.evaluation.statistics import family_gate,wilson_lower
from src.training.runner import collect_episode,evaluate
from src.baselines.nearest_task import NearestTaskDispatcher


def prerequisite_gate(rows,num_agents,minimum=20):
    families=[]
    for case in sorted({r['case'] for r in rows}):
        cases=[r for r in rows if r['case']==case]
        successes=sum(bool(r['operational_prerequisite_passed']) for r in cases)
        roles={r['role']%(num_agents*4) for r in cases}
        bound=wilson_lower(successes,len(cases))
        families.append(dict(case=case,count=len(cases),successes=successes,
            roles_covered=len(roles),roles_required=num_agents*4,wilson_lower95=bound,
            passed=bool(len(cases)>=minimum and len(roles)==num_agents*4 and bound>=.8)))
    return dict(passed=all(r['passed'] for r in families),families=families)


def evaluate_curriculum_gate(model,stages,num_agents,generic_episodes=20,skill_episodes=None,horizon=400,profile='robust',runtime_overrides=None,stage_profiles=None):
    stages=list(stages)
    stage_profiles=dict(stage_profiles or {})
    if profile not in ('core','robust') or any(value not in ('core','robust') for value in stage_profiles.values()):
        raise ValueError('Unknown environment profile')
    if generic_episodes<20:raise ValueError('At least 20 generic development cases required')
    required=max(20,num_agents*4)
    skill_episodes=required if skill_episodes is None else skill_episodes
    if skill_episodes<required:raise ValueError('Skill cases require at least 20 episodes and every worker/heading role')
    cases=sorted({case for stage in stages for case in eligible_cases(stage,num_agents)})
    generic=[];skills=[]
    epsilon=getattr(model,'epsilon',None)
    try:
        if epsilon is not None:model.epsilon=0
        for stage in stages:
            stage_profile=stage_profiles.get(stage,profile)
            generic.extend(dict(r,family=stage,profile=stage_profile) for r in evaluate(model,
                list(range(301000,301000+generic_episodes)),num_agents,stage,horizon=horizon,
                profile=stage_profile,runtime_overrides=runtime_overrides))
        for case in cases:
            for role in range(skill_episodes):
                env,manifest=materialize(case,304000+role,num_agents,horizon,role)
                if runtime_overrides:
                    from dataclasses import replace
                    from src.evaluation.manifests import capture_manifest
                    env.config=replace(env.config,**runtime_overrides)
                    manifest=capture_manifest(env,304000+role,'development',case)
                policy=NearestTaskDispatcher(env) if model is None else model
                _,metrics=collect_episode(env,policy,dict(seed=manifest['seed'],scenario_hash=manifest['manifest_hash']))
                skills.append(dict(case=case,role=role,**metrics,**skill_outcome(case,env,metrics)))
    finally:
        if epsilon is not None:model.epsilon=epsilon
    generic_gate=family_gate(generic)
    skill_gate=prerequisite_gate(skills,num_agents)
    complete=set(cases)=={r['case'] for r in skills}
    return dict(passed=bool(generic_gate['passed'] and skill_gate['passed'] and complete),
        stages=stages,stage_profiles={stage:stage_profiles.get(stage,profile) for stage in stages},required_cases=cases,generic_gate=generic_gate,skill_gate=skill_gate,
        generic_results=generic,skill_results=skills,
        scope='Development prerequisites and retention; independent trained seeds and frozen final tests remain required')
