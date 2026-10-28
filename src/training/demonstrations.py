"""Policy-independent training demonstrations with explicit role coverage.

The larger data budget is deliberate assistance and must be reported separately
from online environment steps. No development or final seeds are selected here.
"""
from src.evaluation.curriculum import STAGES, eligible_cases


def demonstration_schedule(args):
    """Enumerate each prerequisite family for every worker/heading combination."""
    full=getattr(args,'full_demonstration_coverage',True)
    stages = STAGES[:STAGES.index(args.stage) + 1] if full else (args.stage,)
    cases = sorted({case for stage in stages for case in eligible_cases(stage, args.num_agents)})
    families = [('generic', stage) for stage in stages] + [('skill', case) for case in cases]
    if not full:
        # Half ordinary current-stage episodes, half named skills; diagonal
        # rotation spreads worker/headings without silently increasing the budget.
        if not cases:return [('generic',args.stage,role) for role in range(args.num_agents*4)]
        schedule=[]
        for offset in range(len(cases)):
            for role in range(args.num_agents*4):
                schedule.extend([('generic',args.stage,role),('skill',cases[(role+offset)%len(cases)],role)])
        return schedule
    # Interleave families rather than putting late skills at the end of replay.
    return [(kind, family, role) for role in range(args.num_agents * 4)
            for kind, family in families]


def demonstration_count(args, requested):
    if requested < 0:
        raise ValueError('Demonstration count cannot be negative')
    if not requested or not getattr(args, 'balanced_demonstrations', False):
        return requested
    if args.tiny or args.scaling != 'fixed_map':
        raise ValueError('Balanced demonstrations require the full fixed map')
    return max(requested,len(demonstration_schedule(args))) if getattr(args,'full_demonstration_coverage',True) else requested


def demonstration_case(args, seed, index):
    if not getattr(args, 'balanced_demonstrations', False):
        from src.training.runner import training_case
        return training_case(args, seed, index)
    if not 100000 <= seed < 200000:
        raise ValueError('Demonstrations require a training-partition seed')
    from dataclasses import replace
    from src.training.runner import runtime_settings
    from src.evaluation.scenarios import make_scenario
    from src.evaluation.policy_cases import materialize
    from src.evaluation.manifests import capture_manifest
    from src.utils.hashing import sha256_digest
    schedule = demonstration_schedule(args)
    kind, family, role = schedule[index % len(schedule)]
    if kind == 'generic':
        env, manifest = make_scenario(seed, args.num_agents, family, 'train',
            args.horizon, role_index=role, scaling=args.scaling,
            profile=getattr(args, 'stage_profiles', {}).get(family, args.profile),
            runtime_overrides=runtime_settings(args))
        manifest.update(case_id=family, role_index=role,
                        demonstration_kind=kind, demonstration_stage=family)
        return env, manifest
    env, _ = materialize(family, seed, args.num_agents, args.horizon, role)
    env.config = replace(env.config, **runtime_settings(args))
    sealed = capture_manifest(env, seed, 'train', family)
    sealed['split'] = 'train'
    sealed['manifest_hash'] = ''
    sealed['manifest_hash'] = sha256_digest(sealed)
    return env, dict(seed=seed, scenario_hash=sealed['manifest_hash'], case_id=family,
                     role_index=role, materialized=sealed, demonstration_kind=kind)
