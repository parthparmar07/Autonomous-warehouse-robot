from src.training.runner import parser,training_case,TRAINING_CASES

def test_training_materializes_skill_cases_without_reserved_ood_compositions():
    args=parser().parse_args([])
    args.stage='C3';args.num_agents=5
    seen=set();roles={}
    for index in range(4,154,5):
        env,manifest=training_case(args,100000+index,index)
        case=manifest['case_id'];seen.add(case)
        roles.setdefault(case,set()).add(manifest['role_index'])
        assert manifest['materialized']['split']=='train'
        env.assert_invariants()
    assert seen=={'HND06','HND_TRANSFER','ROB04'}
    assert all(len(v)==10 for v in roles.values())
    assert all('COO06' not in cases and 'ROB05' not in cases for cases in TRAINING_CASES.values())
