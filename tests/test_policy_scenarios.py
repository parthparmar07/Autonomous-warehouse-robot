import pytest
from src.evaluation.policy_cases import POLICY_CASES,materialize

@pytest.mark.parametrize('case',POLICY_CASES)
def test_policy_skill_scenarios_materialize_and_preserve_physics(case):
    env,manifest=materialize(case,303001,5,120,role=1)
    assert manifest['stage']==case and len(env.get_training_state())==1818
    for _ in range(45):env.step([4]*5);env.assert_invariants()
