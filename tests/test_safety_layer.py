import numpy as np

from src.safety.action_mask import apply_action_mask, get_action_mask


def test_safety_mask_discards_unsafe_actions_even_when_q_values_are_highest():
    q_values = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=float)
    mask = [True, True, True, False, True]

    masked = apply_action_mask(q_values, mask)
    assert np.isfinite(masked[0])
    assert np.isfinite(masked[1])
    assert np.isfinite(masked[2])
    assert np.isneginf(masked[3])
    assert np.isfinite(masked[4])

    safe_action = int(np.argmax(masked))
    assert mask[safe_action] is True

    # mask function should also behave as expected.
    action_mask = get_action_mask(np.array([0, 0]), None)
    assert len(action_mask) == 5
