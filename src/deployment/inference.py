from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from src.algorithms.dqn.agent import DQNAgent
from src.safety.action_mask import apply_action_mask


_MODEL: DQNAgent | None = None
_MODEL_SOURCE: str = "heuristic"


def _load_model() -> DQNAgent | None:
    global _MODEL, _MODEL_SOURCE
    if _MODEL is not None:
        return _MODEL

    checkpoint = Path(os.getenv("WAREHOUSE_MODEL_CHECKPOINT", "artifacts/dqn_policy.pt"))
    if not checkpoint.exists():
        return None
    try:
        model = DQNAgent(epsilon_start=0.0, epsilon_end=0.0)
        state_dict = torch.load(checkpoint, map_location=model.device, weights_only=True)
        model.online_network.load_state_dict(state_dict)
        model.online_network.eval()
        _MODEL = model
        _MODEL_SOURCE = str(checkpoint)
        return _MODEL
    except (OSError, RuntimeError, ValueError, TypeError):
        return None


def predict_actions(observations: Sequence[Sequence[float]], masks: Sequence[Sequence[bool]] | None = None) -> list[int]:
    """Infer masked actions using the trained DQN when an artifact is available."""
    model = _load_model()
    predictions: list[int] = []
    for index, obs in enumerate(observations):
        observation = np.asarray(obs, dtype=np.float32)
        if observation.shape[0] != 30:
            raise ValueError("local observation too short for action prediction")
        safe_mask = np.ones(5, dtype=bool) if masks is None or masks[index] is None else np.asarray(masks[index], dtype=bool)
        if model is not None:
            predictions.append(model.act(observation, safe_mask))
            continue
        raw_q = np.tanh(observation[:5])
        predictions.append(int(np.argmax(apply_action_mask(raw_q, safe_mask))))
    return predictions


def inference_source() -> str:
    """Return the active artifact path or the fallback inference mode."""
    _load_model()
    return _MODEL_SOURCE
