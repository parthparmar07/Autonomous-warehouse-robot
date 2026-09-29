from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, model_validator

ALGORITHM_NAMES = ("nearest_task", "dqn", "iql", "qmix")


class ObservationRequest(BaseModel):
    """Input contract for raw local-observation batches (programmatic / testing use).

    ``masks`` is optional per-observation action masks (see ``get_action_mask``); without
    one, every action is reported legal for that observation -- callers that need the real
    safety guarantee should supply masks computed from a live WarehouseEnv, or drive the
    simulation through /reset + /tick instead, which always uses the true per-robot mask.
    """

    observations: List[List[float]] = Field(default_factory=list)
    masks: List[List[bool]] | None = None

    @model_validator(mode="after")
    def validate_observations(self):
        if not self.observations:
            raise ValueError("observations must not be empty")
        for obs in self.observations:
            if len(obs) != 30:
                raise ValueError("each observation must contain exactly 30 features")
        if self.masks is not None and len(self.masks) != len(self.observations):
            raise ValueError("masks must have one entry per observation")
        return self


class PredictResponse(BaseModel):
    actions: List[int]
    message: str = "ok"


class StepRequest(BaseModel):
    actions: List[int]

    @model_validator(mode="after")
    def validate_actions(self):
        if not self.actions:
            raise ValueError("actions must not be empty")
        if any(action not in range(5) for action in self.actions):
            raise ValueError("actions must be between 0 and 4")
        return self


class ResetRequest(BaseModel):
    algorithm: str = "nearest_task"
    num_agents: int = 5
    num_tasks: int = 10
    seed: int | None = None

    @model_validator(mode="after")
    def validate_algorithm(self):
        if self.algorithm not in ALGORITHM_NAMES:
            raise ValueError(f"algorithm must be one of {ALGORITHM_NAMES}")
        if not (1 <= self.num_agents <= 40):
            raise ValueError("num_agents must be between 1 and 40")
        return self


class AlgorithmInfo(BaseModel):
    name: str
    trained: bool
    description: str
