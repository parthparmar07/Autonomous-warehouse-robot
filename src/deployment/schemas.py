from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, model_validator


class ObservationRequest(BaseModel):
    """Input contract for a local observation or fleet observation list."""

    observations: List[List[float]] = Field(default_factory=list)
    actions: List[int] | None = None

    @model_validator(mode="after")
    def validate_observations(self):
        if not self.observations:
            raise ValueError("observations must not be empty")
        for obs in self.observations:
            if len(obs) != 28:
                raise ValueError("each observation must contain exactly 28 features")
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
