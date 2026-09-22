from __future__ import annotations

from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.deployment.inference import inference_source, predict_actions
from src.deployment.schemas import ObservationRequest, PredictResponse, StepRequest
from src.environment.task_generator import generate_tasks
from src.environment.warehouse_env import WarehouseEnv

app = FastAPI(title="Warehouse Fleet API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

demo_env = WarehouseEnv(grid_size=(20, 20), num_agents=5, seed=42)
demo_env.reset(seed=42)
demo_env.task_queue = generate_tasks(demo_env, num_tasks=10, seed=42)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: ObservationRequest) -> PredictResponse:
    observations: List[List[float]] = payload.observations
    masks = None
    if payload.actions is not None:
        masks = [[bool(action == idx) for idx in range(5)] for action in payload.actions]
    actions = predict_actions(observations, masks)
    return PredictResponse(actions=actions, message=inference_source())


@app.get("/state")
def state() -> dict:
    return demo_env.get_global_state()


@app.post("/step")
def step(payload: StepRequest) -> dict:
    if len(payload.actions) != demo_env.num_agents:
        raise ValueError(f"Expected {demo_env.num_agents} actions")
    return demo_env.step(payload.actions)
