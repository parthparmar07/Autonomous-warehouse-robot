from __future__ import annotations

from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.baselines.nearest_task import NearestTaskDispatcher
from src.deployment.inference import inference_source, predict_actions
from src.deployment.policies import describe_algorithms, load_policy
from src.deployment.schemas import ObservationRequest, PredictResponse, ResetRequest, StepRequest
from src.environment.layout import default_layout
from src.environment.task_generator import generate_tasks
from src.environment.warehouse_env import WarehouseEnv

app = FastAPI(title="Warehouse Fleet API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_GRID_SIZE = (20, 20)
_DEFAULT_SESSION = "default"


class _Session:
    """One live demo's mutable state: one environment, one active policy, one task
    dispatcher used purely for task *assignment* (which idle robot gets which task) --
    movement is decided by whichever policy /reset selected, exactly mirroring how the
    training scripts separate assignment from movement.

    Multiple sessions (keyed by an arbitrary client-chosen id) can run side by side --
    this is what lets the frontend's "compare algorithms" view tick four independent
    algorithms in parallel instead of sharing one global environment.
    """

    def __init__(self) -> None:
        self.policy: Any = None
        self.dispatcher: NearestTaskDispatcher | None = None
        self.env: WarehouseEnv | None = None
        self._task_batch = 0
        self.reset("nearest_task", 5, 10, 42)

    def reset(self, algorithm: str, num_agents: int, num_tasks: int, seed: int | None) -> Dict[str, Any]:
        self.algorithm = algorithm
        self.num_agents = num_agents
        self.num_tasks = num_tasks
        self.seed = seed if seed is not None else getattr(self, "seed", 42)
        self._task_batch = 0
        self.env = WarehouseEnv(grid_size=_GRID_SIZE, num_agents=num_agents, seed=self.seed, **default_layout(_GRID_SIZE))
        self.env.reset(seed=self.seed)
        self.env.task_queue = generate_tasks(self.env, num_tasks=num_tasks, seed=self.seed)
        self.dispatcher = NearestTaskDispatcher(self.env)
        self.dispatcher.assign_tasks(self.env.task_queue)
        self.policy = load_policy(algorithm, num_agents)
        return self.state()

    def tick(self) -> Dict[str, Any]:
        env = self.env
        assert env is not None and self.dispatcher is not None
        self.dispatcher.assign_pending_tasks(env.task_queue)

        if self.algorithm == "nearest_task":
            actions: List[int] = self.dispatcher.act(env)
        else:
            observations = [env.get_local_observation(agent_id) for agent_id in range(env.num_agents)]
            masks = [env.get_action_mask(agent_id) for agent_id in range(env.num_agents)]
            actions = self.policy.act(observations, masks)

        state = env.step(actions)
        state["actions"] = actions
        state["algorithm"] = self.algorithm

        # Keep the live demo running: refill the task queue instead of stalling once
        # every task is delivered.
        if not env.task_queue:
            self._task_batch += 1
            env.task_queue = generate_tasks(env, num_tasks=self.num_tasks, seed=self.seed + self._task_batch)
            self.dispatcher.assign_tasks(env.task_queue)
            state["task_queue"] = env.task_queue

        return state

    def state(self) -> Dict[str, Any]:
        assert self.env is not None
        state = self.env.get_global_state()
        state["algorithm"] = self.algorithm
        return state


_sessions: Dict[str, _Session] = {}


def _get_session(session_id: str) -> _Session:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"No session '{session_id}'. Call /reset first.")
    return session


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/algorithms")
def algorithms() -> list[dict]:
    return describe_algorithms()


@app.post("/predict", response_model=PredictResponse)
def predict(payload: ObservationRequest) -> PredictResponse:
    actions = predict_actions(payload.observations, payload.masks)
    return PredictResponse(actions=actions, message=inference_source())


@app.get("/state")
def state(session_id: str = _DEFAULT_SESSION) -> dict:
    return _get_session(session_id).state()


@app.post("/reset")
def reset(payload: ResetRequest, session_id: str = _DEFAULT_SESSION) -> dict:
    session = _sessions.get(session_id)
    if session is None:
        session = _Session.__new__(_Session)
        session.policy = None
        session.dispatcher = None
        session.env = None
        session._task_batch = 0
        _sessions[session_id] = session
    try:
        result = session.reset(payload.algorithm, payload.num_agents, payload.num_tasks, payload.seed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    result["session_id"] = session_id
    return result


@app.post("/tick")
def tick(session_id: str = _DEFAULT_SESSION) -> dict:
    """Advance one session's live demo by one step using its selected algorithm.

    Movement actions always pass through the real per-robot action mask (ADR-003):
    corridor/shelf/boundary/robot-conflict cells are excluded before any policy's argmax,
    identically to how the checkpoints were trained.
    """
    return _get_session(session_id).tick()


@app.post("/step")
def step(payload: StepRequest, session_id: str = _DEFAULT_SESSION) -> dict:
    """Advance a session's env with explicit actions, bypassing the active policy.

    Kept for manual control / debugging; prefer /tick to drive the live demo, since this
    endpoint does not consult the active algorithm or reassign tasks.
    """
    session = _get_session(session_id)
    assert session.env is not None
    if len(payload.actions) != session.env.num_agents:
        raise HTTPException(status_code=400, detail=f"Expected {session.env.num_agents} actions")
    return session.env.step(payload.actions)


@app.delete("/session")
def delete_session(session_id: str) -> dict:
    _sessions.pop(session_id, None)
    return {"deleted": session_id}
