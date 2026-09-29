from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.environment.layout import default_layout
from src.environment.task_generator import generate_tasks
from src.environment.warehouse_env import WarehouseEnv
from src.utils.hashing import sha256_digest


def generate_evaluation_manifest(num_agents: int, seed: int, grid_size: tuple[int, int] = (20, 20), num_tasks: int = 20) -> dict[str, Any]:
    """Create a deterministic evaluation manifest from a seeded warehouse environment."""
    env = WarehouseEnv(grid_size=grid_size, num_agents=num_agents, seed=seed, **default_layout(grid_size))
    env.reset(seed=seed)
    tasks = generate_tasks(env, num_tasks=num_tasks, seed=seed)
    manifest = {
        "num_agents": num_agents,
        "seed": seed,
        "grid_size": list(grid_size),
        "num_tasks": num_tasks,
        "tasks": tasks,
        "robot_initial_state": [{
            "id": robot["id"],
            "x": robot["x"],
            "y": robot["y"],
            "orientation": robot["orientation"],
        } for robot in env.robots],
        "manifest_hash": "",
    }
    manifest["manifest_hash"] = sha256_digest(manifest)
    return manifest


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return sha256_digest(manifest)


def generate_manifest_set(output_dir: str | Path = "evaluation/manifests") -> list[str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_paths: list[str] = []
    for num_agents in (5, 10, 20):
        for seed in (100, 101, 102):
            manifest = generate_evaluation_manifest(num_agents=num_agents, seed=seed)
            file_name = f"eval_N{num_agents}_seed{seed}.json"
            target = output / file_name
            write_manifest(target, manifest)
            manifest_paths.append(str(target))
    return manifest_paths
