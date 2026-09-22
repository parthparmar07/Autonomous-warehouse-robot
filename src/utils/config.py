from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent
