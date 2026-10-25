"""Configuration loading and path resolution."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("YAML configuration must contain a mapping at its root")
    resolved = deepcopy(config)
    project_root = config_path.parent.parent
    resolved["_config_path"] = str(config_path)
    resolved["_project_root"] = str(project_root)
    for key in ("train_csv", "test_csv", "output_dir"):
        raw = Path(resolved["paths"][key])
        resolved["paths"][key] = str(raw if raw.is_absolute() else project_root / raw)
    return resolved


def output_path(config: dict[str, Any], filename: str) -> Path:
    directory = Path(config["paths"]["output_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename
