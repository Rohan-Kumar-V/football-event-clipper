from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.json"


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def config_value(path: str, default: Any = None) -> Any:
    current: Any = load_config()
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def config_path(path: str, default: str | Path) -> Path:
    value = config_value(path, str(default))
    candidate = Path(str(value))
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate
