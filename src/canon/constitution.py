from __future__ import annotations

import json
from pathlib import Path

import yaml

from canon.errors import ConfigError
from canon.models import Constitution


def constitution_from_dict(data: dict) -> Constitution:
    mission = data.get("mission")
    principles = data.get("principles")
    if not isinstance(mission, str) or not mission.strip():
        raise ConfigError("constitution requires a non-empty 'mission'")
    if (
        not isinstance(principles, list)
        or not principles
        or not all(isinstance(p, str) and p.strip() for p in principles)
    ):
        raise ConfigError("constitution requires a non-empty list of string 'principles'")
    version = data.get("version")
    return Constitution(
        mission=mission.strip(),
        principles=tuple(p.strip() for p in principles),
        version=str(version) if version is not None else None,
    )


def constitution_from_file(path: str | Path) -> Constitution:
    text = Path(path).read_text()
    try:
        data = yaml.safe_load(text)  # YAML superset also parses JSON
    except yaml.YAMLError:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ConfigError(f"constitution file {path} must be a mapping")
    return constitution_from_dict(data)
