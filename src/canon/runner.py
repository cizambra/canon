from __future__ import annotations

from pathlib import Path

import yaml

from canon.errors import ConfigError
from canon.metric import CoherenceMetric
from canon.models import CoherenceResult


def run_suite(suite_path: str | Path, metric: CoherenceMetric) -> list[CoherenceResult]:
    p = Path(suite_path)
    if not p.exists():
        raise FileNotFoundError(f"suite file not found: {suite_path}")
    try:
        data = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"suite file {suite_path} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"suite file {suite_path} must be a mapping")
    task = data.get("task", "")
    artifacts = data.get("artifacts")
    # A suite with no artifacts must never be reported as a pass: a coherence
    # gate that succeeds on nothing tells you the system is fine when nothing
    # was checked at all. A missing/typo'd key is a config error, not a pass.
    if artifacts is None or (isinstance(artifacts, list) and not artifacts):
        raise ConfigError("suite has no artifacts")
    if not isinstance(artifacts, list) or not all(isinstance(a, str) for a in artifacts):
        raise ConfigError("suite 'artifacts' must be a list of strings")
    return [metric.score(artifact, task) for artifact in artifacts]
