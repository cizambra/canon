from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from canon.errors import ConfigError

# Never settings: the packaged CDT rubric is fixed so scores stay comparable.
# Ignoring them the way unknown keys are ignored would read as "it worked".
_REFUSED_KEYS = ("rubric", "rubric_path")

_DEFAULTS = {
    "judge_model": "openai:gpt-5.6-luna",
    "threshold": 0.85,
    "constitution_path": "constitution.yaml",
    "baselines_dir": "canon/baselines",
    "tolerance": 0.02,
}


def _load_dotenv(directory: Path) -> None:
    """Load dotenv files from the directory owning the project's config.
    Subdirectory runs must still see the repo-root .env, the same way
    they see the repo-root canon.yaml.
    """
    from dotenv import dotenv_values
    merged: dict[str, str] = {}
    app_env = os.environ.get("APP_ENV")
    # Later files win among files: the shared .env, then the per-environment
    # rung, then the developer's own .env.local.
    names = [".env", *( [f".env.{app_env}"] if app_env else [] ), ".env.local"]
    for name in names:
        f = directory / name
        if f.exists():
            merged.update({k: v for k, v in dotenv_values(f).items() if v is not None})
    for k, v in merged.items():
        os.environ.setdefault(k, v)          # a real exported env var is never overwritten


def _as_float(value: object, name: str) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a number, got {value!r}") from exc


def _read_config_file(path: Path) -> dict:
    """Read one config file's settings, regardless of how it was found.
    Format follows the file suffix (.yml is YAML, not TOML); a TOML file
    always yields its [tool.canon] table, not its top level.
    """
    try:
        if path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(path.read_text()) or {}
        else:
            import tomllib
            data = tomllib.loads(path.read_text())
    except Exception as exc:
        raise ConfigError(f"{path} is not valid {path.suffix.lstrip('.')}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must be a mapping")
    if path.suffix == ".toml":
        table = (data.get("tool", {}) or {}).get("canon", {}) or {}
        if not isinstance(table, dict):
            raise ConfigError(f"{path}: [tool.canon] must be a table")
        # A TOML file with Canon settings at its top level and no [tool.canon]
        # table would silently yield the defaults — say so instead.
        if not table and any(k in data for k in _DEFAULTS):
            raise ConfigError(
                f"{path} carries Canon settings at top level but has no "
                f"[tool.canon] table; nest them under [tool.canon]")
        data = table
    for key in _REFUSED_KEYS:
        if key in data:
            raise ConfigError(
                f"{path}: {key!r} is not a Canon setting — the packaged CDT rubric "
                f"is fixed and versioned so scores stay comparable across runs "
                f"and projects; your constitution is what makes a run yours")
    return data


def _discover(start: Path) -> tuple[dict, Path | None]:
    """Walk up for the project config; return it with the FILE it came from."""
    cur = start.resolve()
    for d in [cur, *cur.parents]:
        cy = d / "canon.yaml"
        if cy.exists():
            return _read_config_file(cy), cy
        py = d / "pyproject.toml"
        if py.exists():
            canon_cfg = _read_config_file(py)
            if canon_cfg:              # only a real [tool.canon] stops the walk
                return canon_cfg, py
            # bare pyproject without [tool.canon]: keep walking up
    return {}, None


@dataclass(frozen=True)
class Settings:
    judge_model: str
    threshold: float
    constitution_path: str
    baselines_dir: str
    tolerance: float = 0.02
    # Project directory a discovered config was found in. A relative path in
    # that config is relative to the project, not the current directory —
    # otherwise running from a subdirectory would miss the constitution and
    # scatter baselines wherever you happened to stand.
    root: Path | None = None

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else (self.root or Path.cwd()) / p

    @property
    def constitution_file(self) -> Path:
        return self._resolve(self.constitution_path)

    @property
    def baselines_directory(self) -> Path:
        return self._resolve(self.baselines_dir)

    @classmethod
    def load(cls, start: Path | None = None,
             config_path: str | Path | None = None) -> "Settings":
        """Load settings, optionally from an explicit config file.
        An explicit path replaces discovery outright: a missing file is an
        error, not a silent fall back to defaults. It doesn't relocate the
        project — relative paths still resolve from the current directory."""
        start = Path(start or Path.cwd())
        if config_path is not None:
            cfg_file = Path(config_path)
            if not cfg_file.exists():
                raise ConfigError(f"config file not found: {config_path}")
            discovered, source, root = _read_config_file(cfg_file), cfg_file.resolve(), None
        else:
            discovered, source = _discover(start)
            root = source.parent if source else None
        _load_dotenv(source.parent if source else start.resolve())
        cfg = {**_DEFAULTS, **discovered}
        if os.environ.get("CANON_JUDGE_MODEL"):
            cfg["judge_model"] = os.environ["CANON_JUDGE_MODEL"]
        return cls(judge_model=str(cfg["judge_model"]),
                   threshold=_as_float(cfg["threshold"], "threshold"),
                   constitution_path=str(cfg["constitution_path"]),
                   baselines_dir=str(cfg["baselines_dir"]),
                   tolerance=_as_float(cfg["tolerance"], "tolerance"),
                   root=root)


def default_judge_model() -> str:
    return Settings.load().judge_model


def discover_config_file(start: Path | None = None) -> Path | None:
    """Config file the project's settings were discovered in, if any.
    Shared with `set-judge` so a write targets the same file `Settings.load()`
    would read — writing elsewhere would fork a second config.
    """
    _, source = _discover(Path(start or Path.cwd()))
    return source


def discover_root(start: Path | None = None) -> Path | None:
    """The directory the project's config was discovered from, if any."""
    source = discover_config_file(start)
    return source.parent if source else None


def set_judge(provider: str, model: str, path: Path) -> None:
    data = {}
    if path.exists():
        data = yaml.safe_load(path.read_text()) or {}
    data["judge_model"] = f"{provider}:{model}"
    path.write_text(yaml.safe_dump(data))
