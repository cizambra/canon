from pathlib import Path

import pytest

from canon.config import Settings
from canon.errors import ConfigError


def test_defaults(monkeypatch, tmp_path):
    monkeypatch.delenv("CANON_JUDGE_MODEL", raising=False)
    s = Settings.load(start=tmp_path)
    assert s.judge_model == "openai:gpt-5.6-luna"
    assert s.threshold == 0.85 and s.baselines_dir.endswith("canon/baselines")


def test_canon_yaml_and_env_override(monkeypatch, tmp_path):
    (tmp_path / "canon.yaml").write_text("judge_model: together:deepseek\nthreshold: 0.7\n")
    s = Settings.load(start=tmp_path)
    assert s.judge_model == "together:deepseek" and s.threshold == 0.7
    monkeypatch.setenv("CANON_JUDGE_MODEL", "anthropic:claude")
    assert Settings.load(start=tmp_path).judge_model == "anthropic:claude"


def test_pyproject_tool_canon_discovery(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.canon]\njudge_model = "openai:x"\nthreshold = 0.6\n')
    s = Settings.load(start=tmp_path)
    assert s.judge_model == "openai:x" and s.threshold == 0.6


def test_walkup_skips_bare_pyproject_to_find_canon_yaml(tmp_path):
    (tmp_path / "canon.yaml").write_text("judge_model: together:deep\n")
    sub = tmp_path / "sub"; sub.mkdir()
    (sub / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    s = Settings.load(start=sub)
    assert s.judge_model == "together:deep"


def test_dotenv_local_overrides_env_but_not_real_env(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("CANON_JUDGE_MODEL=from_env\n")
    (tmp_path / ".env.local").write_text("CANON_JUDGE_MODEL=from_local\n")
    monkeypatch.delenv("CANON_JUDGE_MODEL", raising=False)
    assert Settings.load(start=tmp_path).judge_model == "from_local"
    monkeypatch.setenv("CANON_JUDGE_MODEL", "from_shell")
    assert Settings.load(start=tmp_path).judge_model == "from_shell"


def test_set_judge_writes_canon_yaml(tmp_path):
    from canon.config import set_judge
    p = tmp_path / "canon.yaml"
    set_judge("together", "deepseek", p)
    assert "together:deepseek" in p.read_text()


def test_malformed_canon_yaml_raises_config_error(tmp_path):
    import pytest
    from canon.errors import ConfigError
    (tmp_path / "canon.yaml").write_text("judge_model: [unclosed\nthreshold: 0.7\n")
    with pytest.raises(ConfigError, match="canon.yaml"):
        Settings.load(start=tmp_path)


def test_non_numeric_threshold_raises_config_error(tmp_path):
    import pytest
    from canon.errors import ConfigError
    (tmp_path / "canon.yaml").write_text("threshold: not-a-number\n")
    with pytest.raises(ConfigError, match="threshold"):
        Settings.load(start=tmp_path)


def test_dotenv_is_read_from_the_config_root_not_just_cwd(tmp_path, monkeypatch):
    (tmp_path / "canon.yaml").write_text("threshold: 0.7\n")
    (tmp_path / ".env").write_text("CANON_JUDGE_MODEL=from_root_env\n")
    sub = tmp_path / "sub" / "deeper"
    sub.mkdir(parents=True)
    monkeypatch.delenv("CANON_JUDGE_MODEL", raising=False)
    assert Settings.load(start=sub).judge_model == "from_root_env"


def test_tolerance_defaults_and_is_configurable(tmp_path):
    assert Settings.load(start=tmp_path).tolerance == 0.02
    (tmp_path / "canon.yaml").write_text("tolerance: 0.05\n")
    assert Settings.load(start=tmp_path).tolerance == 0.05


def test_tolerance_from_pyproject_tool_canon(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[tool.canon]\ntolerance = 0.07\n')
    assert Settings.load(start=tmp_path).tolerance == 0.07


def test_dotenv_app_env_rung_sits_between_env_and_local(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("CANON_JUDGE_MODEL=from_env\n")
    (tmp_path / ".env.test").write_text("CANON_JUDGE_MODEL=from_app_env\n")
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.delenv("CANON_JUDGE_MODEL", raising=False)
    assert Settings.load(start=tmp_path).judge_model == "from_app_env"

    (tmp_path / ".env.local").write_text("CANON_JUDGE_MODEL=from_local\n")
    monkeypatch.delenv("CANON_JUDGE_MODEL", raising=False)
    assert Settings.load(start=tmp_path).judge_model == "from_local"


def test_explicit_config_path_wins_over_discovery(tmp_path):
    (tmp_path / "canon.yaml").write_text("threshold: 0.7\n")
    other = tmp_path / "other" / "canon.yaml"
    other.parent.mkdir()
    other.write_text("threshold: 0.42\n")
    assert Settings.load(start=tmp_path, config_path=other).threshold == 0.42


def test_explicit_config_path_must_exist(tmp_path):
    import pytest
    from canon.errors import ConfigError
    with pytest.raises(ConfigError, match="config file not found"):
        Settings.load(start=tmp_path, config_path=tmp_path / "nope.yaml")


def test_relative_paths_resolve_against_the_config_root(tmp_path):
    (tmp_path / "canon.yaml").write_text(
        "constitution_path: shared/constitution.yaml\nbaselines_dir: canon/baselines\n")
    sub = tmp_path / "sub" / "deeper"
    sub.mkdir(parents=True)
    s = Settings.load(start=sub)
    assert s.constitution_file == tmp_path / "shared" / "constitution.yaml"
    assert s.baselines_directory == tmp_path / "canon" / "baselines"


def test_absolute_paths_in_the_config_are_left_alone(tmp_path):
    (tmp_path / "canon.yaml").write_text(
        f"constitution_path: /etc/canon/constitution.yaml\nbaselines_dir: /var/canon\n")
    s = Settings.load(start=tmp_path)
    assert s.constitution_file == Path("/etc/canon/constitution.yaml")
    assert s.baselines_directory == Path("/var/canon")


def test_with_no_config_anywhere_paths_stay_relative_to_the_current_directory(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = Settings.load(start=tmp_path)
    assert s.baselines_directory == tmp_path / "canon" / "baselines"


def test_explicit_pyproject_config_reads_the_tool_canon_table(tmp_path, monkeypatch):
    """`--config pyproject.toml` must mean the same thing discovery means by it:
    the [tool.canon] table. Reading the file's top level instead silently
    yields defaults — the settings are there, just never seen."""
    monkeypatch.delenv("CANON_JUDGE_MODEL", raising=False)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\n\n[tool.canon]\nthreshold = 0.42\n'
        'judge_model = "together:deep"\n')
    s = Settings.load(start=tmp_path, config_path=tmp_path / "pyproject.toml")
    assert s.threshold == 0.42 and s.judge_model == "together:deep"


def test_explicit_yml_config_is_parsed_as_yaml(tmp_path, monkeypatch):
    """.yml is the same format as .yaml; parsing it as TOML just errors out."""
    monkeypatch.delenv("CANON_JUDGE_MODEL", raising=False)
    (tmp_path / "canon.yml").write_text("threshold: 0.33\njudge_model: openai:y\n")
    s = Settings.load(start=tmp_path, config_path=tmp_path / "canon.yml")
    assert s.threshold == 0.33 and s.judge_model == "openai:y"


def test_explicit_pyproject_without_tool_canon_yields_defaults(tmp_path):
    """Naming a config file that says nothing about Canon is not an error —
    there is simply nothing configured in it."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')
    assert Settings.load(start=tmp_path,
                         config_path=tmp_path / "pyproject.toml").threshold == 0.85


def test_same_directory_canon_yaml_beats_tool_canon(tmp_path):
    """When both live in the same directory, canon.yaml wins outright —
    [tool.canon] in pyproject.toml is only a fallback for projects with no
    canon.yaml at all, never a second vote once one exists alongside it."""
    (tmp_path / "canon.yaml").write_text("threshold: 0.42\n")
    (tmp_path / "pyproject.toml").write_text('[tool.canon]\nthreshold = 0.99\n')
    assert Settings.load(start=tmp_path).threshold == 0.42

def test_named_toml_with_top_level_canon_keys_but_no_tool_canon_raises(tmp_path):
    """`--config shared.toml` holding `threshold = 0.5` at top level would
    otherwise silently yield the defaults — the exact failure mode the
    [tool.canon] unwrap was meant to kill, resurfacing for non-pyproject
    TOML."""
    f = tmp_path / "shared.toml"
    f.write_text('threshold = 0.5\n')
    with pytest.raises(ConfigError, match=r"tool\.canon"):
        Settings.load(start=tmp_path, config_path=f)
