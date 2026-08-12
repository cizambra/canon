from pathlib import Path

import yaml
from typer.testing import CliRunner

from canon.cli import app

runner = CliRunner()


def _project(tmp_path, judge_choice_all="yes"):
    (tmp_path / "constitution.yaml").write_text(
        "mission: Serve well\nprinciples:\n  - Be fair\nversion: v1\n"
    )
    (tmp_path / "suite.yaml").write_text(
        yaml.safe_dump({"task": "decide", "artifacts": ["a fair decision serving the mission"]})
    )
    # a canon.yaml pointing the judge at a mock is not possible via CLI; tests inject via env flag
    return tmp_path


def test_set_judge_writes_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = runner.invoke(app, ["set-judge", "--provider", "together", "--model", "deepseek"])
    assert r.exit_code == 0
    assert "together:deepseek" in (tmp_path / "canon.yaml").read_text()


def test_check_uses_mock_when_flagged(tmp_path, monkeypatch):
    monkeypatch.chdir(_project(tmp_path))
    monkeypatch.setenv("CANON_MOCK_JUDGE", "yes")  # test hook: all-affirmative mock
    r = runner.invoke(app, ["accept", "--suite", "suite.yaml"])
    assert r.exit_code == 0
    assert Path("canon/baselines/suite.json").exists()
    r2 = runner.invoke(app, ["check", "--suite", "suite.yaml"])
    assert r2.exit_code == 0 and "PASS" in r2.output


def test_mock_judge_emits_warning(tmp_path, monkeypatch):
    monkeypatch.chdir(_project(tmp_path))
    monkeypatch.setenv("CANON_MOCK_JUDGE", "yes")
    r = runner.invoke(app, ["accept", "--suite", "suite.yaml"], catch_exceptions=False)
    assert r.exit_code == 0
    assert "MOCK judge" in r.output


def test_suite_with_no_artifacts_is_a_config_error(tmp_path, monkeypatch):
    """An empty suite must never report PASS — a gate that passes on nothing lies."""
    monkeypatch.chdir(_project(tmp_path))
    monkeypatch.setenv("CANON_MOCK_JUDGE", "yes")
    (tmp_path / "empty.yaml").write_text("task: decide\n")
    r = runner.invoke(app, ["check", "--suite", "empty.yaml"])
    assert r.exit_code == 2
    assert "error: suite has no artifacts" in r.output


def _accept(tmp_path, monkeypatch):
    monkeypatch.chdir(_project(tmp_path))
    monkeypatch.setenv("CANON_MOCK_JUDGE", "yes")
    assert runner.invoke(app, ["accept", "--suite", "suite.yaml"]).exit_code == 0
    return tmp_path / "canon" / "baselines" / "suite.json"


def test_corrupt_baseline_json_is_a_config_error(tmp_path, monkeypatch):
    bp = _accept(tmp_path, monkeypatch)
    bp.write_text("{ not json at all")
    r = runner.invoke(app, ["check", "--suite", "suite.yaml"])
    assert r.exit_code == 2
    assert "error: baseline unreadable or corrupt" in r.output


def test_baseline_with_unknown_key_is_a_config_error(tmp_path, monkeypatch):
    bp = _accept(tmp_path, monkeypatch)
    import json

    data = json.loads(bp.read_text())
    data["surprise"] = 1
    bp.write_text(json.dumps(data))
    r = runner.invoke(app, ["check", "--suite", "suite.yaml"])
    assert r.exit_code == 2
    assert "error: baseline unreadable or corrupt" in r.output


def test_cli_reports_malformed_config_as_error(tmp_path, monkeypatch):
    monkeypatch.chdir(_project(tmp_path))
    monkeypatch.setenv("CANON_MOCK_JUDGE", "yes")
    (tmp_path / "canon.yaml").write_text("threshold: [unclosed\n")
    r = runner.invoke(app, ["check", "--suite", "suite.yaml"])
    assert r.exit_code == 2 and "error:" in r.output


def test_cli_reports_non_numeric_threshold_as_error(tmp_path, monkeypatch):
    monkeypatch.chdir(_project(tmp_path))
    monkeypatch.setenv("CANON_MOCK_JUDGE", "yes")
    (tmp_path / "canon.yaml").write_text("threshold: not-a-number\n")
    r = runner.invoke(app, ["check", "--suite", "suite.yaml"])
    assert r.exit_code == 2 and "error: " in r.output and "threshold" in r.output


def test_rubric_version_change_forces_reaccept(tmp_path, monkeypatch):
    """A changed rubric invalidates the baseline: re-accept, don't silently compare."""
    bp = _accept(tmp_path, monkeypatch)
    import json

    data = json.loads(bp.read_text())
    data["rubric_version"] = "cdt-OLD"
    bp.write_text(json.dumps(data))
    r = runner.invoke(app, ["check", "--suite", "suite.yaml"])
    assert r.exit_code == 2
    assert "error: rubric changed (cdt-OLD -> " in r.output
    assert "canon accept" in r.output


def test_accept_records_per_question_baseline(tmp_path, monkeypatch):
    bp = _accept(tmp_path, monkeypatch)
    import json

    rows = json.loads(bp.read_text())["questions"][0]
    assert {"id", "score", "confidence"} <= set(rows[0])
    assert any(r["id"] == "P1" for r in rows)  # the derived principle question


def test_accept_then_check_catches_a_solid_per_question_flip(tmp_path, monkeypatch):
    import canon.cli as cli
    from canon.constitution import constitution_from_file
    from canon.judge.mock import MockJudge
    from canon.metric import CoherenceMetric

    monkeypatch.chdir(_project(tmp_path))
    flipped = {"on": False}

    def script(q, choices):
        if "in play for THIS decision" in q:
            return "relevant"
        if "SERVE or VIOLATE" in q:
            return "serves"
        if "EITHER true" in q:
            return "no"
        if flipped["on"] and "value-relevant IMPACT" in q:
            return "no"  # A2 flips
        return "yes"

    monkeypatch.setattr(
        cli,
        "_metric",
        lambda path, thr: CoherenceMetric(
            constitution=constitution_from_file(path),
            threshold=thr,
            judge=MockJudge(script=script),
            samples=1,
        ),
    )

    assert runner.invoke(app, ["accept", "--suite", "suite.yaml"]).exit_code == 0
    assert runner.invoke(app, ["check", "--suite", "suite.yaml"]).exit_code == 0

    flipped["on"] = True
    r = runner.invoke(app, ["check", "--suite", "suite.yaml"])
    assert r.exit_code == 1
    assert "A2" in r.output and "solidly violated" in r.output


def test_check_notes_an_old_format_baseline(tmp_path, monkeypatch):
    bp = _accept(tmp_path, monkeypatch)
    import json

    data = json.loads(bp.read_text())
    del data["questions"]
    bp.write_text(json.dumps(data))
    r = runner.invoke(app, ["check", "--suite", "suite.yaml"])
    assert r.exit_code == 0  # still gates, does not error
    assert "note:" in r.output and "canon accept" in r.output


def test_accepted_baseline_records_excluded_principles(tmp_path, monkeypatch):
    import json

    import canon.cli as cli
    from canon.constitution import constitution_from_file
    from canon.judge.mock import MockJudge
    from canon.metric import CoherenceMetric

    monkeypatch.chdir(tmp_path)
    (tmp_path / "constitution.yaml").write_text(
        "mission: Serve well\nprinciples:\n  - Be fair\n  - Handle refunds kindly\nversion: v1\n"
    )
    (tmp_path / "suite.yaml").write_text(
        yaml.safe_dump({"task": "decide", "artifacts": ["a fair decision serving the mission"]})
    )

    def script(q, choices):
        if "in play for THIS decision" in q:
            return "not_relevant" if "refunds" in q else "relevant"
        if "SERVE or VIOLATE" in q:
            return "serves"
        if "EITHER true" in q:
            return "no"
        return "yes"

    monkeypatch.setattr(
        cli,
        "_metric",
        lambda path, thr: CoherenceMetric(
            constitution=constitution_from_file(path),
            threshold=thr,
            judge=MockJudge(script=script),
            samples=1,
        ),
    )
    assert runner.invoke(app, ["accept", "--suite", "suite.yaml"]).exit_code == 0
    data = json.loads((tmp_path / "canon" / "baselines" / "suite.json").read_text())
    assert data["excluded_principles"] == [["Handle refunds kindly"]]


def test_cli_uses_the_config_file_named_by_the_config_option(tmp_path, monkeypatch):
    monkeypatch.chdir(_project(tmp_path))
    monkeypatch.setenv("CANON_MOCK_JUDGE", "yes")
    (tmp_path / "canon.yaml").write_text("baselines_dir: discovered\n")
    cfg = tmp_path / "custom" / "canon.yaml"
    cfg.parent.mkdir()
    cfg.write_text("baselines_dir: chosen\n")
    assert (
        runner.invoke(app, ["accept", "--suite", "suite.yaml", "--config", str(cfg)]).exit_code == 0
    )
    assert (tmp_path / "chosen" / "suite.json").exists()
    assert not (tmp_path / "discovered").exists()
    # and check reads the baseline back from the same named config
    assert (
        runner.invoke(app, ["check", "--suite", "suite.yaml", "--config", str(cfg)]).exit_code == 0
    )


def test_missing_config_file_is_an_error(tmp_path, monkeypatch):
    monkeypatch.chdir(_project(tmp_path))
    monkeypatch.setenv("CANON_MOCK_JUDGE", "yes")
    r = runner.invoke(app, ["check", "--suite", "suite.yaml", "--config", "nope.yaml"])
    assert r.exit_code == 2 and "error: config file not found" in r.output


def test_set_judge_from_subdir_updates_the_discovered_root_canon_yaml(tmp_path, monkeypatch):
    """Running set-judge from a subdirectory must not fork a second config file."""
    (tmp_path / "canon.yaml").write_text("threshold: 0.7\n")
    sub = tmp_path / "sub"
    sub.mkdir()
    monkeypatch.chdir(sub)
    r = runner.invoke(app, ["set-judge", "--provider", "together", "--model", "deepseek"])
    assert r.exit_code == 0
    root_cfg = (tmp_path / "canon.yaml").read_text()
    assert "together:deepseek" in root_cfg
    assert "threshold: 0.7" in root_cfg
    assert not (sub / "canon.yaml").exists()


def test_set_judge_refuses_a_pyproject_configured_project(tmp_path, monkeypatch):
    """Writing canon.yaml next to a [tool.canon] pyproject forks the config:
    canon.yaml wins in the same directory, so every other setting silently
    reverts to its default. Refuse instead of quietly reverting them."""
    (tmp_path / "pyproject.toml").write_text(
        '[tool.canon]\nthreshold = 0.42\njudge_model = "openai:x"\n'
    )
    monkeypatch.chdir(tmp_path)
    r = runner.invoke(app, ["set-judge", "--provider", "together", "--model", "deepseek"])
    assert r.exit_code == 2
    assert "config lives in pyproject.toml [tool.canon]" in r.output
    assert not (tmp_path / "canon.yaml").exists()
    assert "0.42" in (tmp_path / "pyproject.toml").read_text()


def test_paths_in_the_config_resolve_against_the_project_root(tmp_path, monkeypatch):
    """Relative paths in a config mean "relative to that config". Resolving them
    against the current directory instead makes `canon accept` from a subdir
    miss the project constitution and scatter baselines into the subdir."""
    _project(tmp_path)
    (tmp_path / "canon.yaml").write_text("threshold: 0.5\n")
    monkeypatch.setenv("CANON_MOCK_JUDGE", "yes")
    sub = tmp_path / "sub"
    sub.mkdir()
    monkeypatch.chdir(sub)

    r = runner.invoke(app, ["accept", "--suite", str(tmp_path / "suite.yaml")])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "canon" / "baselines" / "suite.json").exists()
    assert not (sub / "canon").exists()

    r2 = runner.invoke(app, ["check", "--suite", str(tmp_path / "suite.yaml")])
    assert r2.exit_code == 0 and "PASS" in r2.output


def test_missing_suite_file_is_an_error(tmp_path, monkeypatch):
    monkeypatch.chdir(_project(tmp_path))
    monkeypatch.setenv("CANON_MOCK_JUDGE", "yes")
    r = runner.invoke(app, ["check", "--suite", "nope.yaml"])
    assert r.exit_code == 2
    assert "error: suite file not found: nope.yaml" in r.output


def test_check_threshold_override_is_honored(tmp_path, monkeypatch):
    """--threshold overrides the configured/default threshold for this run only."""
    import canon.cli as cli
    from canon.constitution import constitution_from_file
    from canon.judge.mock import MockJudge
    from canon.metric import CoherenceMetric

    monkeypatch.chdir(_project(tmp_path))

    def script(q, choices):
        if "in play for THIS decision" in q:
            return "relevant"
        if "SERVE or VIOLATE" in q:
            return "partial"
        if "EITHER true" in q:
            return "no"
        return "partial"

    monkeypatch.setattr(
        cli,
        "_metric",
        lambda path, thr: CoherenceMetric(
            constitution=constitution_from_file(path),
            threshold=thr,
            judge=MockJudge(script=script),
            samples=1,
        ),
    )

    r_default = runner.invoke(app, ["check", "--suite", "suite.yaml"])
    assert r_default.exit_code == 1  # default threshold 0.85 fails a 0.5 score

    r_override = runner.invoke(app, ["check", "--suite", "suite.yaml", "--threshold", "0.3"])
    assert r_override.exit_code == 0
    assert "PASS" in r_override.output


def test_accepted_baseline_json_content(tmp_path, monkeypatch):
    """After `canon accept`, assert the actual baseline content — rubric
    version, per-artifact score, and per-question entries — not just that
    the file exists."""
    import json

    from canon.baseline import artifact_key
    from canon.rubric import Rubric

    bp = _accept(tmp_path, monkeypatch)
    data = json.loads(bp.read_text())

    assert data["rubric_version"] == Rubric.load_default().version
    assert data["scores"] == [1.0]  # the mock scores everything "yes"
    assert data["gated"] == [False]

    rows = data["questions"][0]
    assert {row["id"] for row in rows} >= {"M1", "A1", "H1", "P1"}
    for row in rows:
        assert set(row) == {"id", "score", "confidence", "subject"}
        assert row["score"] == 1.0 and row["confidence"] == 1.0
    # the derived principle question records the principle it was derived from
    assert next(r for r in rows if r["id"] == "P1")["subject"] == "Be fair"
    assert next(r for r in rows if r["id"] == "M1")["subject"] is None

    assert data["artifact_keys"] == [artifact_key("a fair decision serving the mission")]
