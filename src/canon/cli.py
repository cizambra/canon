from __future__ import annotations

import os
from pathlib import Path

import typer

from canon.baseline import Baseline, record_baseline
from canon.config import Settings, discover_config_file, set_judge as _set_judge
from canon.constitution import constitution_from_file
from canon.errors import CanonError, ConfigError
from canon.gate import gate
from canon.metric import CoherenceMetric
from canon.rubric import Rubric
from canon.runner import run_suite

app = typer.Typer(help="Canon — coherence testing.")


def _metric(constitution_path: str, threshold: float) -> CoherenceMetric:
    constitution = constitution_from_file(constitution_path)
    judge = None
    if os.environ.get("CANON_MOCK_JUDGE"):          # test hook only
        typer.echo(
            "WARNING: CANON_MOCK_JUDGE is set — using an all-affirmative MOCK judge; "
            "results are NOT real coherence checks. Unset CANON_MOCK_JUDGE for real runs.",
            err=True,
        )
        from canon.judge.mock import MockJudge
        judge = MockJudge(script=lambda q, ch: (
            "relevant" if "in play" in q else
            "serves" if "SERVE or VIOLATE" in q else
            "no" if "EITHER true" in q else "yes"))
    return CoherenceMetric(constitution, threshold=threshold, judge=judge,
                           samples=1 if judge else 5)


def _baseline_path(settings: Settings, suite: str) -> Path:
    return settings.baselines_directory / (Path(suite).stem + ".json")


@app.command("set-judge")
def set_judge(provider: str = typer.Option(...), model: str = typer.Option(...)):
    source = discover_config_file()
    if source is not None and source.name == "pyproject.toml":
        # canon.yaml beats [tool.canon] in the same directory, so writing one
        # here would not add a setting — it would replace the whole config with
        # a file holding nothing but the judge, reverting everything else.
        typer.echo("error: config lives in pyproject.toml [tool.canon]; set "
                   "judge_model there (set-judge manages canon.yaml only)", err=True)
        raise typer.Exit(code=2)
    target = source if source else Path.cwd() / "canon.yaml"
    _set_judge(provider, model, target)
    typer.echo(f"judge set to {provider}:{model}")


@app.command()
def accept(suite: str = typer.Option("suite.yaml"),
           constitution: str | None = typer.Option(None),
           config: str | None = typer.Option(None, help="Path to a canon.yaml to use")):
    try:
        s = Settings.load(config_path=config)
        results = run_suite(suite, _metric(constitution or str(s.constitution_file),
                                           s.threshold))
    except (FileNotFoundError, CanonError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2)
    bl = record_baseline(results, Rubric.load_default().version)
    bl.save(_baseline_path(s, suite))
    typer.echo(f"accepted baseline: {_baseline_path(s, suite)} ({len(results)} artifacts)")


@app.command()
def check(suite: str = typer.Option("suite.yaml"),
          constitution: str | None = typer.Option(None),
          threshold: float | None = typer.Option(None),
          config: str | None = typer.Option(None, help="Path to a canon.yaml to use")):
    try:
        s = Settings.load(config_path=config)
        thr = threshold if threshold is not None else s.threshold
        bp = _baseline_path(s, suite)
        results = run_suite(suite, _metric(constitution or str(s.constitution_file), thr))
        baseline = Baseline.load(bp) if bp.exists() else None
        rubric_version = Rubric.load_default().version
        if baseline and baseline.rubric_version != rubric_version:
            # Scores from a different rubric are not comparable: a rubric change
            # moves the yardstick, so the baseline has to be re-accepted rather
            # than quietly diffed against measurements of something else.
            raise ConfigError(
                f"rubric changed ({baseline.rubric_version} -> {rubric_version}); "
                f"run canon accept --suite {suite}")
    except (FileNotFoundError, CanonError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2)
    rep = gate(results, baseline, thr, s.tolerance)
    for note in rep.notes:
        typer.echo(f"note: {note}", err=True)
    if rep.passed:
        typer.echo(f"PASS — mean coherence {rep.mean}, pass-rate {rep.pass_rate} "
                   f"(95% CI {rep.pass_rate_ci[0]:.2f}-{rep.pass_rate_ci[1]:.2f})")
        return
    for hf in rep.hard_fails:
        typer.echo(f"FAIL {hf}", err=True)
    if rep.drift:
        typer.echo(f"FAIL {rep.drift}", err=True)
    raise typer.Exit(code=1)
