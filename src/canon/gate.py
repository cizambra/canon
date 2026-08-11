from __future__ import annotations

from dataclasses import dataclass, field

from canon.baseline import Baseline, normalized_score
from canon.models import CoherenceResult
from canon.stats import mean_stderr, wilson_ci


@dataclass
class GateReport:
    passed: bool
    hard_fails: list[str]
    drift: str | None
    mean: float
    pass_rate: float
    pass_rate_ci: tuple[float, float]
    notes: list[str] = field(default_factory=list)


DEFAULT_TOLERANCE = 0.02

# How many standard errors a mean drop must clear to count as drift. A round 2,
# not the 1.96 of a textbook 95% interval: judge scores are not a clean normal
# sample, so the extra 2% of margin is a deliberate lean toward calling a drop
# noise rather than raising a regression nobody can reproduce.
DRIFT_Z = 2

# Layer 1 compares a question against its own accepted answer. "Solid" means the
# judge was both decisive (a lopsided vote) and near the end of the scale; only a
# solid-to-solid reversal is treated as a localized regression, so a borderline
# read never hard-fails on its own — that is the statistical layer's job.
SOLIDLY_SATISFIED = 0.9
SOLIDLY_VIOLATED = 0.1
SOLID_CONFIDENCE = 0.8


def _question_flips(current: list[CoherenceResult],
                    baseline: Baseline) -> tuple[list[str], list[str]]:
    """Compare each artifact against its own accepted record, matched by identity.
    Pairing is always by artifact key, never list position — a suite is an
    editable list, and position-pairing silently re-points every later
    comparison at the wrong record once one artifact is added or removed."""
    fails: list[str] = []
    notes: list[str] = []
    accepted_by_key: dict[str, list[dict]] = {}
    for key, rows in zip(baseline.artifact_keys or [], baseline.questions or []):
        if key is not None:
            accepted_by_key.setdefault(key, rows)
    matched: set[str] = set()
    for i, result in enumerate(current):
        rows = accepted_by_key.get(result.artifact_key) if result.artifact_key else None
        if rows is None:
            continue          # this artifact is new: nothing accepted to regress from
        matched.add(result.artifact_key)
        accepted = {row["id"]: row for row in rows}
        for q in result.questions:
            row = accepted.get(q.id)
            if row is None:
                continue                      # question not in play on both sides
            if row.get("subject") != q.subject:
                # Same id, different thing being asked about — the constitution
                # moved under the question. Comparing the two would report a
                # regression in a facet that was never measured before.
                note = (f"constitution changed under question {q.id}; "
                        f"re-accept to re-baseline")
                if note not in notes:
                    notes.append(note)
                continue
            was, now = row.get("score"), normalized_score(q)
            if was is None or now is None:
                continue
            if (was >= SOLIDLY_SATISFIED and row.get("confidence", 0.0) >= SOLID_CONFIDENCE
                    and now <= SOLIDLY_VIOLATED and q.confidence >= SOLID_CONFIDENCE):
                fails.append(
                    f"[{i}] {q.id}: solidly satisfied in the accepted baseline "
                    f"({was:.2f}) and now solidly violated ({now:.2f}): {q.evidence}")
    # Reported, never failed: an artifact leaving the suite is an ordinary edit,
    # not a regression — but a silent drop is how coverage quietly disappears.
    dropped = len(accepted_by_key) - len(matched)
    if dropped:
        notes.append(f"{dropped} baseline artifact(s) not in this run")
    return fails, notes


def gate(current: list[CoherenceResult], baseline: Baseline | None,
         threshold: float, tolerance: float = DEFAULT_TOLERANCE) -> GateReport:
    """Gate a run against its accepted baseline.
    `tolerance` is the stability band: a mean-score drop within it is not a
    regression. It floors the drift test — without one, identical scores
    give zero standard error and any decrease reads as significant."""
    # Defense in depth behind the runner's suite validation: an empty result set
    # has nothing to attest to, so there is no honest verdict to return.
    if not current:
        raise ValueError("gate() needs at least one result: an empty run cannot pass")
    hard_fails: list[str] = []
    notes: list[str] = []
    for i, r in enumerate(current):
        if r.gated or r.score < threshold:
            hard_fails.append(f"[{i}] non-canon ({r.score}): " + "; ".join(r.reasons))

    if baseline is not None:
        if (baseline.questions is None or baseline.artifact_keys is None
                or not any(baseline.artifact_keys)):
            # No per-question rows means nothing to compare; no artifact keys
            # (absent, or all None — results built outside CoherenceMetric.score
            # carry no key) means no honest way to know which row belongs to
            # which artifact, and a guessed pairing is worse than none.
            notes.append("baseline has no per-question record keyed by artifact, so "
                         "per-question regressions were not checked — run canon accept "
                         "to re-accept it in the current format")
        elif len(baseline.artifact_keys) != len(baseline.questions):
            # A hand-edited baseline can end up with the two lists out of step;
            # pairing the shorter prefix would compare wrong rows in silence.
            notes.append("baseline artifact keys and question rows disagree in length, "
                         "so per-question regressions were not checked — run canon "
                         "accept to re-accept a consistent baseline")
        else:
            layer_one_fails, layer_one_notes = _question_flips(current, baseline)
            hard_fails.extend(layer_one_fails)
            notes.extend(layer_one_notes)

    mean_cur, se_cur = mean_stderr([r.score for r in current])
    drift = None
    if baseline and baseline.scores:
        mean_base, se_base = mean_stderr(baseline.scores)
        combined = (se_cur ** 2 + se_base ** 2) ** 0.5
        # The mean-difference significance test is the drift signal Canon uses.
        # fisher_exact (canon.stats) compares drift as a rate instead — kept as
        # a deliberate alternative for callers who want it, not dead code.
        drop = mean_base - mean_cur
        if drop > max(DRIFT_Z * combined, tolerance):
            drift = (f"coherence regressed vs accepted baseline: "
                     f"{mean_base:.3f} -> {mean_cur:.3f} (significant)")

    passes = sum(1 for r in current if (not r.gated) and r.score >= threshold)
    n = len(current)
    pass_rate = passes / n if n else 0.0
    pass_rate_ci = wilson_ci(passes, n)

    return GateReport(passed=not hard_fails and drift is None,
                      hard_fails=hard_fails, drift=drift, mean=round(mean_cur, 4),
                      pass_rate=round(pass_rate, 4), pass_rate_ci=pass_rate_ci,
                      notes=notes)
