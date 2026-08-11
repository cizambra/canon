from __future__ import annotations

from canon.models import CoherenceResult, QuestionResult

GATE_CAP = 0.3
HUMILITY_CAP = 0.9


def score_result(results: list[QuestionResult], humility_cap: bool = True,
                 excluded_principles: tuple[str, ...] = (),
                 artifact_key: str | None = None) -> CoherenceResult:
    scored = [r for r in results if not r.is_gate and r.score is not None and r.max_score > 0]
    if scored:
        num = sum(r.weight * (r.score / r.max_score) for r in scored)
        den = sum(r.weight for r in scored)
        raw = num / den if den else 0.0
    else:
        raw = 0.0

    reasons: list[str] = []
    if not scored:
        # Nothing was in play, so nothing is known to cohere. The relative
        # Non-Selective principle keeps this non-canon — but a 0.0 verdict with
        # no reason at all is a failure the reader cannot act on.
        reasons.append("no applicable evidence: every rubric facet was N/A "
                       "for this artifact")
    gated = False
    for r in results:
        if r.is_gate and r.gate_tripped:
            gated = True
            reasons.append(f"non-canon: NSCP gate tripped ({r.id}) — {r.evidence}")
        elif r.score is not None and r.max_score > 0 and r.score < r.max_score:
            reasons.append(f"{r.id} ({r.kind}) below full: {r.evidence}")

    score = raw
    if humility_cap:
        hum = next((r for r in results if r.kind == "humility" and r.score is not None), None)
        if hum and hum.max_score > 0 and hum.score < hum.max_score:
            score = min(score, HUMILITY_CAP)
    if gated:
        score = min(score, GATE_CAP)

    return CoherenceResult(score=round(score, 4), gated=gated,
                           questions=tuple(results), reasons=tuple(reasons),
                           excluded_principles=tuple(excluded_principles),
                           artifact_key=artifact_key)
