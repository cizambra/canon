from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

from canon.errors import ConfigError
from canon.models import CoherenceResult, QuestionResult

ARTIFACT_KEY_LENGTH = 12


def artifact_key(artifact: str) -> str:
    """Short, stable identity for an artifact's text; pairs a run to its
    baseline. Position can't do this — removing one artifact shifts every
    later comparison. Truncated because it's only compared to other digests
    of the same corpus, not used as a security claim."""
    return hashlib.sha256(artifact.encode("utf-8")).hexdigest()[:ARTIFACT_KEY_LENGTH]


@dataclass
class Baseline:
    rubric_version: str
    scores: list[float]
    gated: list[bool]
    # Per-artifact, per-question record: [{"id", "score" (0..1 or None), "confidence"}].
    # None means the baseline predates per-question recording; the gate then
    # falls back to whole-artifact scores instead of erroring, so an old
    # baseline keeps working until it is re-accepted.
    questions: list[list[dict]] | None = None
    excluded_principles: list[list[str]] | None = None
    # Identity of each recorded artifact, parallel to `questions`. None means the
    # baseline predates identity matching, so per-question rows cannot be paired
    # with confidence and the gate skips that layer rather than guessing.
    artifact_keys: list[str | None] | None = None

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Baseline":
        # A baseline is hand-editable, so a mangled one is a configuration
        # problem — not a coherence regression. Keeping the two distinguishable
        # is the whole point: exit 1 means "the system drifted", exit 2 means
        # "we could not tell".
        try:
            return cls(**json.loads(Path(path).read_text()))
        except (json.JSONDecodeError, TypeError, KeyError, OSError, ValueError) as exc:
            raise ConfigError(f"baseline unreadable or corrupt ({path}): {exc}") from exc


def normalized_score(q: QuestionResult) -> float | None:
    """A question's score on a 0..1 scale; None when the facet was not in play."""
    if q.score is None or q.max_score <= 0:
        return None
    return q.score / q.max_score


def _question_rows(result: CoherenceResult) -> list[dict]:
    # Gate questions carry no score of their own — the artifact-level `gated`
    # flag already records them — so only scored facets are worth recording.
    # `subject` travels with the id because the id alone is positional: it says
    # WHICH principle "P1" stood for when this baseline was accepted.
    return [{"id": q.id, "score": normalized_score(q), "confidence": q.confidence,
             "subject": q.subject}
            for q in result.questions if not q.is_gate]


def record_baseline(results: list[CoherenceResult], rubric_version: str) -> Baseline:
    return Baseline(rubric_version=rubric_version,
                    scores=[r.score for r in results],
                    gated=[r.gated for r in results],
                    questions=[_question_rows(r) for r in results],
                    excluded_principles=[list(r.excluded_principles) for r in results],
                    artifact_keys=[r.artifact_key for r in results])
