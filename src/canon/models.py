from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Constitution:
    mission: str
    principles: tuple[str, ...]
    version: str | None = None


@dataclass(frozen=True)
class QuestionResult:
    id: str
    kind: str
    score: float | None  # None = N/A (not in play)
    max_score: float
    weight: float
    is_gate: bool
    gate_tripped: bool
    evidence: str
    confidence: float  # 0..1, lopsidedness of the N-sample vote
    # What the question was ABOUT — for principle questions, the principle text.
    # Question ids like "P1" are positional, so the subject is the only thing
    # that says whether two runs' P1s asked about the same principle at all.
    subject: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "score": self.score,
            "max_score": self.max_score,
            "weight": self.weight,
            "is_gate": self.is_gate,
            "gate_tripped": self.gate_tripped,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "subject": self.subject,
        }


@dataclass(frozen=True)
class DirectionVerdict:
    """Which direction an artifact serves — reported beside a coherence score,
    never inside it. An artifact can reason impeccably against a direction that
    was retired last week, and the score alone cannot say so."""

    serves: str  # "current" | "superseded" | "both" | "neither"
    votes: dict[str, int]  # every choice's share of the N-sample vote
    confidence: float  # 0..1, lopsidedness of the N-sample vote
    evidence: str

    def to_dict(self) -> dict:
        return {
            "serves": self.serves,
            "votes": dict(self.votes),
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class CoherenceResult:
    score: float  # 0..1 graded coherence
    gated: bool  # Non-Selective principle (NSCP) gate tripped
    questions: tuple[QuestionResult, ...]
    reasons: tuple[str, ...]  # localized failing-question summaries
    # Principles the relevance pass judged not in play for this artifact. The
    # selection is part of the verdict: which principles were NOT weighed is as
    # auditable as how the weighed ones scored.
    excluded_principles: tuple[str, ...] = ()
    # Which artifact this verdict is about (canon.baseline.artifact_key). It is
    # what lets a later run find this artifact's own accepted record instead of
    # whatever happens to sit at the same list position.
    artifact_key: str | None = None

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "gated": self.gated,
            "questions": [q.to_dict() for q in self.questions],
            "reasons": list(self.reasons),
            "excluded_principles": list(self.excluded_principles),
            "artifact_key": self.artifact_key,
        }
