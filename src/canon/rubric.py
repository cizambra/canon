"""Internal: the packaged CDT rubric and the questions derived from it.
Loading and validating a rubric is not a public extension point — it exists to
author and test the one packaged rubric, which every run uses so that scores
stay comparable. `Rubric` is deliberately absent from `canon`'s exports.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from importlib import resources

import yaml

from canon.errors import RubricError
from canon.judge.base import Judge
from canon.models import Constitution

PRINCIPLE_CHOICES = ("violates", "absent", "partial", "serves")
PRINCIPLE_SCORES = {"violates": 0, "absent": 0, "partial": 1, "serves": 2}


@dataclass(frozen=True)
class Question:
    id: str
    kind: str
    text: str
    choices: tuple[str, ...]
    scores: dict[str, int] = field(default_factory=dict)
    weight: float = 1.0
    is_gate: bool = False
    subject: str | None = None   # for principle questions, the principle itself
    # For gate questions, the answer that means "violation". Resolved once here
    # so no later code has to infer it from choice order.
    trips_on: str | None = None

    def __post_init__(self) -> None:
        """Settle what trips a gate at construction, or refuse to build it.
        Which answer means "violation" is stated, not inferred: authors may
        declare `trips_on`, and "yes" is accepted as unambiguous. Inferring
        from choice order made ["yes", "no"] gate on the clean answer."""
        if not self.is_gate or self.trips_on is not None:
            if self.is_gate and self.trips_on not in self.choices:
                raise RubricError(
                    f"question {self.id!r}: trips_on {self.trips_on!r} is not one of "
                    f"its choices {tuple(self.choices)}")
            return
        if "yes" in self.choices:
            object.__setattr__(self, "trips_on", "yes")
            return
        raise RubricError(
            f"question {self.id!r}: a gate question must declare 'trips_on' — one of "
            f"{tuple(self.choices)} — since none of them is 'yes'")


@dataclass(frozen=True)
class Rubric:
    version: str
    questions: tuple[Question, ...]

    @classmethod
    def load_default(cls) -> "Rubric":
        text = resources.files("canon.data").joinpath("default_rubric.yaml").read_text()
        return cls.from_dict(yaml.safe_load(text))

    @classmethod
    def from_dict(cls, data: dict) -> "Rubric":
        """Build a rubric from its mapping form, validating it.
        A gate must name its tripping answer with `trips_on` unless "yes" is
        a choice. No gate is allowed but warns, since losing one is usually
        an accident. Choices must be strings: unquoted yes/no parse as YAML bools."""
        if "version" not in data or "questions" not in data:
            raise RubricError("rubric needs 'version' and 'questions'")
        raw = data["questions"]
        if not isinstance(raw, list) or not raw:
            raise RubricError("rubric has no questions")
        qs = [_question_from_dict(q) for q in raw]
        seen: set[str] = set()
        for q in qs:
            if q.id in seen:
                raise RubricError(f"question {q.id!r}: duplicate question id")
            seen.add(q.id)
        if not any(q.is_gate for q in qs):
            warnings.warn(
                "rubric has no gate question: nothing can hard-fail an artifact "
                "outright, only the graded score applies", UserWarning, stacklevel=2)
        return cls(version=str(data["version"]), questions=tuple(qs))


def _question_from_dict(q: dict) -> Question:
    if not isinstance(q, dict):
        raise RubricError(f"question must be a mapping, got {q!r}")
    qid = q.get("id")
    if not isinstance(qid, str) or not qid:
        raise RubricError(f"question needs a string 'id', got {qid!r}")
    for key in ("kind", "text"):
        if not isinstance(q.get(key), str) or not q[key]:
            raise RubricError(f"question {qid!r}: needs a non-empty string {key!r}")
    choices = q.get("choices")
    if (not isinstance(choices, list) or len(choices) < 2
            or not all(isinstance(c, str) and c for c in choices)):
        raise RubricError(
            f"question {qid!r}: 'choices' must list at least two strings "
            f"(quote yes/no in YAML — unquoted they parse as booleans), got {choices!r}")
    scores = q.get("scores", {}) or {}
    if not isinstance(scores, dict):
        raise RubricError(f"question {qid!r}: 'scores' must be a mapping")
    is_gate = bool(q.get("is_gate", False))
    for key, value in scores.items():
        if not isinstance(key, str):
            raise RubricError(
                f"question {qid!r}: score key {key!r} is not a string "
                f"(quote yes/no in YAML — unquoted they parse as booleans)")
        if key not in choices:
            raise RubricError(f"question {qid!r}: score key {key!r} is not one of "
                              f"its choices {tuple(choices)}")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise RubricError(f"question {qid!r}: score for {key!r} must be a number")
    if not is_gate and not scores:
        # A scored question with no scores contributes nothing to the average
        # and would just vanish from every run without saying so.
        raise RubricError(f"question {qid!r}: a non-gate question needs 'scores'")
    try:
        weight = float(q.get("weight", 1.0))
    except (TypeError, ValueError) as exc:
        raise RubricError(f"question {qid!r}: 'weight' must be a number") from exc
    if weight < 0:
        raise RubricError(f"question {qid!r}: 'weight' must not be negative, got {weight}")
    trips_on = q.get("trips_on")
    if trips_on is not None and not isinstance(trips_on, str):
        raise RubricError(f"question {qid!r}: 'trips_on' must be a string naming one of "
                          f"its choices (quote yes/no in YAML), got {trips_on!r}")
    return Question(id=qid, kind=q["kind"], text=q["text"], choices=tuple(choices),
                    scores=dict(scores), weight=weight, is_gate=is_gate,
                    trips_on=trips_on)


def derive_questions(rubric: Rubric, constitution: Constitution) -> list[Question]:
    out = list(rubric.questions)
    for i, principle in enumerate(constitution.principles):
        out.append(Question(
            id=f"P{i+1}", kind="principle",
            text=f'Does the decision SERVE or VIOLATE this principle: "{principle}"?',
            choices=PRINCIPLE_CHOICES, scores=dict(PRINCIPLE_SCORES), weight=2.0,
            subject=principle,
        ))
    return out


def relevant_questions(questions: list[Question], artifact: str, task: str,
                       judge: Judge, report_excluded: bool = False):
    """Drop principle questions the judge rules out of play for this artifact.
    With `report_excluded`, also return the dropped principles, so the
    selection is recorded rather than silently discarded.
    """
    kept: list[Question] = []
    excluded: list[str] = []
    for q in questions:
        if q.kind != "principle":
            kept.append(q)                      # mode/gate/humility always in play
            continue
        ans = judge.ask(
            system="You decide whether a principle is relevant to a specific decision.",
            question=f"Task: {task}\nDecision: {artifact}\n\n{q.text}\nIs this principle in play for THIS decision?",
            choices=("relevant", "not_relevant"),
        )
        if ans.choice == "relevant":
            kept.append(q)
        else:
            excluded.append(q.subject or q.id)
    return (kept, tuple(excluded)) if report_excluded else kept
