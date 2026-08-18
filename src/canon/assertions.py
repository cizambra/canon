from __future__ import annotations

from canon.judge import resolve_judge
from canon.metric import CoherenceMetric
from canon.models import Constitution, DirectionVerdict
from canon.sampling import majority_vote

DIRECTION_CHOICES = ("current", "superseded", "both", "neither")


def assert_coheres(
    artifact: str,
    constitution: Constitution,
    threshold: float = 0.85,
    judge=None,
    task: str = "",
    samples: int = 5,
):
    return CoherenceMetric(constitution, threshold, judge, samples).assert_coheres(artifact, task)


def task_is_coherent(
    constitution: Constitution,
    goal: str,
    criteria: list[str],
    threshold: float = 0.85,
    judge=None,
    samples: int = 5,
):
    spec = "GOAL: " + goal + "\nACCEPTANCE CRITERIA:\n" + "\n".join(f"- {c}" for c in criteria)
    return CoherenceMetric(constitution, threshold, judge, samples).assert_coheres(
        spec,
        task=(
            "Judge whether this goal + acceptance criteria are internally "
            "consistent and consistent with the constitution"
        ),
    )


def criteria_covered(
    result_artifact: str,
    goal: str,
    criteria: list[str],
    constitution: Constitution,
    threshold: float = 0.85,
    judge=None,
    samples: int = 5,
):
    # Model each criterion as a principle-style coverage check via a criteria-only constitution.
    cov = Constitution(mission=goal, principles=tuple(criteria), version=constitution.version)
    return CoherenceMetric(cov, threshold, judge, samples).assert_coheres(
        result_artifact, task="Judge whether the result satisfies each acceptance criterion"
    )


def serves_direction(
    artifact: str,
    current: Constitution,
    superseded: Constitution,
    judge=None,
    task: str = "",
    samples: int = 5,
) -> DirectionVerdict:
    """Ask which of two directions an artifact serves: `current`, `superseded`,
    `both` or `neither`. This is a separate reading from the coherence score
    and does not touch it — an agent holding a retired copy of the direction
    can reason well enough against it to score high, and only this says so."""
    if (current.mission, current.principles) == (superseded.mission, superseded.principles):
        raise ValueError(
            "serves_direction() needs two different directions: `current` and `superseded` "
            "state identical missions and principles, so no verdict about the artifact "
            "could tell them apart"
        )
    system = (
        "You decide WHICH of two directions a piece of work serves — not how well "
        "it reasons, and not whether you agree with it.\n\n"
        + _describe("CURRENT", current)
        + "\n\n"
        + _describe("SUPERSEDED", superseded)
        + "\n\nThe superseded direction was replaced by the current one. Judge what the "
        "work actually pursues, citing evidence from it."
    )
    question = (
        f"Task: {task}\nWork:\n{artifact}\n\n"
        "Question: Which direction does this work serve — the CURRENT direction, the "
        "SUPERSEDED one, BOTH about equally, or NEITHER?"
    )
    tally = majority_vote(resolve_judge(judge), system, question, DIRECTION_CHOICES, samples)
    return DirectionVerdict(
        serves=tally.choice,
        votes=tally.votes,
        confidence=tally.confidence,
        evidence=tally.evidence,
    )


def _describe(label: str, constitution: Constitution) -> str:
    return f"{label} DIRECTION\nMISSION: {constitution.mission}\nPRINCIPLES:\n" + "\n".join(
        f"- {p}" for p in constitution.principles
    )
