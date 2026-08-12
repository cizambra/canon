from __future__ import annotations

from canon.metric import CoherenceMetric
from canon.models import Constitution


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
