"""Tests for the public assertion helpers (canon.assertions). These wrap
CoherenceMetric.assert_coheres over a purpose-built artifact/constitution, so
the tests script a judge and check the wrapping, not the underlying scoring
pipeline (covered elsewhere)."""

import pytest

from canon.assertions import criteria_covered, task_is_coherent
from canon.judge.base import Answer, Judge
from canon.models import Constitution


class _ScriptedJudge(Judge):
    """A Judge with per-answer evidence, for tests that need to inspect what
    evidence lands in a failure message (MockJudge hardcodes evidence="mock")."""

    def __init__(self, fn):
        self._fn = fn

    def ask(self, system, question, choices):
        return self._fn(question, choices)


def _rule_judge(rules, default="yes"):
    """rules: ordered list of (substring, choice, evidence); first match wins."""

    def fn(q, choices):
        for substr, choice, evidence in rules:
            if substr in q:
                return Answer(choice=choice, evidence=evidence)
        return Answer(choice=default, evidence="default")

    return _ScriptedJudge(fn)


CON = Constitution(mission="Serve borrowers well", principles=("Be fair", "Be transparent"))


def test_task_is_coherent_fails_on_a_self_contradictory_goal_and_criteria():
    judge = _rule_judge(
        [
            ("in play for THIS decision", "relevant", "e"),
            ("SERVE or VIOLATE", "violates", "e"),
            ("EITHER true", "yes", "e"),
        ],
        default="no",
    )
    with pytest.raises(AssertionError, match="non-canon"):
        task_is_coherent(
            CON,
            goal="approve fast",
            criteria=["approve without checking income", "verify affordability"],
            judge=judge,
            samples=1,
        )


def test_task_is_coherent_ambiguous_case_is_not_distinguishable_from_contradiction():
    """An ambiguous read (no clean contradiction, nothing solid either) still
    raises a plain AssertionError with a numeric score in the message —
    there's no separate signal marking it "ambiguous" vs "contradictory".
    Kept as a known discrepancy, not fixed here."""
    judge = _rule_judge(
        [
            ("in play for THIS decision", "relevant", "e"),
            ("SERVE or VIOLATE", "partial", "e"),
            ("EITHER true", "no", "e"),
        ],
        default="partial",
    )
    with pytest.raises(AssertionError) as exc:
        task_is_coherent(
            CON, goal="ambiguous goal", criteria=["do something vague"], judge=judge, samples=1
        )
    assert "non-canon (0.5)" in str(exc.value)  # same shape as a hard contradiction's message


def test_task_is_coherent_passes_on_a_clean_task():
    judge = _rule_judge(
        [
            ("in play for THIS decision", "relevant", "e"),
            ("SERVE or VIOLATE", "serves", "e"),
            ("EITHER true", "no", "e"),
        ],
        default="yes",
    )
    res = task_is_coherent(
        CON,
        goal="Approve loans fairly",
        criteria=["verify income", "disclose terms"],
        judge=judge,
        samples=1,
    )
    assert res.score >= 0.85 and not res.gated


def test_task_is_coherent_fails_when_it_conflicts_with_a_principle():
    con = Constitution(
        mission="Serve borrowers well", principles=("Never discriminate by protected class",)
    )
    judge = _rule_judge(
        [
            ("in play for THIS decision", "relevant", "e"),
            ("SERVE or VIOLATE", "violates", "e"),
            ("EITHER true", "yes", "e"),
        ],
        default="no",
    )
    with pytest.raises(AssertionError, match="non-canon"):
        task_is_coherent(
            con,
            goal="Speed up approvals",
            criteria=["use zip code as the primary scoring signal"],
            judge=judge,
            samples=1,
        )


def test_criteria_covered_fails_and_names_the_missing_criterion():
    """A missing criterion becomes a principle-style violation; the judge's
    own evidence for that one question is what carries the criterion's
    identity into the failure message — criteria_covered does not otherwise
    tag which criterion an id like 'P2' corresponds to."""
    criteria = [
        "Verify affordability before approval",
        "Disclose the full repayment schedule",
        "Confirm identity documents",
    ]
    missing = "Disclose the full repayment schedule"
    judge = _rule_judge(
        [
            ("in play for THIS decision", "relevant", "e"),
            (missing, "violates", f"missing criterion: {missing!r}"),
            ("SERVE or VIOLATE", "serves", "e"),
            ("EITHER true", "no", "e"),
        ],
        default="yes",
    )
    with pytest.raises(AssertionError) as exc:
        criteria_covered(
            "Approval letter with affordability check but no repayment schedule shown.",
            goal="Approve responsibly",
            criteria=criteria,
            constitution=CON,
            judge=judge,
            threshold=0.92,
            samples=1,
        )
    assert missing in str(exc.value)


def test_criteria_covered_passes_when_all_criteria_are_met():
    criteria = ["Verify affordability before approval", "Disclose the full repayment schedule"]
    judge = _rule_judge(
        [
            ("in play for THIS decision", "relevant", "e"),
            ("SERVE or VIOLATE", "serves", "e"),
            ("EITHER true", "no", "e"),
        ],
        default="yes",
    )
    res = criteria_covered(
        "Approval letter with affordability check and full repayment schedule.",
        goal="Approve responsibly",
        criteria=criteria,
        constitution=CON,
        judge=judge,
        samples=1,
    )
    assert res.score >= 0.85 and not res.gated


# --- Use-case: both assertions driven through one realistic lending flow ---

LENDING_CON = Constitution(
    mission="Serve borrowers well",
    principles=("Be fair", "Be transparent about the cost of credit"),
)


def test_lending_use_case_task_then_result_are_both_checked():
    goal = "Approve a personal loan only when the borrower can afford it"
    criteria = [
        "Verify income and existing debt before approval",
        "Disclose the full repayment schedule and total cost of credit",
    ]

    def task_judge_fn(q, choices):
        if "in play for THIS decision" in q:
            return Answer("relevant", "e")
        if "SERVE or VIOLATE" in q:
            return Answer("serves", "e")
        if "EITHER true" in q:
            return Answer("no", "e")
        return Answer("yes", "e")

    task_res = task_is_coherent(
        LENDING_CON, goal=goal, criteria=criteria, judge=_ScriptedJudge(task_judge_fn), samples=1
    )
    assert task_res.score >= 0.85 and not task_res.gated

    result_artifact = (
        "Approval letter: income and existing debt verified against the "
        "requested payment; full repayment schedule and total cost of "
        "credit disclosed up front."
    )

    def result_judge_fn(q, choices):
        if "in play for THIS decision" in q:
            return Answer("relevant", "e")
        if "SERVE or VIOLATE" in q:
            return Answer("serves", "e")
        if "EITHER true" in q:
            return Answer("no", "e")
        return Answer("yes", "e")

    result_res = criteria_covered(
        result_artifact,
        goal=goal,
        criteria=criteria,
        constitution=LENDING_CON,
        judge=_ScriptedJudge(result_judge_fn),
        samples=1,
    )
    assert result_res.score >= 0.85 and not result_res.gated
