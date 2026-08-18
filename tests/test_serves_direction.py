"""Tests for serves_direction — which direction an artifact serves, answered
beside the coherence score and never folded into it. The rubric measures how
well an artifact reasons against one constitution; these tests cover the
separate question of WHICH constitution it was reasoning against."""

import dataclasses

import pytest

import canon.assertions
import canon.sampling
from canon import CoherenceMetric
from canon.assertions import DIRECTION_CHOICES, serves_direction
from canon.judge.base import Answer, Judge
from canon.judge.mock import MockJudge
from canon.models import Constitution, DirectionVerdict
from canon.rubric import Rubric, derive_questions

CURRENT = Constitution(
    mission="Grow revenue by deepening relationships with existing customers",
    principles=("Expand accounts we already serve", "Protect margin on every deal"),
    version="2026-08",
)
SUPERSEDED = Constitution(
    mission="Grow revenue by signing as many new logos as possible",
    principles=("Win new accounts first", "Buy share with discounts"),
    version="2026-05",
)

FRESH_ARTIFACT = (
    "Plan: move the two enterprise CSMs onto the 40 accounts renewing this "
    "quarter and pitch the analytics add-on at list price."
)
STALE_ARTIFACT = (
    "Plan: fund a 30% first-year discount blitz to sign 400 new logos this "
    "quarter, since new-logo count is what we are measured on."
)


class _SequenceJudge(Judge):
    """Answers a staged sequence, one entry per ask, so a test can compose an
    exact vote. An entry is a choice, or a (choice, evidence) pair."""

    def __init__(self, answers):
        self._answers = list(answers)
        self.asked: list[tuple[str, str]] = []

    def ask(self, system, question, choices):
        entry = self._answers[len(self.asked)]
        self.asked.append((system, question))
        choice, evidence = entry if isinstance(entry, tuple) else (entry, "e")
        return Answer(choice=choice, evidence=evidence)


def test_names_the_current_direction_when_the_artifact_serves_it():
    verdict = serves_direction(
        FRESH_ARTIFACT,
        current=CURRENT,
        superseded=SUPERSEDED,
        judge=_SequenceJudge(["current"] * 3),
        samples=3,
    )
    assert verdict.serves == "current"
    assert verdict.confidence == 1.0


def test_catches_an_artifact_still_serving_the_superseded_direction():
    verdict = serves_direction(
        STALE_ARTIFACT,
        current=CURRENT,
        superseded=SUPERSEDED,
        judge=_SequenceJudge(["superseded"] * 3),
        samples=3,
    )
    assert verdict.serves == "superseded"


def test_both_directions_is_a_verdict_of_its_own():
    verdict = serves_direction(
        "Plan: renew the top 40 accounts and chase 400 new logos with discounts.",
        current=CURRENT,
        superseded=SUPERSEDED,
        judge=_SequenceJudge(["both"] * 3),
        samples=3,
    )
    assert verdict.serves == "both"


def test_an_artifact_serving_neither_direction():
    verdict = serves_direction(
        "Ticket: rotate the staging TLS certificate before it expires.",
        current=CURRENT,
        superseded=SUPERSEDED,
        judge=_SequenceJudge(["neither"] * 3),
        samples=3,
    )
    assert verdict.serves == "neither"


def test_the_majority_wins_and_every_choice_is_reported():
    judge = _SequenceJudge(["superseded", "current", "superseded", "both", "superseded"])
    verdict = serves_direction(
        STALE_ARTIFACT, current=CURRENT, superseded=SUPERSEDED, judge=judge, samples=5
    )
    assert verdict.serves == "superseded"
    assert verdict.votes == {"current": 1, "superseded": 3, "both": 1, "neither": 0}
    assert verdict.confidence == 0.6
    assert set(verdict.votes) == set(DIRECTION_CHOICES)


def test_evidence_comes_from_a_sample_that_voted_the_winner():
    judge = _SequenceJudge(
        [("current", "wrong side"), ("superseded", "cites new-logo count"), "superseded"]
    )
    verdict = serves_direction(
        STALE_ARTIFACT, current=CURRENT, superseded=SUPERSEDED, judge=judge, samples=3
    )
    assert verdict.serves == "superseded"
    assert verdict.evidence == "cites new-logo count"


def test_a_tie_goes_to_the_earliest_sampled_choice():
    """Deliberate: a tied vote resolves to whichever choice was sampled first,
    the same rule the rubric's questions already use, and the 0.5 confidence
    is what tells a reader the vote was split."""
    judge = _SequenceJudge(["superseded", "current"])
    verdict = serves_direction(
        STALE_ARTIFACT, current=CURRENT, superseded=SUPERSEDED, judge=judge, samples=2
    )
    assert verdict.serves == "superseded"
    assert verdict.confidence == 0.5
    assert verdict.votes == {"current": 1, "superseded": 1, "both": 0, "neither": 0}


def test_one_sample_asks_once_and_reports_full_confidence():
    judge = _SequenceJudge(["current"])
    verdict = serves_direction(
        FRESH_ARTIFACT, current=CURRENT, superseded=SUPERSEDED, judge=judge, samples=1
    )
    assert len(judge.asked) == 1
    assert verdict.serves == "current" and verdict.confidence == 1.0


def test_fewer_than_one_sample_is_refused():
    with pytest.raises(ValueError, match="samples"):
        serves_direction(
            FRESH_ARTIFACT,
            current=CURRENT,
            superseded=SUPERSEDED,
            judge=_SequenceJudge(["current"]),
            samples=0,
        )


def test_identical_directions_are_refused_rather_than_answered():
    """Two identical constitutions give the judge nothing to tell apart, so any
    verdict would read as an all-clear the comparison never established."""
    twin = Constitution(mission=CURRENT.mission, principles=CURRENT.principles, version="2026-01")
    with pytest.raises(ValueError, match="identical"):
        serves_direction(
            FRESH_ARTIFACT,
            current=CURRENT,
            superseded=twin,
            judge=_SequenceJudge(["current"]),
            samples=1,
        )


def test_both_directions_reach_the_judge_labelled():
    judge = _SequenceJudge(["current"])
    serves_direction(FRESH_ARTIFACT, current=CURRENT, superseded=SUPERSEDED, judge=judge, samples=1)
    system, question = judge.asked[0]
    assert CURRENT.mission in system and SUPERSEDED.mission in system
    assert "CURRENT" in system and "SUPERSEDED" in system
    assert FRESH_ARTIFACT in question


def test_the_task_is_carried_into_the_question():
    judge = _SequenceJudge(["current"])
    serves_direction(
        FRESH_ARTIFACT,
        current=CURRENT,
        superseded=SUPERSEDED,
        judge=judge,
        task="allocate the CSM team",
        samples=1,
    )
    assert "allocate the CSM team" in judge.asked[0][1]


def test_verdict_is_frozen_and_serializes():
    verdict = DirectionVerdict(
        serves="superseded",
        votes={"current": 1, "superseded": 4, "both": 0, "neither": 0},
        confidence=0.8,
        evidence="cites the retired new-logo mission",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        verdict.serves = "current"
    assert verdict.to_dict() == {
        "serves": "superseded",
        "votes": {"current": 1, "superseded": 4, "both": 0, "neither": 0},
        "confidence": 0.8,
        "evidence": "cites the retired new-logo mission",
    }


# --- Pins on how this is built ---


def test_the_vote_runs_through_canons_shared_sampling_machinery():
    """serves_direction votes with canon.sampling, not a second implementation
    of majority voting that could drift from the rubric's."""
    assert canon.assertions.majority_vote is canon.sampling.majority_vote
    judge = _SequenceJudge(["current"] * 5)
    serves_direction(FRESH_ARTIFACT, current=CURRENT, superseded=SUPERSEDED, judge=judge, samples=5)
    assert len(judge.asked) == 5


def test_the_direction_question_is_absent_from_the_locked_rubric():
    rubric = Rubric.load_default()
    assert rubric.version == "cdt-3"
    for q in derive_questions(rubric, CURRENT):
        assert "superseded" not in q.text.lower()
        assert "which direction" not in q.text.lower()
        assert tuple(q.choices) != DIRECTION_CHOICES


def test_scoring_never_asks_the_direction_question():
    asked: list[str] = []

    def script(q, choices):
        asked.append(q)
        if "in play for THIS decision" in q:
            return "relevant"
        if "SERVE or VIOLATE" in q:
            return "serves"
        if "EITHER true" in q:
            return "no"
        return "yes"

    metric = CoherenceMetric(constitution=CURRENT, judge=MockJudge(script=script), samples=1)
    metric.score(STALE_ARTIFACT, task="allocate the CSM team")
    assert asked
    assert not any("Which direction" in q for q in asked)
    assert not any(SUPERSEDED.mission in q for q in asked)


def test_coherence_metric_takes_no_superseded_direction():
    """The score stays one constitution's yardstick: direction is asked beside
    it, so CoherenceMetric grows no knob for a second constitution."""
    with pytest.raises(TypeError, match="superseded"):
        CoherenceMetric(constitution=CURRENT, superseded=SUPERSEDED)


# --- Use case: a stale agent that reasons beautifully against the old direction ---


def test_a_stale_artifact_can_score_canon_while_serving_the_superseded_direction():
    """The failure this assertion exists for: direction changed, the agent kept
    the old copy, and the rubric — asking only how well it reasons — passes it.
    Direction is the instrument that catches it, and the score is unchanged."""

    def coherence_script(q, choices):
        if "in play for THIS decision" in q:
            return "relevant"
        if "SERVE or VIOLATE" in q:
            return "serves"
        if "EITHER true" in q:
            return "no"
        return "yes"

    metric = CoherenceMetric(
        constitution=CURRENT, threshold=0.85, judge=MockJudge(script=coherence_script), samples=1
    )
    result = metric.assert_coheres(STALE_ARTIFACT, task="allocate the CSM team")
    assert result.score >= 0.85 and not result.gated

    verdict = serves_direction(
        STALE_ARTIFACT,
        current=CURRENT,
        superseded=SUPERSEDED,
        judge=_SequenceJudge([("superseded", "targets new-logo count, the retired mission")] * 3),
        task="allocate the CSM team",
        samples=3,
    )
    assert verdict.serves == "superseded"
    assert verdict.confidence == 1.0
    assert "new-logo" in verdict.evidence

    rescored = metric.score(STALE_ARTIFACT, task="allocate the CSM team")
    assert rescored.score == result.score
