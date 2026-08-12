from canon.judge.mock import MockJudge
from canon.models import Constitution
from canon.rubric import Rubric
from canon.sampling import answer_question
from canon.scoring import GATE_CAP, HUMILITY_CAP, score_result


def _q(qid):
    r = Rubric.load_default()
    return next(q for q in r.questions if q.id == qid)


def test_majority_and_confidence():
    calls = {"n": 0}

    def script(question, choices):
        calls["n"] += 1
        return "yes" if calls["n"] <= 3 else "partial"  # 3 yes, 2 partial of 5

    j = MockJudge(script=script)
    qr = answer_question(
        _q("A1"),
        artifact="a",
        task="t",
        constitution=Constitution(mission="M", principles=()),
        judge=j,
        samples=5,
    )
    assert qr.score == 2.0 and qr.confidence == 0.6


def test_gate_trip_caps_score():
    from canon.models import QuestionResult

    good = QuestionResult("A1", "adjudicate", 2.0, 2.0, 3.0, False, False, "e", 1.0)
    gate = QuestionResult("D1", "gate", None, 0.0, 0.0, True, True, "contradiction", 1.0)
    res = score_result([good, gate])
    assert res.gated is True and res.score <= GATE_CAP


def test_humility_caps_top():
    from canon.models import QuestionResult

    full = QuestionResult("A1", "adjudicate", 2.0, 2.0, 3.0, False, False, "e", 1.0)
    hum_low = QuestionResult("H1", "humility", 0.0, 2.0, 2.0, False, False, "no alts", 1.0)
    res = score_result([full, hum_low])
    assert res.score <= HUMILITY_CAP and not res.gated


def test_na_answer_is_excluded_from_score():
    """A facet answered 'n/a' (not in play) is scored None and excluded from the
    weighted average, rather than docking the artifact."""
    from canon.judge.mock import MockJudge
    from canon.models import Constitution, QuestionResult
    from canon.rubric import Question
    from canon.sampling import answer_question
    from canon.scoring import score_result

    q = Question(
        id="A3",
        kind="seek",
        text="...seek... Answer 'n/a' if the situation left no room.",
        choices=("n/a", "no", "partial", "yes"),
        scores={"no": 0, "partial": 1, "yes": 2},
        weight=1.0,
    )
    qr = answer_question(
        q,
        "a terse artifact",
        "task",
        Constitution(mission="m", principles=()),
        MockJudge(script={"": "n/a"}),
        samples=1,
    )
    assert qr.score is None

    full = QuestionResult("M1", "mission", 2.0, 2.0, 3.0, False, False, "e", 1.0)
    res = score_result([full, qr])
    assert res.score == 1.0


def test_gate_trips_on_the_choice_the_rubric_declares():
    """An org can phrase its gate however it likes — by SAYING which answer is
    the violation, not by relying on where it sits in the list."""
    from canon.rubric import Question

    q = Question(
        id="G1",
        kind="gate",
        text="Did it contradict a principle?",
        choices=("false", "true"),
        weight=0.0,
        is_gate=True,
        trips_on="true",
    )
    qr = answer_question(
        q,
        "an artifact",
        "task",
        Constitution(mission="m", principles=()),
        MockJudge(script=lambda _q, _c: "true"),
        samples=1,
    )
    assert qr.gate_tripped is True

    qr_clear = answer_question(
        q,
        "an artifact",
        "task",
        Constitution(mission="m", principles=()),
        MockJudge(script=lambda _q, _c: "false"),
        samples=1,
    )
    assert qr_clear.gate_tripped is False


def test_gate_listed_yes_first_still_trips_on_yes():
    """The inversion this replaces: with choices ["yes", "no"], tripping on the
    LAST choice meant a clean "no" tripped the gate and a "yes" cleared it."""
    from canon.rubric import Rubric

    r = Rubric.from_dict(
        {
            "version": "v1",
            "questions": [
                {
                    "id": "D1",
                    "kind": "gate",
                    "text": "Did it contradict a principle?",
                    "choices": ["yes", "no"],
                    "is_gate": True,
                }
            ],
        }
    )
    q = r.questions[0]
    tripped = answer_question(
        q,
        "an artifact",
        "task",
        Constitution(mission="m", principles=()),
        MockJudge(script=lambda _q, _c: "yes"),
        samples=1,
    )
    clear = answer_question(
        q,
        "an artifact",
        "task",
        Constitution(mission="m", principles=()),
        MockJudge(script=lambda _q, _c: "no"),
        samples=1,
    )
    assert tripped.gate_tripped is True and clear.gate_tripped is False


def test_default_rubric_gate_still_trips_on_yes():
    qr = answer_question(
        _q("D1"),
        "an artifact",
        "task",
        Constitution(mission="m", principles=()),
        MockJudge(script=lambda _q, _c: "yes"),
        samples=1,
    )
    assert qr.gate_tripped is True
    qr_clear = answer_question(
        _q("D1"),
        "an artifact",
        "task",
        Constitution(mission="m", principles=()),
        MockJudge(script=lambda _q, _c: "no"),
        samples=1,
    )
    assert qr_clear.gate_tripped is False


def test_all_na_artifact_states_why_it_is_non_canon():
    """A 0.0 non-canon verdict with an empty reasons list explains nothing."""
    from canon.models import QuestionResult

    nas = [
        QuestionResult("A1", "adjudicate", None, 0.0, 3.0, False, False, "e", 1.0),
        QuestionResult("A3", "seek", None, 0.0, 1.0, False, False, "e", 1.0),
    ]
    gate_q = QuestionResult("D1", "gate", None, 0.0, 0.0, True, False, "e", 1.0)
    res = score_result(nas + [gate_q])
    assert res.score == 0.0 and not res.gated
    assert res.reasons == ("no applicable evidence: every rubric facet was N/A for this artifact",)


def test_weighted_leverage_a_heavier_question_moves_the_aggregate_more():
    """Two otherwise-identical runs, differing only in WHICH question flips
    to zero: the higher-weight question must move the aggregate strictly
    more than the lower-weight one, given the aggregate is a weighted mean."""
    from canon.models import QuestionResult

    def full(qid, weight):
        return QuestionResult(qid, "mission", 2.0, 2.0, weight, False, False, "e", 1.0)

    def zeroed(qid, weight):
        return QuestionResult(qid, "mission", 0.0, 2.0, weight, False, False, "e", 1.0)

    baseline_score = score_result([full("M1", 3.0), full("A2", 2.0), full("A3", 1.0)]).score
    assert baseline_score == 1.0

    heavy_flip = score_result([zeroed("M1", 3.0), full("A2", 2.0), full("A3", 1.0)]).score
    light_flip = score_result([full("M1", 3.0), full("A2", 2.0), zeroed("A3", 1.0)]).score

    heavy_drop = baseline_score - heavy_flip
    light_drop = baseline_score - light_flip
    assert heavy_drop > light_drop > 0


def test_non_gate_question_with_no_scores_map_is_silently_excluded():
    """An empty `scores` mapping is rejected at Rubric.from_dict load time, but
    building a Question directly bypasses that guard -- max_score comes out
    0.0, and score_result's `max_score > 0` filter silently drops it."""
    from canon.judge.mock import MockJudge
    from canon.models import Constitution, QuestionResult
    from canon.rubric import Question
    from canon.sampling import answer_question

    q = Question(
        id="X1",
        kind="custom",
        text="does it do the thing?",
        choices=("no", "yes"),
        scores={},
        weight=1.0,
        is_gate=False,
    )
    qr = answer_question(
        q,
        "an artifact",
        "task",
        Constitution(mission="m", principles=()),
        MockJudge(script=lambda _q, _c: "yes"),
        samples=1,
    )
    assert qr.score == 0.0 and qr.max_score == 0.0

    full = QuestionResult("M1", "mission", 2.0, 2.0, 3.0, False, False, "e", 1.0)
    res = score_result([full, qr])
    assert res.score == 1.0
    assert res.reasons == ()


def test_even_sample_tie_resolves_first_seen_with_half_confidence():
    """samples=2 with a 1/1 vote: Counter.most_common ties break on
    insertion order, so the first-seen choice wins, with confidence exactly
    0.5. Resolving ties is the statistical drift layer's job, not this
    layer's — this keeps today's arbitrary-but-deterministic tiebreak from changing unnoticed."""
    calls = {"n": 0}

    def script(q, choices):
        calls["n"] += 1
        return "yes" if calls["n"] == 1 else "partial"

    q = _q("A1")
    qr = answer_question(
        q,
        artifact="a",
        task="t",
        constitution=Constitution(mission="M", principles=()),
        judge=MockJudge(script=script),
        samples=2,
    )
    assert qr.confidence == 0.5
    assert qr.score == 2.0


def test_evidence_propagation_majority_samples_evidence_lands_on_the_result():
    """The QuestionResult's evidence is the evidence of the FIRST sample that
    matches the winning (majority) choice — not an arbitrary or a randomly
    chosen one, and not a concatenation of all matching samples' evidence."""
    from canon.judge.base import Answer, Judge

    class _NumberedEvidenceJudge(Judge):
        def __init__(self):
            self.n = 0

        def ask(self, system, question, choices):
            self.n += 1
            choice = "yes" if self.n <= 3 else "partial"  # 3 yes, 2 partial of 5
            return Answer(choice=choice, evidence=f"sample-{self.n}-says-{choice}")

    qr = answer_question(
        _q("A1"),
        artifact="a",
        task="t",
        constitution=Constitution(mission="M", principles=()),
        judge=_NumberedEvidenceJudge(),
        samples=5,
    )
    assert qr.evidence == "sample-1-says-yes"
