import pytest

from canon.errors import RubricError
from canon.judge.mock import MockJudge
from canon.models import Constitution
from canon.rubric import Rubric, derive_questions, relevant_questions


def test_default_rubric_loads_with_gate():
    r = Rubric.load_default()
    assert r.version
    assert any(q.is_gate for q in r.questions)  # the Non-Selective principle gate
    assert any(q.kind == "humility" for q in r.questions)


def test_derive_adds_one_question_per_principle():
    r = Rubric.load_default()
    c = Constitution(mission="M", principles=("be fair", "be transparent"))
    qs = derive_questions(r, c)
    principle_qs = [q for q in qs if q.kind == "principle"]
    assert len(principle_qs) == 2


def test_relevance_pass_keeps_core_drops_offtopic_principle():
    r = Rubric.load_default()
    c = Constitution(mission="M", principles=("be fair", "handle refunds kindly"))
    qs = derive_questions(r, c)
    judge = MockJudge(script={"be fair": "relevant", "handle refunds": "not_relevant"})
    kept = relevant_questions(qs, artifact="a hiring decision", task="hire", judge=judge)
    kept_principles = [q.text for q in kept if q.kind == "principle"]
    assert any("be fair" in t for t in kept_principles)
    assert all("refunds" not in t for t in kept_principles)
    assert any(q.is_gate for q in kept) and any(q.kind == "humility" for q in kept)


def test_relevance_pass_reports_what_it_excluded():
    r = Rubric.load_default()
    c = Constitution(mission="M", principles=("be fair", "handle refunds kindly"))
    qs = derive_questions(r, c)
    judge = MockJudge(script={"be fair": "relevant", "handle refunds": "not_relevant"})
    kept, excluded = relevant_questions(
        qs, artifact="a hiring decision", task="hire", judge=judge, report_excluded=True
    )
    assert [q.kind for q in kept].count("principle") == 1
    assert excluded == ("handle refunds kindly",)


def _rubric(questions, version="v1"):
    return {"version": version, "questions": questions}


def _q(**over):
    q = {
        "id": "M1",
        "kind": "mission",
        "text": "t?",
        "choices": ["no", "yes"],
        "scores": {"no": 0, "yes": 1},
        "is_gate": False,
    }
    q.update(over)
    return q


def test_rubric_with_no_questions_is_rejected():
    with pytest.raises(RubricError, match="no questions"):
        Rubric.from_dict(_rubric([]))


def test_rubric_with_duplicate_question_ids_is_rejected():
    with pytest.raises(RubricError, match="M1"):
        Rubric.from_dict(_rubric([_q(), _q(id="D1", is_gate=True, scores={}), _q()]))


def test_rubric_with_negative_weight_is_rejected():
    with pytest.raises(RubricError, match="M1.*weight"):
        Rubric.from_dict(_rubric([_q(weight=-1.0), _q(id="D1", is_gate=True, scores={})]))


def test_rubric_with_score_key_outside_choices_is_rejected():
    with pytest.raises(RubricError, match="M1.*maybe"):
        Rubric.from_dict(
            _rubric([_q(scores={"no": 0, "maybe": 1}), _q(id="D1", is_gate=True, scores={})])
        )


def test_rubric_with_a_missing_field_is_rejected_by_id():
    bad = _q()
    del bad["text"]
    with pytest.raises(RubricError, match="M1.*text"):
        Rubric.from_dict(_rubric([bad, _q(id="D1", is_gate=True, scores={})]))


def test_rubric_with_boolean_choices_is_rejected():
    """Unquoted yes/no in YAML become booleans; the loader must not accept them."""
    with pytest.raises(RubricError, match="M1.*choices"):
        Rubric.from_dict(
            _rubric([_q(choices=[False, True], scores={}), _q(id="D1", is_gate=True, scores={})])
        )


def test_scored_question_without_scores_is_rejected():
    with pytest.raises(RubricError, match="M1.*scores"):
        Rubric.from_dict(_rubric([_q(scores={}), _q(id="D1", is_gate=True, scores={})]))


def test_rubric_without_a_gate_question_loads_with_one_warning():
    """An org may genuinely not want a hard gate — allowed, but said out loud."""
    with pytest.warns(UserWarning, match="no gate question"):
        r = Rubric.from_dict(_rubric([_q()]))
    assert len(r.questions) == 1


def test_gate_choice_order_does_not_decide_what_trips():
    """Listing the choices as ["yes", "no"] is a presentation choice, not a
    semantic one — it must not silently invert the gate."""
    r = Rubric.from_dict(_rubric([_q(id="D1", is_gate=True, scores={}, choices=["yes", "no"])]))
    assert r.questions[0].trips_on == "yes"


def test_gate_with_explicit_trips_on_uses_it():
    r = Rubric.from_dict(
        _rubric([_q(id="D1", is_gate=True, scores={}, choices=["false", "true"], trips_on="true")])
    )
    assert r.questions[0].trips_on == "true"


def test_gate_without_trips_on_and_without_a_yes_choice_is_rejected():
    """Nothing here says which side is the violation, so guessing is the bug."""
    with pytest.raises(RubricError, match="D1.*trips_on"):
        Rubric.from_dict(_rubric([_q(id="D1", is_gate=True, scores={}, choices=["false", "true"])]))


def test_gate_with_trips_on_outside_its_choices_is_rejected():
    with pytest.raises(RubricError, match="D1.*trips_on"):
        Rubric.from_dict(
            _rubric([_q(id="D1", is_gate=True, scores={}, choices=["no", "yes"], trips_on="maybe")])
        )


def test_packaged_default_rubric_declares_what_trips_its_gate():
    gate = next(q for q in Rubric.load_default().questions if q.is_gate)
    assert gate.trips_on == "yes"


def test_packaged_default_rubric_uses_string_choices_and_score_keys():
    r = Rubric.load_default()
    for q in r.questions:
        assert all(isinstance(c, str) for c in q.choices), q.id
        assert all(isinstance(k, str) for k in q.scores), q.id
