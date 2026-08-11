import dataclasses
import pytest

from canon.models import Constitution, QuestionResult, CoherenceResult
from canon import errors


def test_dataclasses_are_frozen():
    c = Constitution(mission="m", principles=("p",))
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.mission = "x"
    q = QuestionResult(id="A1", kind="adjudicate", score=1.0, max_score=2.0,
                       weight=2.0, is_gate=False, gate_tripped=False,
                       evidence="e", confidence=1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        q.score = 0.0
    r = CoherenceResult(score=0.5, gated=False, questions=(), reasons=())
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.score = 0.9


def test_coherence_result_serializes():
    q = QuestionResult(id="A1", kind="adjudicate", score=1.0, max_score=2.0,
                       weight=2.0, is_gate=False, gate_tripped=False,
                       evidence="values decided the tradeoff", confidence=1.0)
    r = CoherenceResult(score=0.62, gated=False, questions=(q,),
                        reasons=("non-canon: A1 partial",))
    d = r.to_dict()
    assert d["score"] == 0.62 and d["questions"][0]["id"] == "A1"
    assert d["reasons"] == ["non-canon: A1 partial"]


def test_error_hierarchy():
    for name in ("ConfigError", "JudgeError", "RubricError"):
        assert issubclass(getattr(errors, name), errors.CanonError)
