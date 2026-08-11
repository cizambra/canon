import json

from canon.baseline import Baseline, record_baseline
from canon.models import CoherenceResult, QuestionResult


def _qr(qid, score, confidence=1.0, max_score=2.0, is_gate=False, subject=None):
    return QuestionResult(
        id=qid,
        kind="mission",
        score=score,
        max_score=max_score,
        weight=1.0,
        is_gate=is_gate,
        gate_tripped=False,
        evidence="e",
        confidence=confidence,
        subject=subject,
    )


def test_baseline_records_per_question_scores_normalized(tmp_path):
    res = CoherenceResult(
        score=0.75,
        gated=False,
        questions=(_qr("M1", 2.0), _qr("A1", 1.0), _qr("A3", None, max_score=0.0)),
        reasons=(),
    )
    bl = record_baseline([res], rubric_version="cdt-3")
    assert bl.questions == [
        [
            {"id": "M1", "score": 1.0, "confidence": 1.0, "subject": None},
            {"id": "A1", "score": 0.5, "confidence": 1.0, "subject": None},
            {"id": "A3", "score": None, "confidence": 1.0, "subject": None},
        ]
    ]
    p = tmp_path / "b.json"
    bl.save(p)
    assert json.loads(p.read_text())["questions"][0][0]["id"] == "M1"
    assert Baseline.load(p).questions == bl.questions


def test_baseline_without_questions_still_loads(tmp_path):
    """Baselines accepted before per-question recording must keep loading."""
    p = tmp_path / "old.json"
    p.write_text(json.dumps({"rubric_version": "cdt-3", "scores": [0.9], "gated": [False]}))
    bl = Baseline.load(p)
    assert bl.scores == [0.9] and bl.questions is None


def test_baseline_records_the_principle_a_positional_id_stood_for():
    """P1 is positional; the subject is what says which principle it meant."""
    res = CoherenceResult(
        score=0.9, gated=False, questions=(_qr("P1", 2.0, subject="Be fair"),), reasons=()
    )
    bl = record_baseline([res], rubric_version="cdt-3")
    assert bl.questions[0][0]["subject"] == "Be fair"


def test_artifact_key_is_a_short_stable_digest_of_the_artifact_text():
    """Identity, not position, is what pairs a run against its baseline."""
    from canon.baseline import artifact_key

    k = artifact_key("a fair decision serving the mission")
    assert len(k) == 12 and all(c in "0123456789abcdef" for c in k)
    assert k == artifact_key("a fair decision serving the mission")
    assert k != artifact_key("a different decision")


def test_record_baseline_stores_each_artifacts_key(tmp_path):
    res = CoherenceResult(
        score=0.9, gated=False, questions=(_qr("M1", 2.0),), reasons=(), artifact_key="abc123def456"
    )
    bl = record_baseline([res], rubric_version="cdt-3")
    assert bl.artifact_keys == ["abc123def456"]
    p = tmp_path / "b.json"
    bl.save(p)
    assert Baseline.load(p).artifact_keys == ["abc123def456"]


def test_gate_questions_are_not_recorded_per_question():
    """The gate facet has no score of its own; the artifact-level flag holds it."""
    res = CoherenceResult(
        score=0.3,
        gated=True,
        questions=(_qr("M1", 2.0), _qr("D1", None, max_score=0.0, is_gate=True)),
        reasons=("gate",),
    )
    bl = record_baseline([res], rubric_version="cdt-3")
    assert [row["id"] for row in bl.questions[0]] == ["M1"]
    assert bl.gated == [True]
