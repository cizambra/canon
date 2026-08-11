from canon import CoherenceMetric
from canon.models import Constitution
from canon.judge.mock import MockJudge
import pytest


def _constitution():
    return Constitution(mission="Serve borrowers well", principles=("Be fair",))


def test_metric_passes_when_all_yes():
    def script(q, choices):
        if "in play for THIS decision" in q: return "relevant"
        if "SERVE or VIOLATE" in q: return "serves"
        if "EITHER true" in q: return "no"          # gate not tripped
        return "yes"
    m = CoherenceMetric(constitution=_constitution(), threshold=0.85,
                        judge=MockJudge(script=script), samples=1)
    res = m.assert_coheres("a fair, mission-serving decision", task="decide")
    assert res.score >= 0.85 and not res.gated


def test_metric_fails_and_localizes_on_gate():
    def script(q, choices):
        if "in play for THIS decision" in q: return "relevant"
        if "SERVE or VIOLATE" in q: return "violates"
        if "EITHER true" in q: return "yes"         # gate tripped (commission)
        return "yes"
    m = CoherenceMetric(constitution=_constitution(), threshold=0.85,
                        judge=MockJudge(script=script), samples=1)
    with pytest.raises(AssertionError) as exc:
        m.assert_coheres("a decision that violates fairness", task="decide")
    assert "non-canon" in str(exc.value)


def test_metric_accepts_a_judge_model_string(monkeypatch):
    """judge="provider:model" is documented shorthand; it must resolve to a LiteLLMJudge."""
    import litellm
    seen = {}

    def fake_completion(**kwargs):
        seen["model"] = kwargs["model"]
        q = kwargs["messages"][1]["content"]
        choice = ("relevant" if "in play for THIS decision" in q else
                  "serves" if "SERVE or VIOLATE" in q else
                  "no" if "EITHER true" in q else "yes")
        return {"choices": [{"message": {
            "content": '{"choice": "%s", "evidence": "e"}' % choice}}]}

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.delenv("CANON_JUDGE_MODEL", raising=False)
    m = CoherenceMetric(constitution=_constitution(), threshold=0.85,
                        judge="openai:gpt-5.6-luna", samples=1)
    res = m.score("a fair, mission-serving decision", task="decide")
    assert res.score >= 0.85 and not res.gated
    assert seen["model"] == "openai/gpt-5.6-luna"


def test_samples_zero_is_rejected():
    with pytest.raises(ValueError, match="samples must be >= 1"):
        CoherenceMetric(constitution=_constitution(), judge=MockJudge(script=lambda q, c: c[-1]),
                        samples=0)


def test_sampling_rejects_zero_samples_directly():
    from canon.rubric import Rubric
    from canon.sampling import answer_question
    q = next(x for x in Rubric.load_default().questions if x.id == "M1")
    with pytest.raises(ValueError, match="samples must be >= 1"):
        answer_question(q, "a", "t", _constitution(),
                        MockJudge(script=lambda _q, _c: "yes"), samples=0)


def test_single_sample_is_full_confidence():
    def script(q, choices):
        if "in play for THIS decision" in q: return "relevant"
        if "SERVE or VIOLATE" in q: return "serves"
        if "EITHER true" in q: return "no"
        return "yes"
    m = CoherenceMetric(constitution=_constitution(), threshold=0.85,
                        judge=MockJudge(script=script), samples=1)
    res = m.score("a fair, mission-serving decision", task="decide")
    assert res.questions and all(q.confidence == 1.0 for q in res.questions)


def test_excluded_principles_are_recorded_on_the_result():
    """The relevance pass is a selection; a silent selection is not auditable."""
    con = Constitution(mission="Serve borrowers well",
                       principles=("Be fair", "Handle refunds kindly"))

    def script(q, choices):
        if "in play for THIS decision" in q:
            return "not_relevant" if "refunds" in q else "relevant"
        if "SERVE or VIOLATE" in q: return "serves"
        if "EITHER true" in q: return "no"
        return "yes"

    res = CoherenceMetric(constitution=con, threshold=0.85,
                          judge=MockJudge(script=script), samples=1).score("a", task="t")
    assert res.excluded_principles == ("Handle refunds kindly",)
    assert res.to_dict()["excluded_principles"] == ["Handle refunds kindly"]


def test_gate_omission_arm_a_silently_complied_with_drift_is_gated():
    """D1 is a two-sided gate: contradicts (commission) OR silently complies
    with drift (omission). This drives the omission arm;
    test_metric_fails_and_localizes_on_gate covers commission."""
    def script(q, choices):
        if "in play for THIS decision" in q: return "relevant"
        if "SERVE or VIOLATE" in q: return "partial"     # no outright violation
        if "EITHER true" in q: return "yes"              # omission arm trips the gate
        return "yes"
    m = CoherenceMetric(constitution=_constitution(), threshold=0.85,
                        judge=MockJudge(script=script), samples=1)
    with pytest.raises(AssertionError) as exc:
        m.assert_coheres(
            "The request drifted from the agreed fair-lending policy mid-thread; "
            "the decision went along with it without naming or pushing back on the drift.",
            task="decide")
    assert "non-canon" in str(exc.value) and "gate" in str(exc.value).lower()


def test_ungated_sub_threshold_score_raises_with_the_score_in_the_message():
    """A score below threshold with NO gate trip still raises — the message
    must carry the actual numeric score, not just a generic failure."""
    def script(q, choices):
        if "in play for THIS decision" in q: return "relevant"
        if "SERVE or VIOLATE" in q: return "partial"
        if "EITHER true" in q: return "no"               # gate never trips
        return "partial"
    m = CoherenceMetric(constitution=_constitution(), threshold=0.85,
                        judge=MockJudge(script=script), samples=1)
    with pytest.raises(AssertionError) as exc:
        m.assert_coheres("a middling decision", task="decide")
    msg = str(exc.value)
    assert "non-canon" in msg and "0.5" in msg


def test_threshold_boundary_score_exactly_equal_to_threshold_passes():
    """score == threshold is inclusive (`score < threshold` is the only
    failing comparison) — deliberate, not an off-by-one."""
    def script(q, choices):
        if "in play for THIS decision" in q: return "relevant"
        if "EITHER true" in q: return "no"
        return "partial"                                  # every graded facet -> raw 0.5
    con = Constitution(mission="M", principles=())
    m = CoherenceMetric(constitution=con, threshold=0.5,
                        judge=MockJudge(script=script), samples=1)
    res = m.assert_coheres("an artifact", task="t")        # must NOT raise
    assert res.score == 0.5


def test_stability_same_deterministic_mock_scores_the_same_artifact_identically():
    """A deterministic judge (no real-world sampling noise)
    scoring the same artifact twice must return an identical score and
    verdict — the pipeline itself introduces no nondeterminism."""
    def script(q, choices):
        if "in play for THIS decision" in q: return "relevant"
        if "SERVE or VIOLATE" in q: return "serves"
        if "EITHER true" in q: return "no"
        return "yes"
    m1 = CoherenceMetric(constitution=_constitution(), threshold=0.85,
                         judge=MockJudge(script=script), samples=1)
    m2 = CoherenceMetric(constitution=_constitution(), threshold=0.85,
                         judge=MockJudge(script=script), samples=1)
    r1 = m1.score("a fair, mission-serving decision", task="decide")
    r2 = m2.score("a fair, mission-serving decision", task="decide")
    assert r1.score == r2.score and r1.gated == r2.gated


def test_stability_samples_5_with_a_4_1_vote_stays_within_tolerance_across_runs():
    """samples=5 with a majority-vote (4/1) script: repeated independent runs
    must land within the declared 0.02 stability tolerance of each other."""
    def make_script():
        calls = {"n": 0}
        def script(q, choices):
            if "in play for THIS decision" in q: return "relevant"
            if "SERVE or VIOLATE" in q: return "serves"
            if "EITHER true" in q: return "no"
            calls["n"] += 1
            return "yes" if calls["n"] % 5 != 0 else "partial"   # 4/5 yes per question
        return script
    m1 = CoherenceMetric(constitution=_constitution(), threshold=0.85,
                         judge=MockJudge(script=make_script()), samples=5)
    m2 = CoherenceMetric(constitution=_constitution(), threshold=0.85,
                         judge=MockJudge(script=make_script()), samples=5)
    r1 = m1.score("a fair, mission-serving decision", task="decide")
    r2 = m2.score("a fair, mission-serving decision", task="decide")
    assert abs(r1.score - r2.score) <= 0.02


def test_custom_rubric_end_to_end_scores_with_the_custom_questions():
    """CoherenceMetric(rubric=<custom>) must use the custom questions instead
    of the packaged default rubric — a use-case for orgs that bring their own."""
    from canon.rubric import Rubric
    custom = Rubric.from_dict({
        "version": "custom-1",
        "questions": [
            {"id": "C1", "kind": "custom", "text": "Does it use plain language?",
             "choices": ["no", "yes"], "scores": {"no": 0, "yes": 1}, "weight": 1.0},
            {"id": "GATE", "kind": "gate", "text": "Any contradiction?",
             "choices": ["no", "yes"], "is_gate": True},
        ],
    })
    con = Constitution(mission="Serve well", principles=())

    def script(q, choices):
        if "in play for THIS decision" in q: return "relevant"
        if "Any contradiction" in q: return "no"
        if "plain language" in q: return "yes"
        return "yes"

    m = CoherenceMetric(constitution=con, threshold=0.5, judge=MockJudge(script=script),
                        samples=1, rubric=custom)
    res = m.score("a plainly written decision", task="t")
    assert {q.id for q in res.questions} == {"C1", "GATE"}
    assert res.score == 1.0 and not res.gated


def test_non_saturation_spread_use_case_six_artifacts_do_not_pile_at_1_0():
    """Use case: a suite of 6 artifacts of varying quality must produce a
    graded SPREAD of scores rather than clustering at the ceiling — the
    rubric's whole point is resolving *how* coherent, not just pass/fail."""
    from canon.runner import run_suite
    import tempfile
    from pathlib import Path
    import yaml

    con = Constitution(mission="Serve borrowers well", principles=("Be fair",))
    levels = {"LEVEL_A": "yes", "LEVEL_B": "partial", "LEVEL_C": "no",
             "LEVEL_D": "yes", "LEVEL_E": "partial", "LEVEL_F": "no"}
    flip_mission = {"LEVEL_D", "LEVEL_E", "LEVEL_F"}

    def script(q, choices):
        if "in play for THIS decision" in q: return "relevant"
        if "SERVE or VIOLATE" in q: return "serves"
        if "EITHER true" in q: return "no"
        marker = next(m for m in levels if m in q)
        if "ADVANCE the stated mission" in q:
            return "no" if marker in flip_mission else levels[marker]
        if "credible better-aligned ALTERNATIVES" in q:      # H1 has no n/a option
            return "partial"
        return levels[marker]

    artifacts = [f"decision text tagged {m} with enough detail to answer." for m in levels]
    suite = {"task": "decide", "artifacts": artifacts}
    p = Path(tempfile.mkdtemp()) / "suite.yaml"
    p.write_text(yaml.safe_dump(suite))

    m = CoherenceMetric(constitution=con, threshold=0.5, judge=MockJudge(script=script), samples=1)
    results = run_suite(p, m)
    scores = [r.score for r in results]
    assert len(set(scores)) >= 3
    assert not any(s == 1.0 for s in scores)
