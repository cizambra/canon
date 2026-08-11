from canon.baseline import artifact_key, record_baseline
from canon.gate import gate
from canon.models import CoherenceResult, QuestionResult


def _r(score, gated=False):
    return CoherenceResult(
        score=score, gated=gated, questions=(), reasons=() if not gated else ("gate",)
    )


def test_hard_fail_below_threshold():
    rep = gate([_r(0.6)], baseline=None, threshold=0.85)
    assert rep.passed is False and rep.hard_fails


def test_pass_when_above_threshold_no_baseline():
    rep = gate([_r(0.9), _r(0.88)], baseline=None, threshold=0.85)
    assert rep.passed is True


def test_drift_fail_vs_baseline():
    base = record_baseline([_r(0.95)] * 8, rubric_version="cdt-1")
    rep = gate([_r(0.70)] * 8, baseline=base, threshold=0.5)  # threshold low so only drift trips
    assert rep.passed is False and rep.drift


def test_gate_reports_pass_rate_and_ci():
    rep = gate([_r(0.9), _r(0.9), _r(0.6)], baseline=None, threshold=0.85)
    assert rep.pass_rate == round(2 / 3, 4)
    lo, hi = rep.pass_rate_ci
    assert 0.0 <= lo < hi <= 1.0


def test_gate_on_empty_current_raises():
    import pytest

    with pytest.raises(ValueError):
        gate([], baseline=None, threshold=0.85)


def test_tiny_decrease_within_tolerance_passes():
    """A 0.0001 drop is noise, not a regression, even when the stderr is 0."""
    base = record_baseline([_r(0.9001)], rubric_version="cdt-1")
    rep = gate([_r(0.9000)], baseline=base, threshold=0.5)
    assert rep.drift is None and rep.passed is True


def test_large_decrease_with_zero_stderr_fails():
    base = record_baseline([_r(0.95)], rubric_version="cdt-1")
    rep = gate([_r(0.90)], baseline=base, threshold=0.5)
    assert rep.drift and rep.passed is False


def test_tolerance_is_configurable_on_the_gate():
    base = record_baseline([_r(0.95)], rubric_version="cdt-1")
    rep = gate([_r(0.90)], baseline=base, threshold=0.5, tolerance=0.10)
    assert rep.drift is None and rep.passed is True


def _qr(qid, score, confidence, max_score=2.0, subject=None):
    return QuestionResult(
        id=qid,
        kind="mission",
        score=score,
        max_score=max_score,
        weight=1.0,
        is_gate=False,
        gate_tripped=False,
        evidence="e",
        confidence=confidence,
        subject=subject,
    )


def _with_q(score, questions, key=None):
    return CoherenceResult(
        score=score, gated=False, questions=tuple(questions), reasons=(), artifact_key=key
    )


_KEY_A = artifact_key("artifact A")
_KEY_B = artifact_key("artifact B")


def test_solid_flip_on_one_question_is_a_localized_hard_fail():
    base = record_baseline([_with_q(0.9, [_qr("P1", 2.0, 1.0)], _KEY_A)], rubric_version="cdt-1")
    rep = gate([_with_q(0.9, [_qr("P1", 0.0, 1.0)], _KEY_A)], baseline=base, threshold=0.85)
    assert rep.passed is False
    assert any("P1" in hf and "[0]" in hf for hf in rep.hard_fails)


def test_borderline_flip_does_not_hard_fail():
    """A low-confidence flip is noise the statistical layer owns, not a hard fail."""
    base = record_baseline([_with_q(0.9, [_qr("P1", 2.0, 0.5)], _KEY_A)], rubric_version="cdt-1")
    rep = gate([_with_q(0.9, [_qr("P1", 0.0, 0.5)], _KEY_A)], baseline=base, threshold=0.85)
    assert rep.hard_fails == [] and rep.passed is True


def test_removing_an_artifact_from_the_suite_does_not_shift_layer_one_pairing():
    """Layer 1 pairs by artifact identity, not list position -- the survivor
    must compare against its OWN record, not the removed artifact's (which
    positional pairing would false-fail on)."""
    base = record_baseline(
        [_with_q(0.9, [_qr("P1", 2.0, 1.0)], _KEY_A), _with_q(0.9, [_qr("P1", 0.0, 1.0)], _KEY_B)],
        rubric_version="cdt-1",
    )
    rep = gate([_with_q(0.9, [_qr("P1", 0.0, 1.0)], _KEY_B)], baseline=base, threshold=0.85)
    assert rep.hard_fails == [] and rep.passed is True
    assert any("1 baseline artifact(s) not in this run" in n for n in rep.notes)


def test_a_genuine_flip_is_still_caught_when_the_suite_is_reordered():
    """Identity matching must not blunt the check: reorder the suite and a real
    solid flip is still named, against the right artifact's record."""
    base = record_baseline(
        [_with_q(0.9, [_qr("P1", 2.0, 1.0)], _KEY_A), _with_q(0.9, [_qr("P1", 2.0, 1.0)], _KEY_B)],
        rubric_version="cdt-1",
    )
    rep = gate(
        [
            _with_q(0.9, [_qr("P1", 0.0, 1.0)], _KEY_B),  # B flipped, now first
            _with_q(0.9, [_qr("P1", 2.0, 1.0)], _KEY_A),
        ],
        baseline=base,
        threshold=0.85,
    )
    assert rep.passed is False
    assert any("P1" in hf and "[0]" in hf for hf in rep.hard_fails)
    assert len(rep.hard_fails) == 1


def test_an_artifact_with_no_baseline_row_is_skipped_not_failed():
    """A newly added artifact has nothing accepted to regress from."""
    base = record_baseline([_with_q(0.9, [_qr("P1", 2.0, 1.0)], _KEY_A)], rubric_version="cdt-1")
    rep = gate(
        [
            _with_q(0.9, [_qr("P1", 2.0, 1.0)], _KEY_A),
            _with_q(0.9, [_qr("P1", 0.0, 1.0)], artifact_key("a brand new artifact")),
        ],
        baseline=base,
        threshold=0.85,
    )
    assert rep.hard_fails == [] and rep.passed is True and rep.notes == []


def test_reordered_principles_do_not_produce_a_false_flip():
    """A principle id is positional: P1 means "the first principle". Reorder or
    delete a principle and P1 names something else entirely, so comparing the
    two P1s compares unrelated facets."""
    base = record_baseline(
        [_with_q(0.9, [_qr("P1", 2.0, 1.0, subject="Be fair")], _KEY_A)], rubric_version="cdt-1"
    )
    rep = gate(
        [_with_q(0.9, [_qr("P1", 0.0, 1.0, subject="Handle refunds kindly")], _KEY_A)],
        baseline=base,
        threshold=0.85,
    )
    assert rep.hard_fails == []
    assert any("constitution changed under question P1" in n for n in rep.notes)


def test_a_flip_under_the_same_principle_still_hard_fails():
    """Subject matching must not blunt Layer 1 where the subject is unchanged."""
    base = record_baseline(
        [_with_q(0.9, [_qr("P1", 2.0, 1.0, subject="Be fair")], _KEY_A)], rubric_version="cdt-1"
    )
    rep = gate(
        [_with_q(0.9, [_qr("P1", 0.0, 1.0, subject="Be fair")], _KEY_A)],
        baseline=base,
        threshold=0.85,
    )
    assert rep.passed is False
    assert any("P1" in hf for hf in rep.hard_fails)
    assert rep.notes == []


def test_baseline_without_artifact_keys_skips_layer_one_with_a_note():
    """A baseline accepted before identity matching cannot be paired safely."""
    base = record_baseline([_with_q(0.9, [_qr("P1", 2.0, 1.0)], _KEY_A)], rubric_version="cdt-1")
    base.artifact_keys = None
    rep = gate([_with_q(0.9, [_qr("P1", 0.0, 1.0)], _KEY_A)], baseline=base, threshold=0.85)
    assert rep.hard_fails == []
    assert any("re-accept" in n for n in rep.notes)


def test_old_format_baseline_still_gates_on_score_with_a_note():
    old = record_baseline([_r(0.95)], rubric_version="cdt-1")
    old.questions = None  # a baseline accepted before V3
    rep = gate([_r(0.60)], baseline=old, threshold=0.85)
    assert rep.passed is False and rep.hard_fails  # score gate still applies
    assert any("re-accept" in n for n in rep.notes)


def test_a_drop_past_tolerance_is_still_absorbed_by_real_variance():
    """The drop is 0.05, well past the 0.02 tolerance, so only the noise term
    (2*combined_SE = 0.126 given this spread) can excuse it. The control below
    is the same drop with no variance, which must fail."""
    base = record_baseline(
        [_r(s) for s in (1.0, 0.8, 1.0, 0.8, 0.9)], rubric_version="cdt-1"
    )  # mean 0.900, SE 0.045
    rep = gate(
        [_r(s) for s in (0.95, 0.75, 0.95, 0.75, 0.85)], baseline=base, threshold=0.5
    )  # mean 0.850, SE 0.045
    assert rep.drift is None and rep.passed is True

    flat_base = record_baseline([_r(0.90)] * 5, rubric_version="cdt-1")
    flat = gate([_r(0.85)] * 5, baseline=flat_base, threshold=0.5)
    assert flat.drift and flat.passed is False  # same drop, no noise to hide in


def test_a_drop_past_the_noise_band_is_still_absorbed_by_tolerance():
    """These runs are so repeatable that 2*combined_SE is 0.0013, which the
    0.015 drop clears easily -- the declared 0.02 stability band is the only thing holding it."""
    base = record_baseline(
        [_r(s) for s in (0.901, 0.899, 0.900, 0.901, 0.899)], rubric_version="cdt-1"
    )  # mean 0.900
    rep = gate(
        [_r(s) for s in (0.886, 0.884, 0.885, 0.886, 0.884)], baseline=base, threshold=0.5
    )  # mean 0.885
    assert rep.drift is None and rep.passed is True

    tight = gate(
        [_r(s) for s in (0.886, 0.884, 0.885, 0.886, 0.884)],
        baseline=base,
        threshold=0.5,
        tolerance=0.005,
    )
    assert tight.drift and tight.passed is False  # same drop, tighter band


def test_the_drift_margin_is_two_standard_errors_not_one_point_nine_six():
    """DRIFT_Z is 2, a deliberately round margin rather than the 95% z of 1.96.
    This drop of 0.0990 sits in the gap between them: 1.96*combined_SE is
    0.0980 and 2*combined_SE is 0.1000, so it passes only under 2."""
    from canon.gate import DRIFT_Z

    assert DRIFT_Z == 2

    base = record_baseline([_r(0.94), _r(0.86)], rubric_version="cdt-1")  # mean 0.900
    rep = gate([_r(0.831), _r(0.771)], baseline=base, threshold=0.5)  # mean 0.801
    assert rep.drift is None and rep.passed is True


def test_threshold_boundary_at_the_gate_layer_is_inclusive():
    """score == threshold passes — `r.score < threshold` is the only failing
    comparison. The inclusive bound is deliberate."""
    rep = gate([_r(0.85)], baseline=None, threshold=0.85)
    assert rep.passed is True and rep.hard_fails == []


def test_keys_present_but_all_none_is_treated_as_old_format():
    """Results built outside CoherenceMetric.score (hand-built, bypassing the
    loader) carry no artifact key, so the recorded keys are all None -- Layer 1
    pairs nothing, and says so instead of silently checking nothing."""
    base = record_baseline([_with_q(0.9, [_qr("P1", 2.0, 1.0)], _KEY_A)], rubric_version="cdt-1")
    base.artifact_keys = [None]
    rep = gate([_with_q(0.9, [_qr("P1", 0.0, 1.0)], _KEY_A)], baseline=base, threshold=0.85)
    assert rep.hard_fails == []
    assert any("re-accept" in n for n in rep.notes)


def test_mismatched_key_and_question_lengths_note_instead_of_truncating():
    """A hand-edited baseline whose keys and question rows disagree in length
    must say so, not silently compare the shorter prefix."""
    base = record_baseline([_with_q(0.9, [_qr("P1", 2.0, 1.0)], _KEY_A)], rubric_version="cdt-1")
    base.artifact_keys = [_KEY_A, artifact_key("phantom second artifact")]
    rep = gate([_with_q(0.9, [_qr("P1", 0.0, 1.0)], _KEY_A)], baseline=base, threshold=0.85)
    assert rep.hard_fails == []
    assert any("disagree in length" in n for n in rep.notes)
