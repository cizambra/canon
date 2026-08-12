import pytest

from canon.stats import fisher_exact, mean_stderr, wilson_ci


def test_wilson_bounds_within_0_1():
    lo, hi = wilson_ci(38, 40)
    assert 0.0 <= lo < hi <= 1.0 and lo > 0.7


def test_wilson_perfect_is_not_zero_width():
    lo, hi = wilson_ci(40, 40)
    assert hi == 1.0 or hi > 0.99
    assert lo < 1.0  # not a degenerate zero-width interval


def test_mean_stderr():
    m, se = mean_stderr([0.8, 0.9, 1.0, 0.7])
    assert abs(m - 0.85) < 1e-9 and se > 0


def test_fisher_symmetric_significant():
    # 9/12 vs 0/12 is a strong difference
    p = fisher_exact(9, 3, 0, 12)
    assert p < 0.01


def test_wilson_zero_successes_lower_bound_is_exactly_zero():
    lo, hi = wilson_ci(0, 20)
    assert lo == 0.0 and 0.0 < hi <= 1.0


def test_wilson_all_successes_upper_bound_is_exactly_one():
    lo, hi = wilson_ci(20, 20)
    assert hi == 1.0 and 0.0 <= lo < 1.0


def test_wilson_no_observations_is_total_ignorance():
    """n=0 knows nothing, so the interval is the whole range — not a false 0-width 0."""
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_fisher_exact_rejects_negative_cell():
    """A negative count must surface as a clean ValueError, not a KeyError
    from deep inside the log-factorial cache."""
    import pytest

    with pytest.raises(ValueError, match="non-negative"):
        fisher_exact(-1, 3, 0, 12)


def test_fisher_exact_still_works_on_valid_tables():
    p = fisher_exact(9, 3, 0, 12)
    assert 0.0 <= p <= 1.0


def test_mean_stderr_empty_list():
    assert mean_stderr([]) == (0.0, 0.0)


def test_mean_stderr_single_value():
    assert mean_stderr([0.7]) == (0.7, 0.0)


def test_fisher_exact_degenerate_zero_row_is_near_certain():
    """A row/column that is entirely zero leaves nothing to be surprising
    about — Fisher's exact test on a degenerate table returns ~1.0."""
    assert fisher_exact(0, 0, 5, 7) == pytest.approx(1.0)


def test_fisher_exact_degenerate_zero_column_is_near_certain():
    assert fisher_exact(0, 5, 0, 7) == pytest.approx(1.0)
