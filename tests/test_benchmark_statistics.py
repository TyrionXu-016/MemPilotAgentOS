import pytest

from agentos.benchmark.statistics import exact_mcnemar, holm_adjust, wilson_interval


def test_wilson_interval_matches_known_proportion():
    low, high = wilson_interval(successes=50, total=100)

    assert low == pytest.approx(0.4038, abs=0.0001)
    assert high == pytest.approx(0.5962, abs=0.0001)


def test_exact_mcnemar_uses_two_sided_binomial_probability():
    assert exact_mcnemar(b=10, c=0) == pytest.approx(0.001953125)


def test_holm_adjust_preserves_original_order():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
