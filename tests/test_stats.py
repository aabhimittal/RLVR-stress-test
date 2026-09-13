import math

from cheater.stats import auc, bootstrap_ci, clamp, kendall_tau_b, mean, spearman, stdev, wilson_interval


def test_auc_undefined_with_one_class():
    assert auc([0.1, 0.9], [1, 1]) is None
    assert auc([0.1, 0.9], [0, 0]) is None


def test_auc_ranks_and_ties():
    assert auc([0.9, 0.1], [1, 0]) == 1.0
    assert auc([0.1, 0.9], [1, 0]) == 0.0
    assert auc([0.5, 0.5], [1, 0]) == 0.5  # ties are worth exactly chance


def test_spearman_and_kendall_degenerate():
    assert spearman([1, 1, 1], [1, 2, 3]) == 0.0  # constant input, not a correlation
    assert spearman([], []) == 0.0
    assert kendall_tau_b([1], [1]) == 0.0
    assert round(kendall_tau_b([1, 2, 3, 4], [1, 2, 3, 4]), 6) == 1.0


def test_bootstrap_ci_degenerate_and_bounded():
    assert bootstrap_ci([]) == (0.0, 0.0)
    assert bootstrap_ci([0.5] * 9) == (0.5, 0.5)
    lo, hi = bootstrap_ci([0.0] * 10 + [1.0] * 10, reps=400, seed=1)
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0


def test_bootstrap_is_seeded():
    assert bootstrap_ci([0, 1, 0, 1, 1], reps=200, seed=4) == bootstrap_ci([0, 1, 0, 1, 1], reps=200, seed=4)


def test_wilson_stable_at_extremes():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = wilson_interval(0, 20)
    assert lo == 0.0 and 0.0 < hi < 0.25
    lo, hi = wilson_interval(20, 20)
    assert hi == 1.0 and lo > 0.75


def test_moments():
    assert mean([]) == 0.0
    assert stdev([1.0]) == 0.0
    assert math.isclose(stdev([1, 2, 3]), 1.0)
    assert clamp(-5) == 0.0 and clamp(5) == 1.0
