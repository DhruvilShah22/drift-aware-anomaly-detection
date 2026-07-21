"""Tests for the threshold rules.

The property that matters is **robustness to a contaminated warm-up window**,
because that is the difference between the two datasets that broke a fixed
quantile: NAB's warm-up is 0% anomalous, SKAB's valve1 warm-up is about 35%.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.thresholds import (  # noqa: E402
    ThresholdRule,
    build_rule,
    fixed_quantile,
    robust_z,
    target_rate,
    tukey,
)


def clean_scores(n: int = 1000, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(10.0, 1.0, n)


def contaminated_scores(fraction: float, n: int = 1000, seed: int = 0) -> np.ndarray:
    """Warm-up scores where `fraction` of points are shifted well above normal."""
    rng = np.random.default_rng(seed)
    scores = rng.normal(10.0, 1.0, n)
    n_bad = int(n * fraction)
    scores[:n_bad] = rng.normal(20.0, 1.0, n_bad)
    return scores


def relative_shift(rule, fraction: float) -> float:
    clean = rule(clean_scores())
    return abs(rule(contaminated_scores(fraction)) - clean) / clean


def test_robust_z_is_far_less_distorted_than_a_quantile_at_35_percent():
    """The exact situation on SKAB valve1.

    Bounds come from measurement, not from theory: at 35% contamination the
    fixed quantile shifts ~81% while robust_z shifts ~34%. robust_z is not
    unaffected — the MAD does inflate — it is merely the only rule here that
    stays in a usable range.
    """
    robust = relative_shift(robust_z(3.0), 0.35)
    quantile = relative_shift(fixed_quantile(0.98), 0.35)

    assert robust < 0.40
    assert quantile > 0.75
    assert robust < quantile / 2


def test_fixed_quantile_is_badly_distorted_by_the_same_contamination():
    """The failure the robust rule exists to avoid."""
    rule = fixed_quantile(0.98)
    clean = rule(clean_scores())
    dirty = rule(contaminated_scores(0.35))

    # The upper tail is now made of anomalies, so the threshold jumps far above
    # anything a detector should require and recall collapses.
    assert dirty > clean * 1.5
    assert (dirty - clean) > 8.0


def test_robust_z_degrades_gradually_rather_than_collapsing():
    """Measured: roughly 4%, 16%, 34%, 48% distortion at 10/25/35/40%.

    The point is the shape of that curve. A fixed quantile is already at 74%
    distortion by 10% contamination — it does not degrade, it falls over.
    """
    rule = robust_z(3.0)
    shifts = [relative_shift(rule, f) for f in (0.10, 0.25, 0.35, 0.40)]

    assert shifts == sorted(shifts), "distortion should grow with contamination"
    assert shifts[0] < 0.10, "should be nearly unaffected by light contamination"
    assert shifts[-1] < 0.55, "should still be finite approaching the breakdown point"

    quantile_at_10 = relative_shift(fixed_quantile(0.98), 0.10)
    assert quantile_at_10 > shifts[-1], (
        "a fixed quantile at 10% contamination should already be worse than "
        "robust_z at 40%"
    )


def test_tukey_breaks_down_before_robust_z_does():
    """Tukey's fence has a ~25% breakdown point, below valve1's 35%.

    Measured: at 35% contamination tukey shifts ~171% against robust_z's ~34%.
    Included because it confirms the breakdown point is what drives the choice,
    rather than robustness being a vague property.
    """
    assert relative_shift(tukey(1.5), 0.10) < 0.10, "fine while lightly contaminated"
    assert relative_shift(tukey(1.5), 0.35) > 1.0, "collapses past its breakdown point"
    assert relative_shift(robust_z(3.0), 0.35) < relative_shift(tukey(1.5), 0.35) / 3


def test_larger_k_gives_a_higher_threshold():
    scores = clean_scores()
    assert robust_z(2.0)(scores) < robust_z(3.0)(scores) < robust_z(5.0)(scores)
    assert tukey(1.5)(scores) < tukey(3.0)(scores)


def test_target_rate_flags_about_the_requested_fraction():
    scores = clean_scores(n=5000)
    threshold = target_rate(0.05)(scores)
    flagged = (scores > threshold).mean()
    assert 0.04 < flagged < 0.06


def test_target_rate_matches_the_equivalent_quantile():
    scores = clean_scores()
    assert target_rate(0.02)(scores) == pytest.approx(fixed_quantile(0.98)(scores))


@pytest.mark.parametrize(
    "rule", [fixed_quantile(0.98), robust_z(3.0), tukey(1.5), target_rate(0.02)]
)
def test_constant_warmup_scores_yield_a_usable_threshold(rule):
    """HST can produce an all-identical warm-up window; nothing may divide by zero."""
    scores = np.full(500, 0.94)
    threshold = rule(scores)
    assert np.isfinite(threshold)
    assert threshold >= 0.94


@pytest.mark.parametrize(
    "rule", [fixed_quantile(0.98), robust_z(3.0), tukey(1.5), target_rate(0.02)]
)
def test_non_finite_scores_are_ignored_not_propagated(rule):
    scores = np.concatenate([clean_scores(100), [np.nan, np.inf, -np.inf]])
    assert np.isfinite(rule(scores))


@pytest.mark.parametrize(
    "rule", [fixed_quantile(0.98), robust_z(3.0), tukey(1.5), target_rate(0.02)]
)
def test_empty_scores_raise(rule):
    with pytest.raises(ValueError, match="no finite warm-up scores"):
        rule(np.array([]))


def test_build_rule_accepts_a_bare_quantile_for_backwards_compatibility():
    rule = build_rule(0.98)
    assert isinstance(rule, ThresholdRule)
    scores = clean_scores()
    assert rule(scores) == pytest.approx(fixed_quantile(0.98)(scores))


@pytest.mark.parametrize(
    "spec,expected_name",
    [
        ("robust_z@3", "robust_z@3"),
        ("tukey@1.5", "tukey@1.5"),
        ("quantile@0.9", "quantile@0.9"),
        ("target_rate@0.05", "target_rate@0.05"),
    ],
)
def test_build_rule_parses_specs(spec, expected_name):
    assert build_rule(spec).name == expected_name


def test_build_rule_passes_through_an_existing_rule():
    rule = robust_z(4.0)
    assert build_rule(rule) is rule


def test_build_rule_rejects_nonsense():
    with pytest.raises(ValueError, match="unknown threshold rule"):
        build_rule("magic@3")
    with pytest.raises(ValueError, match="needs a parameter"):
        build_rule("robust_z")


def test_rule_parameters_are_validated():
    with pytest.raises(ValueError, match="quantile must be in"):
        fixed_quantile(1.5)
    with pytest.raises(ValueError, match="k must be positive"):
        robust_z(0)
    with pytest.raises(ValueError, match="k must be positive"):
        tukey(-1)
    with pytest.raises(ValueError, match="rate must be in"):
        target_rate(0.0)
