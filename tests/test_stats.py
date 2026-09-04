"""Tests for the aggregation and significance machinery in `src/stats.py`.

The point of these is that the statistics are the part of the paper a reader is
least able to check by eye, so they are checked here against constructed data
where the right answer is known in advance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.stats import (
    MIN_PAIRS_FOR_WILCOXON,
    bootstrap_ci,
    compare_all_pairs,
    holm_bonferroni,
    paired_compare,
    summarise_seeds,
)


def make_frame(effect: float, n_seeds: int = 12, noise: float = 0.01, seed: int = 0):
    """Two strategies over shared seeds, where `a` beats `b` by `effect`.

    The per-seed baseline moves a lot more than the effect does, which is the
    situation pairing exists for and the situation this project is actually in.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_seeds):
        base = rng.normal(0.5, 0.08)
        rows.append({"seed": s, "strategy": "a", "f1": base + effect + rng.normal(0, noise)})
        rows.append({"seed": s, "strategy": "b", "f1": base + rng.normal(0, noise)})
    return pd.DataFrame(rows)


class TestBootstrapCI:
    def test_interval_brackets_the_mean(self):
        interval = bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5])
        assert interval.low <= interval.mean <= interval.high
        assert interval.mean == pytest.approx(0.3)
        assert interval.n == 5

    def test_constant_input_gives_a_zero_width_interval(self):
        # Deterministic estimators such as LOF produce exactly this, and the
        # paper has to be able to say so rather than crash on it.
        interval = bootstrap_ci([0.42] * 8)
        assert interval.low == pytest.approx(0.42)
        assert interval.high == pytest.approx(0.42)
        assert interval.sd == pytest.approx(0.0)

    def test_single_value_is_its_own_interval(self):
        interval = bootstrap_ci([0.7])
        assert (interval.mean, interval.low, interval.high) == (0.7, 0.7, 0.7)

    def test_empty_input_is_nan_not_an_exception(self):
        interval = bootstrap_ci([])
        assert np.isnan(interval.mean)
        assert interval.n == 0

    def test_nans_are_dropped_rather_than_propagated(self):
        interval = bootstrap_ci([0.2, np.nan, 0.4])
        assert interval.n == 2
        assert interval.mean == pytest.approx(0.3)

    def test_wider_spread_gives_a_wider_interval(self):
        tight = bootstrap_ci(np.full(20, 0.5) + np.random.default_rng(1).normal(0, 0.01, 20))
        loose = bootstrap_ci(np.full(20, 0.5) + np.random.default_rng(1).normal(0, 0.10, 20))
        assert (loose.high - loose.low) > (tight.high - tight.low)


class TestPairedCompare:
    def test_detects_a_real_effect_hidden_by_between_seed_spread(self):
        frame = make_frame(effect=0.04)
        result = paired_compare(frame, "f1", "a", "b")
        assert result.diff.mean == pytest.approx(0.04, abs=0.01)
        assert result.diff.low > 0
        assert result.significant

    def test_reports_no_effect_when_there_is_none(self):
        frame = make_frame(effect=0.0)
        result = paired_compare(frame, "f1", "a", "b")
        assert not result.significant
        assert result.diff.low < 0 < result.diff.high

    def test_sign_follows_the_argument_order(self):
        frame = make_frame(effect=0.04)
        forward = paired_compare(frame, "f1", "a", "b")
        backward = paired_compare(frame, "f1", "b", "a")
        assert forward.diff.mean == pytest.approx(-backward.diff.mean)

    def test_identical_strategies_give_p_of_one_rather_than_an_error(self):
        rows = []
        for s in range(8):
            rows.append({"seed": s, "strategy": "a", "f1": 0.5})
            rows.append({"seed": s, "strategy": "b", "f1": 0.5})
        result = paired_compare(pd.DataFrame(rows), "f1", "a", "b")
        assert result.p_value == 1.0
        assert not result.significant

    def test_small_samples_are_flagged_underpowered_and_never_significant(self):
        # Four seeds cannot reach p < 0.05 under the signed-rank test no matter
        # how large the effect, so a "not significant" verdict here says nothing
        # about the effect and the flag has to say so.
        frame = make_frame(effect=0.5, n_seeds=4)
        result = paired_compare(frame, "f1", "a", "b")
        assert result.n_pairs < MIN_PAIRS_FOR_WILCOXON
        assert result.underpowered
        assert not result.significant

    def test_minimum_pairs_constant_is_six(self):
        assert MIN_PAIRS_FOR_WILCOXON == 6

    def test_unshared_seeds_are_dropped_not_misaligned(self):
        frame = make_frame(effect=0.04, n_seeds=10)
        frame = frame[~((frame.strategy == "b") & (frame.seed == 3))]
        result = paired_compare(frame, "f1", "a", "b")
        assert result.n_pairs == 9

    def test_missing_column_names_the_column(self):
        with pytest.raises(KeyError, match="nonexistent"):
            paired_compare(make_frame(0.0), "nonexistent", "a", "b")

    def test_duplicate_pairing_labels_are_refused(self):
        """Pairing on a label that appears more than once per strategy would
        align as a cross product and invent pairs. The per-recording frame has
        exactly that shape until the seeds are averaged away, so this has to
        fail loudly instead of returning a plausible-looking number."""
        rows = []
        for recording in ("a", "b", "c"):
            for seed in range(4):
                rows.append({"segment": recording, "seed": seed, "strategy": "x", "f1": 0.5})
                rows.append({"segment": recording, "seed": seed, "strategy": "y", "f1": 0.4})
        frame = pd.DataFrame(rows)

        with pytest.raises(ValueError, match="more than one row per segment"):
            paired_compare(frame, "f1", "x", "y", pair_on="segment")

    def test_aggregating_first_makes_the_pairing_valid(self):
        rows = []
        for recording in ("a", "b", "c"):
            for seed in range(4):
                rows.append({"segment": recording, "seed": seed, "strategy": "x", "f1": 0.5})
                rows.append({"segment": recording, "seed": seed, "strategy": "y", "f1": 0.4})
        collapsed = (
            pd.DataFrame(rows)
            .groupby(["strategy", "segment"], as_index=False)
            .f1.mean()
        )
        result = paired_compare(collapsed, "f1", "x", "y", pair_on="segment")
        assert result.n_pairs == 3
        assert result.diff.mean == pytest.approx(0.1)


class TestSummariseSeeds:
    def test_produces_one_row_per_group_with_intervals(self):
        summary = summarise_seeds(make_frame(0.04), ["f1"], ["strategy"])
        assert len(summary) == 2
        assert set(summary.columns) >= {"f1", "f1_mean", "f1_ci_low", "f1_ci_high", "n_seeds"}
        assert (summary.n_seeds == 12).all()

    def test_bare_metric_column_still_holds_the_mean(self):
        # Existing plotting code reads `f1`; it must keep working.
        summary = summarise_seeds(make_frame(0.04), ["f1"], ["strategy"])
        assert (summary["f1"] == summary["f1_mean"]).all()

    def test_unknown_metric_is_rejected_up_front(self):
        with pytest.raises(KeyError):
            summarise_seeds(make_frame(0.0), ["not_a_metric"], ["strategy"])


class TestCompareAllPairs:
    def test_covers_every_unordered_pair(self):
        # A third strategy has to be one row per seed like the others. Relabelling
        # the whole frame would give it two rows per seed, which the pairing
        # guard now rejects.
        frame = make_frame(0.04)
        third = frame[frame.strategy == "a"].assign(strategy="c", f1=lambda d: d.f1 - 0.02)
        frame = pd.concat([frame, third], ignore_index=True)
        pairs = compare_all_pairs(frame, "f1")
        assert len(pairs) == 3
        assert set(zip(pairs.strategy_a, pairs.strategy_b)) == {("a", "b"), ("a", "c"), ("b", "c")}

    def test_within_splits_the_comparison_per_stream(self):
        frame = make_frame(0.04).assign(stream="one")
        other = make_frame(0.0, seed=5).assign(stream="two")
        pairs = compare_all_pairs(
            pd.concat([frame, other], ignore_index=True), "f1", within=["stream"]
        )
        assert set(pairs.stream) == {"one", "two"}
        assert pairs[pairs.stream == "one"].iloc[0].significant
        assert not pairs[pairs.stream == "two"].iloc[0].significant


class TestHolmBonferroni:
    def test_rejects_only_the_smallest_when_the_rest_are_large(self):
        assert list(holm_bonferroni([0.001, 0.04, 0.6])) == [True, False, False]

    def test_rejects_all_when_every_p_is_tiny(self):
        assert list(holm_bonferroni([0.0001, 0.0002, 0.0003])) == [True, True, True]

    def test_is_stricter_than_an_uncorrected_threshold(self):
        # p = 0.04 clears a bare alpha of 0.05 and should not survive correction
        # across ten tests.
        p_values = [0.04] + [0.9] * 9
        assert not holm_bonferroni(p_values)[0]

    def test_is_at_least_as_powerful_as_bonferroni(self):
        p_values = np.array([0.004, 0.02, 0.03, 0.5])
        holm = holm_bonferroni(p_values)
        bonferroni = p_values <= 0.05 / len(p_values)
        assert holm.sum() >= bonferroni.sum()

    def test_empty_input_returns_empty(self):
        assert holm_bonferroni([]).size == 0

    def test_order_of_the_output_matches_the_input(self):
        # The largest p is first here; the decision array must not be sorted.
        assert list(holm_bonferroni([0.6, 0.001])) == [False, True]
