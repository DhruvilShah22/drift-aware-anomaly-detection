"""Tests for the published metrics in `src/metrics_lit.py`.

Most of these check hand-computed values on tiny arrays. That is deliberate: the
whole reason for implementing four published conventions is so the paper can say
which one it is using and be checked, and an implementation nobody verified by
hand would defeat the point.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluate import event_metrics
from src.metrics_lit import (
    all_metrics,
    composite_f,
    point_adjust_k_metrics,
    point_adjust_metrics,
    point_metrics,
    range_metrics,
)


# Two true events: rows 2-4 (length 3) and 7-8 (length 2). One predicted flag at
# row 3, inside the first event. Every expected number below is derived from
# these by hand in the test that uses it.
Y_TRUE = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
Y_PRED = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])


class TestPointwise:
    def test_matches_hand_count(self):
        # One true positive, no false positives, four false negatives.
        m = point_metrics(Y_TRUE, Y_PRED)
        assert m.precision == pytest.approx(1.0)
        assert m.recall == pytest.approx(1 / 5)
        assert m.f1 == pytest.approx(2 * 1.0 * 0.2 / 1.2)

    def test_all_zero_prediction_scores_zero_not_nan(self):
        m = point_metrics(Y_TRUE, np.zeros_like(Y_TRUE))
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0


class TestRangeBased:
    def test_overlap_recall_is_fraction_of_each_range_covered(self):
        # Event one: 1 of 3 rows covered. Event two: none. Recall averages the
        # per-event rewards, so (1/3 + 0) / 2.
        m = range_metrics(Y_TRUE, Y_PRED)
        assert m.recall == pytest.approx((1 / 3 + 0) / 2)

    def test_precision_is_fraction_of_each_prediction_inside_a_range(self):
        # The single predicted range sits entirely inside a true one.
        m = range_metrics(Y_TRUE, Y_PRED)
        assert m.precision == pytest.approx(1.0)

    def test_alpha_one_pays_only_for_existence(self):
        # With alpha=1 the size of the overlap stops mattering: event one was
        # touched, event two was not, so recall is exactly 1/2.
        m = range_metrics(Y_TRUE, Y_PRED, alpha=1.0)
        assert m.recall == pytest.approx(0.5)

    def test_cardinality_penalises_a_fragmented_detection(self):
        # One true event of length 5. `fragmented` covers four of its rows but
        # in two separate pieces; `whole` covers only three, in one piece. The
        # reciprocal gamma halves the fragmented reward to 0.4, putting it below
        # the unsplit 0.6 despite the better coverage.
        y_true = np.array([0, 1, 1, 1, 1, 1, 0])
        fragmented = np.array([0, 1, 1, 0, 1, 1, 0])
        whole = np.array([0, 1, 1, 1, 0, 0, 0])

        split = range_metrics(y_true, fragmented, gamma="reciprocal")
        unsplit = range_metrics(y_true, whole, gamma="reciprocal")
        assert split.recall == pytest.approx(0.4)
        assert unsplit.recall == pytest.approx(0.6)
        assert split.recall < unsplit.recall

        # Switch the penalty off and raw coverage decides instead, which
        # reverses the ranking. The whole point of gamma is that this is a
        # choice the evaluator makes, not a property of the detector.
        no_penalty = range_metrics(y_true, fragmented, gamma="one")
        assert no_penalty.recall == pytest.approx(0.8)
        assert no_penalty.recall > unsplit.recall

    def test_front_bias_rewards_catching_a_fault_early(self):
        y_true = np.array([0, 1, 1, 1, 1, 0])
        early = np.array([0, 1, 0, 0, 0, 0])
        late = np.array([0, 0, 0, 0, 1, 0])
        assert (
            range_metrics(y_true, early, bias="front").recall
            > range_metrics(y_true, late, bias="front").recall
        )
        # Flat bias is indifferent to where inside the event the hit landed.
        assert range_metrics(y_true, early, bias="flat").recall == pytest.approx(
            range_metrics(y_true, late, bias="flat").recall
        )

    def test_rejects_out_of_range_alpha(self):
        with pytest.raises(ValueError):
            range_metrics(Y_TRUE, Y_PRED, alpha=1.5)


class TestPointAdjust:
    def test_one_hit_credits_the_whole_segment(self):
        # The single flag at row 3 credits all three rows of event one, so
        # recall becomes 3/5 rather than the honest 1/5.
        m = point_adjust_metrics(Y_TRUE, Y_PRED)
        assert m.recall == pytest.approx(3 / 5)

    def test_inflates_the_score_relative_to_pointwise(self):
        plain = point_metrics(Y_TRUE, Y_PRED)
        adjusted = point_adjust_metrics(Y_TRUE, Y_PRED)
        assert adjusted.f1 > plain.f1

    def test_a_single_random_looking_flag_scores_absurdly_well(self):
        # This is the documented failure of point-adjust and the reason we
        # report it only as an upper bound: one flag per event, 3 rows of 300
        # in total, scores a recall of 1.0.
        y_true = np.zeros(300, dtype=int)
        y_true[0:100] = 1
        y_true[150:250] = 1
        y_pred = np.zeros(300, dtype=int)
        y_pred[5] = 1
        y_pred[160] = 1
        assert point_adjust_metrics(y_true, y_pred).recall == pytest.approx(1.0)
        assert point_metrics(y_true, y_pred).recall == pytest.approx(2 / 200)


class TestPointAdjustK:
    def test_k_above_the_hit_fraction_withholds_the_adjustment(self):
        # Event one is 1/3 flagged. K=0.5 is not met, so nothing is credited and
        # the score falls back to the point-wise count.
        strict = point_adjust_k_metrics(Y_TRUE, Y_PRED, k=0.5)
        assert strict.recall == pytest.approx(1 / 5)

    def test_k_below_the_hit_fraction_grants_it(self):
        lenient = point_adjust_k_metrics(Y_TRUE, Y_PRED, k=0.2)
        assert lenient.recall == pytest.approx(3 / 5)

    def test_is_monotone_decreasing_in_k(self):
        recalls = [
            point_adjust_k_metrics(Y_TRUE, Y_PRED, k=k).recall
            for k in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]
        assert recalls == sorted(recalls, reverse=True)

    def test_rejects_k_outside_the_unit_interval(self):
        with pytest.raises(ValueError):
            point_adjust_k_metrics(Y_TRUE, Y_PRED, k=2.0)


class TestCompositeF:
    def test_agrees_with_our_own_event_metric(self):
        """The claim the paper makes about `event_metrics` being the composite
        F-score of Garg et al. is checked here rather than asserted in prose."""
        rng = np.random.default_rng(11)
        for _ in range(50):
            y_true = (rng.random(400) < 0.2).astype(int)
            y_pred = (rng.random(400) < 0.3).astype(int)
            ours = event_metrics(y_true, y_pred)
            theirs = composite_f(y_true, y_pred)
            assert theirs.precision == pytest.approx(ours.precision)
            assert theirs.recall == pytest.approx(ours.recall)
            assert theirs.f1 == pytest.approx(ours.f1)


class TestOrdering:
    def test_the_conventions_disagree_on_ranking(self):
        """The paper's central claim in miniature: two detectors can swap places
        depending only on which published metric is used to score them.

        `dense` flags one fault heavily and misses the other. `sparse` touches
        both faults once each. Point-wise F1 prefers the dense one because it
        accumulates true positives; event-level and point-adjust prefer the
        sparse one because it noticed both faults.
        """
        y_true = np.zeros(600, dtype=int)
        y_true[100:200] = 1
        y_true[400:500] = 1

        dense = np.zeros(600, dtype=int)
        dense[110:190] = 1

        sparse = np.zeros(600, dtype=int)
        sparse[150] = 1
        sparse[450] = 1

        assert point_metrics(y_true, dense).f1 > point_metrics(y_true, sparse).f1
        assert composite_f(y_true, sparse).recall > composite_f(y_true, dense).recall
        assert point_adjust_metrics(y_true, sparse).f1 > point_metrics(y_true, sparse).f1


class TestAllMetrics:
    def test_returns_every_convention_with_distinct_prefixes(self):
        out = all_metrics(Y_TRUE, Y_PRED)
        for prefix in ("point_", "composite_", "range_", "range_alpha1_", "pa_", "pak_"):
            for field in ("precision", "recall", "f1"):
                assert f"{prefix}{field}" in out
        assert len(out) == 18

    def test_every_value_is_a_finite_fraction(self):
        out = all_metrics(Y_TRUE, Y_PRED)
        for key, value in out.items():
            assert 0.0 <= value <= 1.0, f"{key} out of range: {value}"

    def test_shape_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            all_metrics(np.zeros(5), np.zeros(6))
