"""Tests for the metrics.

Every case here has a hand-computable answer written into the test, so the
metrics are checked against arithmetic rather than against their own output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import (  # noqa: E402
    anomaly_metrics,
    drift_metrics,
    event_metrics,
    results_table,
    rolling_f1,
    summarise,
    to_events,
)


def test_anomaly_metrics_against_a_hand_counted_example():
    #        idx: 0  1  2  3  4  5  6  7
    y_true = [0, 1, 1, 0, 1, 0, 0, 1]
    y_pred = [0, 1, 0, 1, 1, 0, 1, 0]
    # tp at 1 and 4 -> 2; fp at 3 and 6 -> 2; fn at 2 and 7 -> 2
    m = anomaly_metrics(y_true, y_pred)

    assert (m.true_positives, m.false_positives, m.false_negatives) == (2, 2, 2)
    assert m.precision == 0.5
    assert m.recall == 0.5
    assert m.f1 == 0.5
    assert m.n_flagged == 4
    assert m.n_actual == 4


def test_perfect_and_inverted_predictions():
    y_true = [0, 1, 1, 0]
    assert anomaly_metrics(y_true, y_true).f1 == 1.0
    assert anomaly_metrics(y_true, [1, 0, 0, 1]).f1 == 0.0


def test_silent_predictor_scores_zero_rather_than_undefined():
    """A strategy that never flags anything must score badly, not drop out."""
    m = anomaly_metrics([0, 1, 1, 0], [0, 0, 0, 0])
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0


def test_no_anomalies_to_find():
    m = anomaly_metrics([0, 0, 0], [0, 1, 0])
    assert m.recall == 0.0
    assert m.precision == 0.0
    assert m.false_positives == 1


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        anomaly_metrics([0, 1], [0, 1, 1])


def test_rolling_f1_tracks_a_model_that_goes_bad_halfway():
    # Perfect for the first half, then flags nothing at all.
    y_true = [1, 0] * 100
    y_pred = [1, 0] * 50 + [0, 0] * 50

    series = rolling_f1(y_true, y_pred, window=20)

    assert len(series) == len(y_true)
    assert series[90] == 1.0, "should be perfect while predictions are correct"
    assert series[-1] == 0.0, "should have collapsed once predictions stopped"
    assert series[110] < series[90], "decay must be visible shortly after the change"


def test_rolling_f1_rejects_a_bad_window():
    with pytest.raises(ValueError, match="window must be at least 1"):
        rolling_f1([0, 1], [0, 1], window=0)


def test_to_events_groups_contiguous_runs():
    #      idx: 0  1  2  3  4  5  6  7  8
    labels = [0, 1, 1, 0, 0, 1, 0, 1, 1]
    # runs of 1s: rows 1-2, row 5, rows 7-8 -> half-open ranges
    assert to_events(labels) == [(1, 3), (5, 6), (7, 9)]


def test_to_events_on_edges_and_empties():
    assert to_events([1, 1, 1]) == [(0, 3)]  # runs at the very start and end
    assert to_events([0, 0, 0]) == []
    assert to_events([]) == []


def test_event_recall_credits_a_long_block_caught_by_a_single_flag():
    """The whole point: one flag inside a fault block detects the event in full."""
    #       idx: 0  1  2  3  4  5  6  7  8  9
    y_true = [0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  # one event spanning rows 1-5
    y_pred = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  # a single flag at row 3, inside it

    m = event_metrics(y_true, y_pred)
    # Point-wise recall would be 1/5 = 0.2; event recall is 1/1 = 1.0.
    assert m.recall == 1.0
    assert m.n_true_events == 1
    assert m.n_detected_events == 1
    # Precision stays point-wise: the one flag is a hit, so 1/1.
    assert m.precision == 1.0
    assert m.f1 == 1.0
    # Contrast with the point-wise metric on the same arrays.
    assert anomaly_metrics(y_true, y_pred).recall == pytest.approx(0.2)


def test_event_precision_is_the_volume_penalty():
    #       idx: 0  1  2  3  4  5  6  7  8  9
    y_true = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  # one event, rows 2-3
    y_pred = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # flag everything

    m = event_metrics(y_true, y_pred)
    # Every event is trivially caught, so recall is perfect...
    assert m.recall == 1.0
    # ...but precision falls to the base rate: 2 of 10 flags land on a fault.
    assert m.precision == pytest.approx(0.2)
    # f1 = 2 * 0.2 * 1.0 / (1.2) = 1/3, the flag-everything ceiling here.
    assert m.f1 == pytest.approx(1 / 3)


def test_event_metrics_missed_event_and_spurious_flags():
    #       idx: 0  1  2  3  4  5  6  7
    y_true = [0, 1, 1, 0, 0, 1, 1, 0]  # two events: rows 1-2 and 5-6
    y_pred = [1, 1, 0, 0, 0, 0, 0, 0]  # catches the first, misses the second

    m = event_metrics(y_true, y_pred)
    assert m.n_true_events == 2
    assert m.n_detected_events == 1
    assert m.recall == 0.5
    # Two flags raised, one lands in event 1 -> precision 1/2.
    assert m.precision == 0.5
    assert m.f1 == 0.5


def test_event_tolerance_credits_an_early_flag():
    #       idx: 0  1  2  3  4  5
    y_true = [0, 0, 0, 1, 1, 0]  # event at rows 3-4
    y_pred = [0, 1, 0, 0, 0, 0]  # flag two steps before onset

    strict = event_metrics(y_true, y_pred, tolerance=0)
    assert strict.recall == 0.0
    assert strict.precision == 0.0  # the early flag is a miss and a false alarm

    lenient = event_metrics(y_true, y_pred, tolerance=2)
    assert lenient.recall == 1.0  # now within the front-extended window
    assert lenient.precision == 1.0


def test_event_metrics_degenerate_cases():
    # No true events: recall is 0 rather than undefined, matching anomaly_metrics.
    assert event_metrics([0, 0, 0], [0, 1, 0]).recall == 0.0
    # Silent predictor: no hits anywhere.
    silent = event_metrics([0, 1, 1, 0], [0, 0, 0, 0])
    assert silent.recall == 0.0
    assert silent.precision == 0.0
    assert silent.f1 == 0.0


def test_event_metrics_rejects_bad_input():
    with pytest.raises(ValueError, match="shape mismatch"):
        event_metrics([0, 1], [0, 1, 1])
    with pytest.raises(ValueError, match="tolerance must be non-negative"):
        event_metrics([0, 1], [0, 1], tolerance=-1)


def test_drift_timing_on_an_exact_example():
    # Change points at 100 and 500; detections 10 and 25 steps later.
    m = drift_metrics(detections=[110, 525], change_points=[100, 500], n_steps=1000)

    assert m.n_change_points == 2
    assert m.n_detected == 2
    assert m.n_missed == 0
    assert m.delays == (10, 25)
    assert m.mean_delay == 17.5
    assert m.median_delay == 17.5
    assert m.n_false_alarms == 0
    assert m.detection_rate == 1.0


def test_detections_before_any_change_point_are_false_alarms():
    m = drift_metrics(detections=[50, 110], change_points=[100], n_steps=1000)
    assert m.n_detected == 1
    assert m.delays == (10,)
    assert m.n_false_alarms == 1
    assert m.false_alarms_per_1k == 1.0


def test_a_detection_beyond_the_horizon_is_a_miss_and_a_false_alarm():
    """Credit is not given for noticing a change 900 steps late."""
    m = drift_metrics(
        detections=[1000], change_points=[100], n_steps=2000, horizon=750
    )
    assert m.n_detected == 0
    assert m.n_missed == 1
    assert m.n_false_alarms == 1
    assert np.isnan(m.mean_delay)


def test_a_detection_after_the_next_change_point_is_not_credited_to_the_first():
    """Once the next change has happened, the cause of a detection is ambiguous."""
    m = drift_metrics(
        detections=[250], change_points=[100, 200], n_steps=1000, horizon=750
    )
    # 250 is within 750 of change point 100, but 200 has already occurred, so it
    # is credited to 200 instead.
    assert m.n_detected == 1
    assert m.delays == (50,)
    assert m.n_missed == 1
    assert m.n_false_alarms == 0


def test_each_detection_is_credited_at_most_once():
    m = drift_metrics(
        detections=[105], change_points=[100, 800], n_steps=2000, horizon=750
    )
    assert m.n_detected == 1
    assert m.n_missed == 1
    assert m.n_false_alarms == 0


def test_a_silent_detector_misses_everything_without_false_alarms():
    m = drift_metrics(detections=[], change_points=[100, 500], n_steps=1000)
    assert m.n_detected == 0
    assert m.n_missed == 2
    assert m.n_false_alarms == 0
    assert m.detection_rate == 0.0
    assert np.isnan(m.mean_delay)


def test_false_alarm_rate_is_normalised_by_stream_length():
    """The same three spurious detections must cost more on a shorter stream."""
    short = drift_metrics(detections=[1, 2, 3], change_points=[], n_steps=1000)
    long = drift_metrics(detections=[1, 2, 3], change_points=[], n_steps=6000)

    assert short.false_alarms_per_1k == 3.0
    assert long.false_alarms_per_1k == 0.5
    assert np.isnan(short.detection_rate), "no change points means no rate to report"


def test_drift_metrics_rejects_a_bad_stream_length():
    with pytest.raises(ValueError, match="n_steps must be positive"):
        drift_metrics(detections=[], change_points=[], n_steps=0)


def test_summarise_and_table_assembly():
    anomaly = anomaly_metrics([0, 1, 1, 0], [0, 1, 1, 0])
    drift = drift_metrics(detections=[110], change_points=[100], n_steps=1000)

    row = summarise("drift-triggered", "sudden", anomaly, drift, n_adaptations=3)
    assert row["strategy"] == "drift-triggered"
    assert row["stream"] == "sudden"
    assert row["f1"] == 1.0
    assert row["mean_delay"] == 10.0
    assert row["n_adaptations"] == 3

    table = results_table([row, summarise("static", "sudden", anomaly)])
    assert len(table) == 2
    assert set(table["strategy"]) == {"drift-triggered", "static"}


def test_empty_results_table_raises():
    with pytest.raises(ValueError, match="no result rows"):
        results_table([])
