"""Tests for the drift injectors.

The property that matters is that drift lands exactly where the injector claims
it did: the signal must change at the declared points and nowhere else. If that
is not true, every detection-timing number this project reports is meaningless.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.drift_injection import (  # noqa: E402
    DRIFT_KINDS,
    inject,
    inject_gradual,
    inject_incremental,
    inject_recurring,
    inject_sudden,
)

N = 400


@pytest.fixture
def stream() -> tuple[pd.DataFrame, pd.Series]:
    """A stationary two-sensor stream with unit-ish variance and no drift."""
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {
            "sensor_a": rng.normal(10.0, 1.0, N),
            "sensor_b": rng.normal(-5.0, 2.0, N),
        }
    )
    return frame, pd.Series(np.zeros(N, dtype=int))


def test_sudden_changes_only_after_the_drift_point(stream):
    frame, labels = stream
    result = inject_sudden(frame, labels, at=0.5, magnitude=4.0)

    (point,) = result.drift_points
    assert point == 200

    # Nothing before the point may move, and everything after must.
    before = result.frame.iloc[:point]
    pd.testing.assert_frame_equal(before, frame.iloc[:point])
    assert not np.allclose(result.frame.iloc[point:], frame.iloc[point:])

    # The shift is 4 standard deviations of each column, applied flat.
    for column in frame.columns:
        offset = result.frame[column].iloc[point:] - frame[column].iloc[point:]
        assert np.allclose(offset, 4.0 * frame[column].std())

    assert result.drift_mask.sum() == N - point
    assert not result.drift_mask[:point].any()


def test_incremental_ramps_monotonically_then_holds(stream):
    frame, labels = stream
    result = inject_incremental(frame, labels, start=0.25, end=0.75, magnitude=3.0)

    (point,) = result.drift_points
    assert point == 100

    delta = (result.frame["sensor_a"] - frame["sensor_a"]).to_numpy()
    assert np.allclose(delta[:100], 0.0)

    # Strictly increasing through the ramp, then flat at full magnitude.
    ramp = delta[100:300]
    assert np.all(np.diff(ramp) > 0)
    expected_full = 3.0 * frame["sensor_a"].std()
    assert np.allclose(delta[300:], expected_full)
    assert ramp[-1] < expected_full


def test_gradual_interleaves_whole_rows_from_both_regimes(stream):
    frame, labels = stream
    result = inject_gradual(frame, labels, start=0.25, end=0.75, magnitude=3.0, seed=7)

    (point,) = result.drift_points
    assert point == 100

    delta = (result.frame["sensor_a"] - frame["sensor_a"]).to_numpy()
    expected_full = 3.0 * frame["sensor_a"].std()

    # Each row is fully in one regime or the other, never part-way between.
    assert np.all(np.isclose(delta, 0.0) | np.isclose(delta, expected_full))
    assert np.allclose(delta[:100], 0.0)
    assert np.allclose(delta[300:], expected_full)

    # Inside the window both regimes appear, and the new one grows more common.
    window = result.drift_mask[100:300]
    assert 0 < window.sum() < len(window)
    assert window[: len(window) // 2].mean() < window[len(window) // 2 :].mean()


def test_recurring_alternates_blocks_and_reports_every_boundary(stream):
    frame, labels = stream
    result = inject_recurring(frame, labels, period=0.25, magnitude=2.0)

    assert result.drift_points == [100, 200, 300]

    mask = result.drift_mask
    assert not mask[:100].any()
    assert mask[100:200].all()
    assert not mask[200:300].any()
    assert mask[300:].all()

    # The regime really does return to baseline, not just decay toward it.
    pd.testing.assert_frame_equal(result.frame.iloc[200:300], frame.iloc[200:300])


def test_recurring_truncates_a_trailing_partial_block():
    """A boundary a few rows from the end is not something a detector can be scored on."""
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"sensor_a": rng.normal(0.0, 1.0, 405)})
    labels = pd.Series(np.zeros(405, dtype=int))

    result = inject_recurring(frame, labels, period=0.25, magnitude=2.0)

    # period 0.25 of 405 rounds to a block of 101, so 4 whole blocks fit.
    assert result.n_rows == 404
    assert result.drift_points == [101, 202, 303]
    assert max(result.drift_points) < result.n_rows - 1
    assert len(result.labels) == result.n_rows


@pytest.mark.parametrize("kind", DRIFT_KINDS)
def test_every_injector_preserves_shape_labels_and_is_deterministic(kind, stream):
    frame, labels = stream
    first = inject(kind, frame, labels, magnitude=3.0)
    second = inject(kind, frame, labels, magnitude=3.0)

    assert first.kind == kind
    assert first.frame.shape == frame.shape
    assert list(first.frame.columns) == list(frame.columns)
    assert len(first.labels) == len(labels)
    assert first.drift_points, "an injector must report at least one drift point"
    assert all(0 < p < len(frame) for p in first.drift_points)

    pd.testing.assert_frame_equal(first.frame, second.frame)


@pytest.mark.parametrize("kind", DRIFT_KINDS)
def test_zero_magnitude_leaves_the_signal_untouched(kind, stream):
    frame, labels = stream
    result = inject(kind, frame, labels, magnitude=0.0)
    pd.testing.assert_frame_equal(result.frame, frame)


def test_injectors_only_touch_the_columns_they_are_given(stream):
    frame, labels = stream
    result = inject_sudden(frame, labels, magnitude=5.0, columns=["sensor_a"])

    assert not np.allclose(result.frame["sensor_a"], frame["sensor_a"])
    pd.testing.assert_series_equal(result.frame["sensor_b"], frame["sensor_b"])
    assert result.affected_columns == ["sensor_a"]


def test_constant_columns_are_left_alone(stream):
    """A column with no variance has no distribution to shift."""
    frame, labels = stream
    frame = frame.assign(flat=1.0)
    result = inject_sudden(frame, labels, magnitude=3.0)
    pd.testing.assert_series_equal(result.frame["flat"], frame["flat"])


def test_bad_arguments_raise(stream):
    frame, labels = stream

    with pytest.raises(ValueError, match="unknown drift kind"):
        inject("exponential", frame, labels)
    with pytest.raises(ValueError, match="columns not in frame"):
        inject_sudden(frame, labels, columns=["nonexistent"])
    with pytest.raises(ValueError, match="outside the stream"):
        inject_sudden(frame, labels, at=1.0)
    with pytest.raises(ValueError, match="invalid ramp window"):
        inject_incremental(frame, labels, start=0.8, end=0.3)
    with pytest.raises(ValueError, match="empty frame"):
        inject_sudden(pd.DataFrame(), labels)
