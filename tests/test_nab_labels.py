"""Tests for expanding NAB's time-window labels into per-row flags.

NAB ships anomaly labels as [start, end] timestamp ranges rather than per-row
flags. Getting that expansion subtly wrong — off-by-one at the boundaries, or a
timezone mismatch that makes windows miss the index entirely — would produce a
plausible-looking label vector and quietly invalidate every metric computed from
it. These run offline against a hand-built index.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import DatasetError, expand_windows_to_labels  # noqa: E402


@pytest.fixture
def index() -> pd.DatetimeIndex:
    # 10 points, five minutes apart, 00:00 through 00:45.
    return pd.date_range("2014-01-01 00:00:00", periods=10, freq="5min")


def test_window_is_inclusive_at_both_ends(index):
    labels = expand_windows_to_labels(
        index, [["2014-01-01 00:10:00", "2014-01-01 00:20:00"]]
    )
    assert list(labels) == [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]


def test_multiple_windows_are_unioned(index):
    labels = expand_windows_to_labels(
        index,
        [
            ["2014-01-01 00:00:00", "2014-01-01 00:05:00"],
            ["2014-01-01 00:35:00", "2014-01-01 00:45:00"],
        ],
    )
    assert list(labels) == [1, 1, 0, 0, 0, 0, 0, 1, 1, 1]
    assert labels.sum() == 5


def test_window_edges_falling_between_samples(index):
    """NAB window bounds need not coincide with sample timestamps."""
    labels = expand_windows_to_labels(
        index, [["2014-01-01 00:07:00", "2014-01-01 00:18:00"]]
    )
    # Covers 00:10 and 00:15 only.
    assert list(labels) == [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]


def test_overlapping_windows_do_not_double_count(index):
    labels = expand_windows_to_labels(
        index,
        [
            ["2014-01-01 00:05:00", "2014-01-01 00:20:00"],
            ["2014-01-01 00:15:00", "2014-01-01 00:25:00"],
        ],
    )
    assert set(labels.unique()) <= {0, 1}
    assert list(labels) == [0, 1, 1, 1, 1, 1, 0, 0, 0, 0]


def test_no_windows_means_no_anomalies(index):
    labels = expand_windows_to_labels(index, [])
    assert labels.sum() == 0
    assert len(labels) == len(index)


def test_windows_that_miss_the_index_entirely_raise(index):
    """The failure that would look like a clean series instead of a loading bug."""
    with pytest.raises(DatasetError, match="none overlap its timestamps"):
        expand_windows_to_labels(
            index, [["2020-06-01 00:00:00", "2020-06-02 00:00:00"]], source="fake.csv"
        )


def test_malformed_window_raises(index):
    with pytest.raises(DatasetError, match="malformed label window"):
        expand_windows_to_labels(index, [["2014-01-01 00:00:00"]])


def test_reversed_window_raises(index):
    with pytest.raises(DatasetError, match="reversed label window"):
        expand_windows_to_labels(
            index, [["2014-01-01 00:30:00", "2014-01-01 00:10:00"]]
        )


def test_labels_align_with_the_given_index(index):
    labels = expand_windows_to_labels(
        index, [["2014-01-01 00:10:00", "2014-01-01 00:20:00"]]
    )
    pd.testing.assert_index_equal(labels.index, index)
    assert labels.dtype == int
