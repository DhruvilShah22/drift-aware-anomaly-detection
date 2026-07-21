"""Tests for the robustness sweep's bookkeeping.

The sweep itself is slow and needs real data, so what is tested here is the
arithmetic that turns raw runs into claims — above all the attribution against
the magnitude-0.0 control, which is the thing standing between an honest result
and a flattering one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sweep import attribution_table, summarise_sweep  # noqa: E402


def make_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"kind": "sudden", "magnitude": 0.0, "detector": "adwin",
         "n_detections_total": 18, "detection_rate": 1.0, "mean_delay": 393.0,
         "false_alarms_per_1k": 2.0, "alarm_rate": 0.01},
        {"kind": "sudden", "magnitude": 3.0, "detector": "adwin",
         "n_detections_total": 22, "detection_rate": 1.0, "mean_delay": 9.0,
         "false_alarms_per_1k": 2.0, "alarm_rate": 0.09},
        {"kind": "sudden", "magnitude": 0.0, "detector": "kswin",
         "n_detections_total": 26, "detection_rate": 1.0, "mean_delay": 67.0,
         "false_alarms_per_1k": 3.0, "alarm_rate": 0.02},
        {"kind": "sudden", "magnitude": 3.0, "detector": "kswin",
         "n_detections_total": 25, "detection_rate": 1.0, "mean_delay": 13.0,
         "false_alarms_per_1k": 3.0, "alarm_rate": 0.04},
    ])


def test_excess_is_measured_against_the_matching_control():
    result = attribution_table(make_table())
    by_key = result.set_index(["kind", "magnitude", "detector"])["excess_firings"]

    assert by_key[("sudden", 3.0, "adwin")] == 22 - 18
    # KSWIN fired *fewer* times with drift injected than without. That is a real
    # outcome on this data and must survive as a negative number rather than
    # being clipped to zero, which would quietly overstate the method.
    assert by_key[("sudden", 3.0, "kswin")] == 25 - 26 == -1
    assert by_key[("sudden", 0.0, "adwin")] == 0


def test_controls_are_matched_per_detector_not_pooled():
    """Using one detector's control for another would corrupt every excess figure."""
    result = attribution_table(make_table())
    controls = result.set_index(["kind", "magnitude", "detector"])["control_firings"]
    assert controls[("sudden", 3.0, "adwin")] == 18
    assert controls[("sudden", 3.0, "kswin")] == 26


def test_attribution_refuses_a_sweep_with_no_control():
    table = make_table()
    with pytest.raises(ValueError, match="no magnitude=0.0 control"):
        attribution_table(table[table.magnitude > 0])


def test_summary_excludes_the_control_rows():
    """The control is a reference point, not a magnitude to average in."""
    summary = summarise_sweep(make_table())
    assert set(summary.magnitude) == {3.0}
    assert len(summary) == 2  # one row per detector


def test_summary_needs_something_to_summarise():
    table = make_table()
    with pytest.raises(ValueError, match="no non-zero magnitudes"):
        summarise_sweep(table[table.magnitude == 0.0])
