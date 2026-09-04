"""The multi-seed harness: does it actually vary anything, and pair correctly?

These run on a small generated stream rather than SKAB, so they work without the
downloaded dataset and finish in seconds. What they check is the part that is
easy to get wrong and invisible when it is: that a seed reaches the model, that
every strategy at a given seed is compared against the same stream, and that the
per-recording rows line up with the segments they came from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experiment import RunConfig, Scenario, Strategy
from src.multiseed import labelled_seed_runs

SEEDS = (0, 1, 2)


def make_scenario(n: int = 2400, n_segments: int = 3) -> Scenario:
    """A stream with three equal recordings, each carrying one fault block."""
    rng = np.random.default_rng(12)
    columns = [f"sensor_{i}" for i in range(4)]
    values = rng.normal(0.0, 1.0, size=(n, 4))
    labels = np.zeros(n, dtype=int)

    segments = []
    span = n // n_segments
    for k in range(n_segments):
        start = k * span
        segments.append((start, start + span, f"{k}.csv"))
        # One fault per recording, placed past the warm-up so it is scored.
        fault_start = start + int(span * 0.7)
        fault_end = fault_start + 120
        values[fault_start:fault_end] += 4.0
        labels[fault_start:fault_end] = 1

    return Scenario(
        name="synthetic",
        frame=pd.DataFrame(values, columns=columns),
        labels=pd.Series(labels),
        change_points=[],
        has_true_anomalies=True,
        threshold_quantile=0.90,
        segments=segments,
    )


STRATEGIES = [
    Strategy("hst-static", model_kind="hst", adapt="never"),
    Strategy("hst-periodic", model_kind="hst", adapt="periodic", retrain_every=500),
    Strategy("iforest-static", model_kind="iforest", adapt="never"),
    Strategy("lof-static", model_kind="lof", adapt="never"),
]


@pytest.fixture(scope="module")
def runs():
    scenario = make_scenario()
    config = RunConfig(warmup_size=400, threshold_quantile=0.90)
    stream, segments = labelled_seed_runs(
        scenario, STRATEGIES, seeds=SEEDS, config=config, verbose=False
    )
    return stream, segments


class TestStreamRows:
    def test_one_row_per_strategy_per_seed(self, runs):
        stream, _ = runs
        assert len(stream) == len(STRATEGIES) * len(SEEDS)
        counts = stream.groupby("seed").strategy.nunique()
        assert set(counts) == {len(STRATEGIES)}

    def test_carries_every_scoring_convention(self, runs):
        stream, _ = runs
        for prefix in ("point_", "composite_", "range_", "pa_", "pak_"):
            assert f"{prefix}f1" in stream.columns

    def test_records_the_model_and_policy_factors(self, runs):
        stream, _ = runs
        row = stream[stream.strategy == "hst-periodic"].iloc[0]
        assert row.model == "hst"
        assert row.policy == "periodic"

    def test_baseline_and_lift_are_consistent(self, runs):
        stream, _ = runs
        expected = stream.composite_f1 - stream.flag_everything_f1
        np.testing.assert_allclose(stream.composite_lift, expected)

    def test_flag_everything_matches_the_closed_form(self, runs):
        stream, _ = runs
        b = stream.base_rate
        np.testing.assert_allclose(stream.flag_everything_f1, 2 * b / (1 + b))

    def test_every_metric_is_a_fraction(self, runs):
        stream, _ = runs
        for column in stream.columns:
            if column.endswith(("_f1", "_precision", "_recall")):
                assert stream[column].between(0, 1).all(), column


class TestSeedsActuallyVary:
    def test_a_seeded_model_moves_between_seeds(self, runs):
        """If the seed never reached the model, every interval in the paper
        would be zero-width and the whole exercise would be theatre."""
        stream, _ = runs
        hst = stream[stream.strategy == "hst-static"]
        assert hst.point_f1.nunique() > 1

    def test_a_deterministic_model_does_not(self, runs):
        """LOF has no randomness, so its spread across seeds must be exactly
        zero. The paper reports that as a degenerate interval rather than
        pretending it is a measurement."""
        stream, _ = runs
        lof = stream[stream.strategy == "lof-static"]
        assert lof.point_f1.nunique() == 1

    def test_strategies_within_a_seed_saw_the_same_stream(self, runs):
        stream, _ = runs
        for _, block in stream.groupby("seed"):
            assert block.base_rate.nunique() == 1
            assert block.n_steps.nunique() == 1


class TestSegmentRows:
    def test_one_row_per_segment_per_strategy_per_seed(self, runs):
        _, segments = runs
        assert not segments.empty
        per_run = segments.groupby(["seed", "strategy"]).size()
        assert per_run.nunique() == 1

    def test_segments_are_named_after_their_recording(self, runs):
        _, segments = runs
        assert set(segments.segment) <= {"0.csv", "1.csv", "2.csv"}

    def test_a_segment_with_no_fault_is_dropped(self):
        """Scoring an F1 on a recording containing no anomaly would be zero for
        everyone and would drag every mean toward zero for no reason."""
        scenario = make_scenario()
        labels = scenario.labels.to_numpy().copy()
        labels[1600:] = 0  # wipe the fault out of the last recording
        scenario = Scenario(
            name=scenario.name,
            frame=scenario.frame,
            labels=pd.Series(labels),
            change_points=[],
            has_true_anomalies=True,
            threshold_quantile=0.90,
            segments=scenario.segments,
        )
        _, segments = labelled_seed_runs(
            scenario, STRATEGIES[:1], seeds=(0,),
            config=RunConfig(warmup_size=400, threshold_quantile=0.90),
            verbose=False,
        )
        assert "2.csv" not in set(segments.segment)

    def test_no_segments_means_an_empty_frame(self):
        scenario = make_scenario()
        scenario = Scenario(
            name=scenario.name, frame=scenario.frame, labels=scenario.labels,
            change_points=[], has_true_anomalies=True,
            threshold_quantile=0.90, segments=[],
        )
        _, segments = labelled_seed_runs(
            scenario, STRATEGIES[:1], seeds=(0,),
            config=RunConfig(warmup_size=400, threshold_quantile=0.90),
            verbose=False,
        )
        assert segments.empty


class TestPairingIsUsable:
    def test_paired_comparison_runs_on_the_output(self, runs):
        """The whole point of the harness is that `src.stats` can consume it."""
        from src.stats import paired_compare

        stream, _ = runs
        result = paired_compare(stream, "composite_f1", "hst-periodic", "hst-static")
        assert result.n_pairs == len(SEEDS)
        # Three seeds cannot reach significance under the signed-rank test.
        assert result.underpowered
