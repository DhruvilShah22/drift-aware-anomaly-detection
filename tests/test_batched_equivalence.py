"""The batched scoring path must agree with the row-by-row loop exactly.

`run_strategy` has two implementations. The streaming loop scores one row at a
time and is the obvious, plainly-correct one. The batched path finds the
adaptation points first and then scores each inter-adaptation segment in a
single vectorised call, which is what makes the multi-seed grid finish in
minutes rather than hours.

The batched path is only sound because nothing in the adaptation schedule
depends on the model's own scores: `periodic` runs off a fixed clock and the
drift monitor watches the input's distance from the reference window. That is an
argument, and arguments about equivalence are exactly the kind that quietly stop
holding after an edit, so it is tested rather than trusted.

These tests run on generated data instead of SKAB so they work without the
downloaded dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.experiment as experiment
from src.experiment import RunConfig, Strategy, run_strategy


CONFIG = RunConfig(warmup_size=400, threshold_quantile=0.95, seed=3)


@pytest.fixture(scope="module")
def stream() -> pd.DataFrame:
    """A 1,600-row, 4-sensor stream with a level shift two thirds of the way in.

    The shift is there so the drift-triggered strategy actually fires and the
    comparison covers the interesting path rather than a stream where nothing
    ever adapts. Kept small on purpose: the slow path scores sklearn estimators
    one row at a time, so every extra row costs about a millisecond twice over.
    """
    rng = np.random.default_rng(7)
    columns = [f"sensor_{i}" for i in range(4)]
    normal = rng.normal(0.0, 1.0, size=(1100, 4))
    shifted = rng.normal(2.5, 1.0, size=(500, 4))
    return pd.DataFrame(np.vstack([normal, shifted]), columns=columns)


def _run_both(stream: pd.DataFrame, strategy: Strategy, config: RunConfig):
    original = experiment.BATCHED_SCORING
    try:
        experiment.BATCHED_SCORING = True
        fast = run_strategy(stream, strategy, config)
        experiment.BATCHED_SCORING = False
        slow = run_strategy(stream, strategy, config)
    finally:
        experiment.BATCHED_SCORING = original
    return fast, slow


@pytest.fixture(scope="module")
def both_paths(stream):
    """Run one strategy down each path, memoised across the whole module.

    Each strategy is checked on five separate properties, and the slow path is
    slow enough that running it once per assertion turned this file into the
    longest thing in the suite. The cache is keyed on the strategy name.
    """
    cache: dict[str, tuple] = {}

    def run(strategy: Strategy, config: RunConfig | None = None):
        config = config or CONFIG
        key = f"{strategy.name}:{config.warmup_size}:{config.seed}"
        if key not in cache:
            cache[key] = _run_both(stream, strategy, config)
        return cache[key]

    return run


STRATEGIES = [
    Strategy("iforest-static", model_kind="iforest", adapt="never"),
    Strategy("iforest-periodic", model_kind="iforest", adapt="periodic", retrain_every=400),
    Strategy("iforest-drift", model_kind="iforest", adapt="drift"),
    Strategy("hst-static", model_kind="hst", adapt="never"),
    Strategy("hst-periodic", model_kind="hst", adapt="periodic", retrain_every=400),
    Strategy("hst-drift", model_kind="hst", adapt="drift"),
    Strategy("lof-static", model_kind="lof", adapt="never"),
    Strategy("elliptic-static", model_kind="elliptic", adapt="never"),
]


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.name)
class TestPathsAgree:
    def test_flags_are_identical(self, strategy, both_paths):
        fast, slow = both_paths(strategy)
        np.testing.assert_array_equal(fast.flags, slow.flags)

    def test_scores_are_identical(self, strategy, both_paths):
        fast, slow = both_paths(strategy)
        np.testing.assert_allclose(fast.scores, slow.scores, equal_nan=True)

    def test_thresholds_are_identical(self, strategy, both_paths):
        fast, slow = both_paths(strategy)
        np.testing.assert_allclose(fast.thresholds, slow.thresholds, equal_nan=True)

    def test_adaptation_points_are_identical(self, strategy, both_paths):
        fast, slow = both_paths(strategy)
        assert fast.adaptations == slow.adaptations

    def test_detection_points_are_identical(self, strategy, both_paths):
        fast, slow = both_paths(strategy)
        assert fast.detections == slow.detections


class TestTheComparisonIsNotVacuous:
    """If the drift and periodic strategies never adapted, the tests above would
    pass for an uninteresting reason. These pin down that they do."""

    def test_drift_strategy_actually_adapts(self, both_paths):
        fast, _ = both_paths(Strategy("d", model_kind="iforest", adapt="drift"))
        assert fast.n_adaptations > 0

    def test_periodic_strategy_actually_adapts(self, both_paths):
        fast, _ = both_paths(
            Strategy("p", model_kind="iforest", adapt="periodic", retrain_every=400)
        )
        assert fast.n_adaptations > 0

    def test_some_rows_are_flagged_and_some_are_not(self, both_paths):
        fast, _ = both_paths(Strategy("s", model_kind="iforest", adapt="never"))
        scored = fast.flags[fast.eval_start :]
        assert 0 < scored.sum() < scored.size


class TestOnlineControlIsUnaffected:
    def test_online_strategy_ignores_the_flag_entirely(self, stream):
        """`hst-online` learns between rows, so it can only take the slow path.
        Flipping the flag must make no difference to it at all."""
        strategy = Strategy("online", model_kind="hst", adapt="never", learn_online=True)
        with_flag, without_flag = _run_both(stream, strategy, CONFIG)
        np.testing.assert_array_equal(with_flag.flags, without_flag.flags)


class TestRebuildWindow:
    def test_rebuild_uses_the_window_ending_at_the_adaptation_row(self, both_paths):
        """The streaming loop appends the current row to its window before the
        adaptation check, so a rebuild at row i trains on rows (i-w, i]. Getting
        this off by one would shift every threshold after the first rebuild, and
        the equivalence tests above would catch it — this names the reason."""
        fast, slow = both_paths(
            Strategy("p", model_kind="iforest", adapt="periodic", retrain_every=400)
        )

        # Thresholds change only at rebuilds, so comparing the distinct values
        # in order checks the rebuild windows without depending on any one row.
        def distinct_in_order(values):
            out = []
            for v in values:
                if not out or not np.isclose(out[-1], v):
                    out.append(v)
            return out

        fast_steps = distinct_in_order(fast.thresholds[fast.eval_start :])
        slow_steps = distinct_in_order(slow.thresholds[slow.eval_start :])
        assert len(fast_steps) > 1
        np.testing.assert_allclose(fast_steps, slow_steps)
