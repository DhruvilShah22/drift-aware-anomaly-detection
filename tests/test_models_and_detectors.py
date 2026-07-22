"""Tests for the anomaly scorers and drift monitors.

These check contract and behaviour on small synthetic streams: that both models
honour the shared interface, that the static baseline really does not adapt,
that the online model does, and that the drift monitors fire on a real shift
while respecting their cooldown.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detectors import DETECTOR_KINDS, build_detector  # noqa: E402
from src.models import (  # noqa: E402
    OnlineHalfSpaceTrees,
    StaticIsolationForest,
    _feature_limits,
    build_model,
)


@pytest.fixture
def warmup() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "a": rng.normal(0.0, 1.0, 600),
            "b": rng.normal(20.0, 5.0, 600),
        }
    )


def _row(a: float, b: float) -> dict[str, float]:
    return {"a": a, "b": b}


def test_feature_limits_widen_the_observed_range(warmup):
    limits = _feature_limits(warmup, margin=0.1)
    low, high = limits["a"]
    assert low < warmup["a"].min()
    assert high > warmup["a"].max()


def test_feature_limits_handle_a_constant_column():
    frame = pd.DataFrame({"flat": np.ones(10)})
    low, high = _feature_limits(frame)["flat"]
    assert low < high, "a constant column still needs a splittable band"


@pytest.mark.parametrize("kind", ["hst", "iforest"])
def test_both_models_honour_the_shared_interface(kind, warmup):
    model = build_model(kind)
    model.warm_up(warmup)

    x = _row(0.0, 20.0)
    assert isinstance(model.score_one(x), float)
    assert model.predict_one(x) in (0, 1)
    assert np.isfinite(model.threshold)

    scores = model.score_many(warmup.head(50))
    assert scores.shape == (50,)
    assert np.all(np.isfinite(scores))

    model.learn_one(x)  # must not raise for either model


@pytest.mark.parametrize("kind", ["hst", "iforest"])
def test_scoring_before_warm_up_raises(kind):
    with pytest.raises(RuntimeError, match="warm_up must be called"):
        build_model(kind).score_one(_row(0.0, 20.0))


@pytest.mark.parametrize("kind", ["hst", "iforest"])
def test_warm_up_on_an_empty_frame_raises(kind):
    with pytest.raises(ValueError, match="empty frame"):
        build_model(kind).warm_up(pd.DataFrame())


@pytest.mark.parametrize("kind", ["hst", "iforest"])
def test_shifted_points_score_above_the_threshold(kind, warmup):
    """A point far outside the warm-up distribution must clear the alarm threshold."""
    model = build_model(kind)
    model.warm_up(warmup)

    far = _row(12.0, 90.0)
    assert model.score_one(far) > model.threshold
    assert model.predict_one(far) == 1


def test_threshold_flags_roughly_the_intended_fraction_of_warmup(warmup):
    """The 0.98 quantile should flag about 2% of the data it was set on."""
    model = StaticIsolationForest(threshold_quantile=0.98)
    model.warm_up(warmup)
    flagged = (model.score_many(warmup) > model.threshold).mean()
    assert 0.0 < flagged <= 0.05


def test_static_model_does_not_learn(warmup):
    model = StaticIsolationForest()
    model.warm_up(warmup)

    far = _row(12.0, 90.0)
    before = model.score_one(far)
    for _ in range(500):
        model.learn_one(far)
    assert model.score_one(far) == before
    assert model.is_online is False


def test_online_model_adapts_to_a_repeated_new_regime(warmup):
    """The whole premise of the online model: it stops alarming once a shift persists."""
    model = OnlineHalfSpaceTrees(window_size=100)
    model.warm_up(warmup)
    assert model.is_online is True

    far = _row(8.0, 60.0)
    before = model.score_one(far)

    rng = np.random.default_rng(11)
    for _ in range(1500):
        model.learn_one(_row(float(rng.normal(8.0, 1.0)), float(rng.normal(60.0, 5.0))))

    assert model.score_one(far) < before


def test_warm_up_shorter_than_the_hst_window_is_rejected():
    """HST scores 0 until its first window fills; a threshold from that is meaningless."""
    frame = pd.DataFrame({"a": np.random.default_rng(0).normal(size=50)})
    model = OnlineHalfSpaceTrees(window_size=250)
    with pytest.raises(ValueError, match="no usable scores"):
        model.warm_up(frame)


def test_build_model_rejects_unknown_kinds():
    with pytest.raises(ValueError, match="unknown model kind"):
        build_model("autoencoder")


def _step_stream(n: int = 1200, at: int = 600, seed: int = 5):
    """Stationary N(0,1) that steps to N(6,1) at `at`."""
    rng = np.random.default_rng(seed)
    return [float(rng.normal(0.0, 1.0) if i < at else rng.normal(6.0, 1.0)) for i in range(n)]


@pytest.mark.parametrize("kind", DETECTOR_KINDS)
def test_detectors_react_promptly_to_a_real_shift(kind):
    monitor = build_detector(kind, cooldown=0)
    fired_after = [i for i, v in enumerate(_step_stream()) if monitor.update(v) and i >= 600]

    assert fired_after, f"{kind} missed a 6-sigma shift"
    assert fired_after[0] - 600 < 200, f"{kind} took {fired_after[0] - 600} steps to react"


# ADWIN is conservative to the point of silence on stationary data; KSWIN runs a
# KS test on every window and pays for its faster reactions with false alarms.
# These bounds are set from measured behaviour, not from the papers: over five
# seeds of 5000 stationary steps, ADWIN fired 0 times every time and KSWIN fired
# 4-8 times. They are here to catch a regression, not to flatter either detector.
#
# adwin_var is the chattiest on *white noise*: the sampling fluctuation of a
# rolling variance is itself a signal ADWIN reacts to (15-22 over five seeds).
# That is the honest cost of watching spread, and it looks worse here than in
# use — on the smoother reference statistic the experiment actually monitors, its
# variance is stable except during drift, so on SKAB it is far quieter than the
# mean detector (see the sweep in PROGRESS.md). This budget is a regression guard
# on the worst case, not a claim about its operating behaviour.
# adwin_meanvar is the OR of the mean and variance branches, so on white noise it
# inherits the variance branch's chattiness (the mean branch is silent here).
STATIONARY_FALSE_ALARM_BUDGET = {
    "adwin": 0, "kswin": 15, "adwin_var": 25, "adwin_meanvar": 25,
}


@pytest.mark.parametrize("kind", DETECTOR_KINDS)
def test_stationary_false_alarms_stay_within_the_measured_budget(kind):
    rng = np.random.default_rng(9)
    monitor = build_detector(kind, cooldown=0)
    for _ in range(5000):
        monitor.update(float(rng.normal(0.0, 1.0)))
    assert monitor.n_detections <= STATIONARY_FALSE_ALARM_BUDGET[kind]


def test_adwin_does_not_re_fire_after_handling_a_shift():
    """ADWIN cuts its window at the change, so one shift yields exactly one alarm."""
    monitor = build_detector("adwin", cooldown=0)
    for value in _step_stream():
        monitor.update(value)
    assert monitor.n_detections == 1


def test_cooldown_suppresses_repeat_detections():
    """KSWIN keeps firing while its reference window turns over; the cooldown absorbs that."""
    stream = _step_stream()

    without = build_detector("kswin", cooldown=0)
    for value in stream:
        without.update(value)

    with_cooldown = build_detector("kswin", cooldown=400)
    for value in stream:
        with_cooldown.update(value)

    assert without.n_detections > 1, "expected KSWIN to fire repeatedly without a cooldown"
    assert with_cooldown.n_detections < without.n_detections
    assert with_cooldown.n_suppressed > 0
    assert np.all(np.diff(with_cooldown.detections) >= 400)


@pytest.mark.parametrize("kind", DETECTOR_KINDS)
def test_reset_clears_the_window_but_keeps_history(kind):
    monitor = build_detector(kind, cooldown=0)
    for value in _step_stream():
        monitor.update(value)

    before = list(monitor.detections)
    assert before, "need at least one detection to make this test meaningful"

    monitor.reset()
    assert monitor.detections == before, "history must survive a reset"
    assert not monitor.detector.drift_detected, "a fresh detector cannot already be firing"


def test_resetting_on_every_detection_still_tracks_a_shift():
    """Reset-on-adapt must not break detection, whatever it does to alarm counts."""
    monitor = build_detector("adwin", cooldown=0)

    fired = []
    for i, value in enumerate(_step_stream()):
        if monitor.update(value):
            fired.append(i)
            # Stand in for an adaptation: the reference moves, so the monitor
            # must forget the window that preceded it.
            monitor.reset()

    assert fired, "resetting must not prevent the shift from being detected"
    assert any(i >= 600 for i in fired)


def test_reset_requires_a_factory():
    monitor = build_detector("adwin")
    monitor._factory = None
    with pytest.raises(RuntimeError, match="cannot reset"):
        monitor.reset()


def test_build_detector_rejects_unknown_kinds():
    with pytest.raises(ValueError, match="unknown detector kind"):
        build_detector("page-hinkley")


# --- the dispersion transform and the adwin_var detector -------------------

def test_rolling_dispersion_reports_windowed_variance():
    from src.detectors import RollingDispersion

    disp = RollingDispersion(window=3)
    assert disp(5.0) == 0.0, "one point has no dispersion yet"
    # After [5, 7]: variance of two points about their mean (6) is 1.0.
    assert disp(7.0) == pytest.approx(1.0)
    # Window is 3; feeding a fourth value drops the first.
    disp(9.0)  # buffer now [5, 7, 9]
    assert disp(11.0) == pytest.approx(np.var([7.0, 9.0, 11.0]))


def test_rolling_dispersion_reset_and_bad_window():
    from src.detectors import RollingDispersion

    disp = RollingDispersion(window=5)
    disp(1.0)
    disp(9.0)
    disp.reset()
    assert disp(3.0) == 0.0, "after reset the window is empty again"

    with pytest.raises(ValueError, match="window must be at least 2"):
        RollingDispersion(window=1)


def test_adwin_var_catches_a_spread_change_that_the_mean_detector_misses():
    """The reason adwin_var exists: a change in spread with the mean held fixed.

    This is the gradual-drift case in miniature — the signal starts flipping
    between two values around a constant mean, so its location never moves but its
    variance jumps. The plain mean detector should stay silent; adwin_var should
    fire.
    """
    rng = np.random.default_rng(3)
    # First half: tight around 0. Second half: same mean, far wider spread.
    stream = [float(rng.normal(0.0, 0.05)) for _ in range(600)]
    stream += [float(rng.normal(0.0, 3.0)) for _ in range(600)]

    mean_det = build_detector("adwin", cooldown=0)
    var_det = build_detector("adwin_var", cooldown=0)
    mean_fired = [i for i, v in enumerate(stream) if mean_det.update(v) and i >= 600]
    var_fired = [i for i, v in enumerate(stream) if var_det.update(v) and i >= 600]

    assert not mean_fired, "a pure spread change should not move the mean detector"
    assert var_fired, "adwin_var should catch the spread change"
    assert var_fired[0] - 600 < 200


def test_adwin_var_attaches_a_transform_and_passes_its_window():
    from src.detectors import RollingDispersion

    monitor = build_detector("adwin_var", window=17)
    assert isinstance(monitor.transform, RollingDispersion)
    assert monitor.transform.window == 17
    # The plain detectors carry no transform.
    assert build_detector("adwin").transform is None


def test_adwin_var_reset_clears_the_transform_window():
    monitor = build_detector("adwin_var", cooldown=0)
    for value in _step_stream():
        monitor.update(value)
    assert monitor.transform is not None and len(monitor.transform._buffer) > 0

    monitor.reset()
    assert len(monitor.transform._buffer) == 0, "reset must clear the transform's window"


def test_adwin_meanvar_catches_both_a_mean_shift_and_a_spread_change():
    """The combined detector is a superset: a step (mean) and a spread change both fire.

    adwin alone misses the spread change; adwin_var alone misses a pure ramp. The
    OR catches each through the relevant branch.
    """
    # Pure spread change, mean held at 0 — the case adwin cannot see.
    rng = np.random.default_rng(3)
    spread = [float(rng.normal(0.0, 0.05)) for _ in range(600)]
    spread += [float(rng.normal(0.0, 3.0)) for _ in range(600)]

    mean_only = build_detector("adwin", cooldown=0)
    combined = build_detector("adwin_meanvar", cooldown=0)
    assert not [i for i, v in enumerate(spread) if mean_only.update(v) and i >= 600], \
        "adwin should be blind to a pure spread change"
    assert [i for i, v in enumerate(spread) if combined.update(v) and i >= 600], \
        "adwin_meanvar must catch the spread change via its variance branch"

    # A step shift — caught through the mean branch.
    combined_step = build_detector("adwin_meanvar", cooldown=0)
    step_fired = [i for i, v in enumerate(_step_stream()) if combined_step.update(v) and i >= 600]
    assert step_fired and step_fired[0] - 600 < 200


def test_adwin_meanvar_carries_no_monitor_transform():
    """The composite applies the variance transform internally, so the monitor has none."""
    from src.detectors import MeanOrVariance

    monitor = build_detector("adwin_meanvar")
    assert monitor.transform is None
    assert isinstance(monitor.detector, MeanOrVariance)
    # A fresh composite is not already firing.
    assert monitor.detector.drift_detected is False
