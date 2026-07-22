"""Drift detectors: thin wrappers over river's ADWIN and KSWIN.

Both detectors consume one scalar per step and raise a flag when that scalar's
distribution appears to have changed. The wrapper adds the two things the
experiment needs on top of river's API: a record of *when* each detection fired,
and a cooldown.

**What signal to monitor.** The detectors watch the model's own anomaly score,
not a raw sensor channel. That choice is deliberate: a shift in a sensor is only
worth reacting to if it actually degrades the detector, and the anomaly score is
the one number that summarises all eight channels through the model's eyes. It
also means the same wrapper works unchanged whichever sensors a stream contains.

**Why a cooldown.** After a genuine change, the score stays shifted for a while,
so a detector will keep firing every few steps until its reference window has
fully turned over. Without a cooldown the drift-triggered strategy would retrain
in a burst and each retrain would land on partially-stale data. The cooldown
holds off further adaptation until the model has had time to settle.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import numpy as np
from river import drift

DETECTOR_KINDS = ("adwin", "kswin", "adwin_var", "adwin_meanvar")

# Steps to ignore further detections after one fires. Roughly the warm-up
# length, so an adaptation completes before another can be triggered.
DEFAULT_COOLDOWN = 250

# Trailing window the dispersion transform computes its variance over. Fifty is
# short enough to register a burst of outliers within a few dozen steps, long
# enough that a single spike does not swing the variance on its own.
DEFAULT_DISPERSION_WINDOW = 50


class RollingDispersion:
    """Turns a stream of scalars into the variance of its trailing window.

    Feeding this to ADWIN instead of the raw signal makes the detector watch how
    *spread out* the signal is rather than where it sits. That is the difference
    that catches **gradual** drift: early in a gradual transition individual rows
    flip between the old and new regime, so the monitored signal grows spiky —
    its variance climbs — well before its mean has moved enough for a
    location-based detector to react. On the smooth **incremental** ramp, by
    contrast, the variance barely changes and this transform is blind to it, so
    it is a complement to the plain mean detector, not a replacement. See the
    sweep in PROGRESS.md for the measured trade-off.
    """

    def __init__(self, window: int = DEFAULT_DISPERSION_WINDOW) -> None:
        if window < 2:
            raise ValueError("dispersion window must be at least 2")
        self.window = window
        self._buffer: deque[float] = deque(maxlen=window)

    def __call__(self, value: float) -> float:
        self._buffer.append(float(value))
        # Below two points variance is undefined / trivially zero; report 0 so the
        # detector simply sees no dispersion yet rather than a spurious jump.
        if len(self._buffer) < 2:
            return 0.0
        return float(np.var(self._buffer))

    def reset(self) -> None:
        self._buffer.clear()


class MeanOrVariance:
    """Two ADWINs in parallel; fires when *either* one does.

    One ADWIN watches the raw signal (a location change — the smooth incremental
    ramp), the other watches its trailing variance (a spread change — the
    interleaved gradual case). `adwin` alone is flat on gradual, `adwin_var` alone
    is blind to incremental; ORing them catches every drift shape in one monitor.

    The trade is not hidden: the OR inherits the *union* of the two detectors'
    false alarms, and the mean branch is the chatty one on this data because the
    base recording drifts on its own. So this is the sensitive option — reach for
    it when missing a drift is worse than an extra rebuild, not when quiet is the
    priority. Measured cost is in PROGRESS.md.

    It presents river's DriftDetector surface — `update(value)` then read
    `drift_detected` — so `DriftMonitor` drives it exactly like a plain detector,
    with no transform of its own (this class applies the variance transform to its
    own branch internally).
    """

    def __init__(self, window: int = DEFAULT_DISPERSION_WINDOW, **adwin_kwargs) -> None:
        self._mean = drift.ADWIN(**adwin_kwargs)
        self._var = drift.ADWIN(**adwin_kwargs)
        self._dispersion = RollingDispersion(window=window)
        self._drift = False

    def update(self, value: float) -> "MeanOrVariance":
        self._mean.update(value)
        self._var.update(self._dispersion(value))
        self._drift = bool(self._mean.drift_detected or self._var.drift_detected)
        return self

    @property
    def drift_detected(self) -> bool:
        return self._drift


@dataclass
class DriftMonitor:
    """One drift detector plus a firing history and a cooldown.

    `update` is called once per stream step with the value being monitored and
    returns True only on steps where an adaptation should actually happen —
    detections suppressed by the cooldown return False but are still recorded in
    `suppressed_at`, so the tuning phase can see how chatty a setting is.
    """

    kind: str
    detector: drift.base.DriftDetector
    cooldown: int = DEFAULT_COOLDOWN
    detections: list[int] = field(default_factory=list)
    suppressed_at: list[int] = field(default_factory=list)
    # Optional stateful preprocessor applied to each value before the detector
    # sees it — e.g. RollingDispersion, which makes ADWIN watch variance instead
    # of location. None means the raw value is passed through unchanged.
    transform: RollingDispersion | None = None
    _factory: Callable[[], drift.base.DriftDetector] | None = None
    _step: int = 0
    _last_fired: int | None = None

    def reset(self) -> None:
        """Discard the detector's accumulated window and start a fresh one.

        Called whenever the *meaning* of the monitored signal changes — in this
        project, when the reference window the signal is standardised against is
        refit after an adaptation. Otherwise the detector compares
        post-adaptation values against a pre-adaptation window and can read the
        refit itself as a change.

        Honest note on its impact: I added this expecting it to cut down a
        suspected feedback loop, and on the SKAB streams it changed the results
        by nothing at all — adaptation counts were identical with and without
        it. The repeated adaptations turned out to have a different cause (real
        non-stationarity in the base recording; see PROGRESS.md). It is kept
        because comparing against a stale reference is wrong regardless of
        whether it happens to bite on this particular dataset, but it is not
        load-bearing for any reported number.

        Step counter, cooldown state, and detection history are all preserved;
        only the statistical window is cleared.
        """
        if self._factory is None:
            raise RuntimeError("this monitor was built without a factory and cannot reset")
        self.detector = self._factory()
        # The transform carries its own trailing window; it too is compared
        # against a reference that has just moved, so clear it alongside.
        if self.transform is not None:
            self.transform.reset()

    def update(self, value: float) -> bool:
        step = self._step
        self._step += 1

        if self.transform is not None:
            value = self.transform(value)
        self.detector.update(value)
        if not self.detector.drift_detected:
            return False

        if self._last_fired is not None and step - self._last_fired < self.cooldown:
            self.suppressed_at.append(step)
            return False

        self._last_fired = step
        self.detections.append(step)
        return True

    @property
    def n_detections(self) -> int:
        return len(self.detections)

    @property
    def n_suppressed(self) -> int:
        return len(self.suppressed_at)


def build_detector(
    kind: str,
    cooldown: int = DEFAULT_COOLDOWN,
    seed: int = 42,
    **kwargs,
) -> DriftMonitor:
    """Construct a drift monitor by name.

    Extra keyword arguments go straight to the underlying river detector, so a
    sweep can pass `delta=...` for ADWIN or `alpha=...` for KSWIN.
    """
    transform: RollingDispersion | None = None
    if kind == "adwin":
        factory: Callable[[], drift.base.DriftDetector] = partial(drift.ADWIN, **kwargs)
    elif kind == "adwin_var":
        # Same ADWIN, but fed the trailing variance of the signal rather than the
        # signal itself, so it reacts to a change in spread. `window` is peeled off
        # for the transform; anything else goes to ADWIN.
        window = kwargs.pop("window", DEFAULT_DISPERSION_WINDOW)
        transform = RollingDispersion(window=window)
        factory = partial(drift.ADWIN, **kwargs)
    elif kind == "adwin_meanvar":
        # A mean ADWIN OR a variance ADWIN — catches every shape. The composite
        # applies the variance transform to its own branch, so no monitor-level
        # transform is attached. `window` sizes that branch; the rest go to both
        # ADWINs.
        window = kwargs.pop("window", DEFAULT_DISPERSION_WINDOW)
        factory = partial(MeanOrVariance, window=window, **kwargs)
    elif kind == "kswin":
        # KSWIN samples from its reference window, so it needs a seed to be
        # reproducible; ADWIN is deterministic and takes none.
        factory = partial(drift.KSWIN, seed=seed, **kwargs)
    else:
        raise ValueError(
            f"unknown detector kind {kind!r}; expected one of {DETECTOR_KINDS}"
        )

    return DriftMonitor(
        kind=kind,
        detector=factory(),
        cooldown=cooldown,
        transform=transform,
        _factory=factory,
    )
