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

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

from river import drift

DETECTOR_KINDS = ("adwin", "kswin")

# Steps to ignore further detections after one fires. Roughly the warm-up
# length, so an adaptation completes before another can be triggered.
DEFAULT_COOLDOWN = 250


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

    def update(self, value: float) -> bool:
        step = self._step
        self._step += 1

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
    if kind == "adwin":
        factory: Callable[[], drift.base.DriftDetector] = partial(drift.ADWIN, **kwargs)
    elif kind == "kswin":
        # KSWIN samples from its reference window, so it needs a seed to be
        # reproducible; ADWIN is deterministic and takes none.
        factory = partial(drift.KSWIN, seed=seed, **kwargs)
    else:
        raise ValueError(
            f"unknown detector kind {kind!r}; expected one of {DETECTOR_KINDS}"
        )

    return DriftMonitor(
        kind=kind, detector=factory(), cooldown=cooldown, _factory=factory
    )
