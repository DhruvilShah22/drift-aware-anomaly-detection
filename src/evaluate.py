"""Metrics: anomaly detection quality, drift-detection timing, and false alarms.

Three things get measured, because a strategy can look good on one and be
useless in practice on another:

1. **Anomaly quality** — precision, recall, F1 over the flags a strategy raised,
   both overall and as a rolling series so the demo can show quality changing
   as the stream progresses. Reported two ways: point-wise (every row scored
   independently) and event-level (each contiguous fault block counts once).
   The event-level view exists because point-wise F1 is nearly useless on the
   labelled SKAB streams — see `EventMetrics`.
2. **Drift timing** — how many steps after each true change point the detector
   actually fired, and how many change points it missed entirely. A detector
   that eventually notices every change but takes two thousand steps to do it is
   not much use.
3. **False alarms** — detections that cannot be attributed to any real change,
   reported as a rate per 1000 steps so streams of different lengths compare.

The drift matching rule is deliberately strict and stated here so the numbers
can be read without guessing: each true change point is matched to the *first*
detection at or after it, within `horizon` steps, and before the next change
point. Every detection that fails to match something is a false alarm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# A detection more than this many steps after a change point is not credited to
# it. Chosen to be a few HST windows: long enough to be fair to a slow detector,
# short enough that a "detection" still means something operationally.
DEFAULT_HORIZON = 750


@dataclass(frozen=True)
class AnomalyMetrics:
    """Point-wise anomaly detection quality."""

    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    n_flagged: int
    n_actual: int

    def as_dict(self) -> dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "n_flagged": self.n_flagged,
            "n_actual": self.n_actual,
        }


@dataclass(frozen=True)
class EventMetrics:
    """Event-level anomaly quality: each contiguous fault block counts once.

    Point-wise F1 is dominated by block length and base rate on the labelled SKAB
    streams. Catching a 500-row fault with a handful of well-placed flags scores
    near-zero point-wise *recall*, so a genuinely useful sparse detector looks the
    same as a silent one, while flag-everything sits at the top purely on volume.
    Every strategy ends up clustered near the trivial baseline and F1 cannot tell
    them apart. This metric scores two things that pull apart instead:

    - **recall** is the fraction of distinct true fault *events* that received at
      least one flag. A block caught anywhere counts once, in full; its length no
      longer sets the reward. This is what frees a sparse detector.
    - **precision** stays point-wise: the fraction of flagged points that land
      inside a true fault (front-extended by `tolerance`). This is the volume
      penalty — flagging everything drives precision down to the base rate.

    The asymmetry is deliberate. Recall asks "was each fault noticed at all",
    which is an event question; precision asks "how much of the alarm budget was
    spent on real faults", which is a volume question. `f1` is their harmonic
    mean. Note this does *not* lower flag-everything's score — with recall 1.0 and
    precision at the base rate it still reaches 2b/(1+b), the same ceiling as
    point-wise F1. The gain is that a good detector is no longer pinned there too.
    """

    precision: float
    recall: float
    f1: float
    n_true_events: int
    n_detected_events: int
    n_flagged: int
    n_flagged_hits: int

    def as_dict(self) -> dict[str, float]:
        return {
            "event_precision": self.precision,
            "event_recall": self.recall,
            "event_f1": self.f1,
            "n_true_events": self.n_true_events,
            "n_detected_events": self.n_detected_events,
        }


@dataclass(frozen=True)
class DriftMetrics:
    """Drift-detection timing and false-alarm behaviour."""

    n_change_points: int
    n_detected: int
    n_missed: int
    n_false_alarms: int
    false_alarms_per_1k: float
    mean_delay: float
    median_delay: float
    delays: tuple[int, ...]

    @property
    def detection_rate(self) -> float:
        if self.n_change_points == 0:
            return float("nan")
        return self.n_detected / self.n_change_points

    def as_dict(self) -> dict[str, float]:
        return {
            "n_change_points": self.n_change_points,
            "n_detected": self.n_detected,
            "n_missed": self.n_missed,
            "detection_rate": self.detection_rate,
            "n_false_alarms": self.n_false_alarms,
            "false_alarms_per_1k": self.false_alarms_per_1k,
            "mean_delay": self.mean_delay,
            "median_delay": self.median_delay,
        }


def anomaly_metrics(y_true, y_pred) -> AnomalyMetrics:
    """Precision, recall and F1 for binary anomaly flags.

    Written out rather than delegated to sklearn so the degenerate cases are
    explicit: with nothing flagged precision is 0, and with nothing to find
    recall is 0. Both are treated as 0 rather than as undefined, so a strategy
    that stays silent scores badly instead of being excluded from a comparison.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return AnomalyMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        n_flagged=int(y_pred.sum()),
        n_actual=int(y_true.sum()),
    )


def rolling_f1(y_true, y_pred, window: int = 200) -> np.ndarray:
    """F1 computed over a trailing window at every step.

    This is the prequential view: quality as the stream is consumed, which is
    what makes a static model's decay visible. Positions before a full window
    has accumulated are scored on what is available so far.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if window < 1:
        raise ValueError("window must be at least 1")

    n = len(y_true)
    out = np.zeros(n)
    for i in range(n):
        lo = max(0, i - window + 1)
        out[i] = anomaly_metrics(y_true[lo : i + 1], y_pred[lo : i + 1]).f1
    return out


def to_events(labels) -> list[tuple[int, int]]:
    """Maximal contiguous runs of 1 as half-open ``[start, end)`` ranges.

    A ground-truth fault that spans rows 100..149 inclusive is one event
    ``(100, 150)``, not fifty. Everything event-level is built on this.
    """
    a = np.asarray(labels).astype(int)
    events: list[tuple[int, int]] = []
    i, n = 0, len(a)
    while i < n:
        if a[i] == 1:
            j = i
            while j < n and a[j] == 1:
                j += 1
            events.append((i, j))
            i = j
        else:
            i += 1
    return events


def event_metrics(y_true, y_pred, tolerance: int = 0) -> EventMetrics:
    """Event-level recall paired with point-level precision. See EventMetrics.

    ``tolerance`` front-extends each true event, so a flag up to that many steps
    *before* a fault's onset still counts as catching it — an early warning is a
    hit, not a false alarm. It defaults to 0 (strict: the flag must land in the
    labelled block itself).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    events = to_events(y_true)
    n_true_events = len(events)

    # The region a flag can score in: each true event, front-extended by
    # `tolerance`. Precision and recall both judge hits against this one region,
    # so an early-warning flag is credited consistently on both sides.
    tp_region = np.zeros(len(y_true), dtype=bool)
    for start, end in events:
        tp_region[max(0, start - tolerance) : end] = True

    flagged = y_pred == 1
    n_flagged = int(flagged.sum())
    n_flagged_hits = int(np.sum(flagged & tp_region))
    precision = n_flagged_hits / n_flagged if n_flagged else 0.0

    detected = sum(
        1 for start, end in events if flagged[max(0, start - tolerance) : end].any()
    )
    recall = detected / n_true_events if n_true_events else 0.0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return EventMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        n_true_events=n_true_events,
        n_detected_events=detected,
        n_flagged=n_flagged,
        n_flagged_hits=n_flagged_hits,
    )


def drift_metrics(
    detections,
    change_points,
    n_steps: int,
    horizon: int = DEFAULT_HORIZON,
) -> DriftMetrics:
    """Match detections to true change points and score the timing.

    Each change point takes the first unused detection at or after it, within
    `horizon` steps and before the next change point. Anything left over is a
    false alarm. Detections before the first change point are false alarms by
    construction, since there was nothing yet to detect.
    """
    detections = sorted(int(d) for d in detections)
    change_points = sorted(int(c) for c in change_points)
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")

    matched: set[int] = set()
    delays: list[int] = []

    for i, point in enumerate(change_points):
        # A detection may not be credited to a change point once the next one
        # has already happened; by then it is ambiguous which caused it.
        next_point = change_points[i + 1] if i + 1 < len(change_points) else n_steps
        limit = min(point + horizon, next_point)

        for detection in detections:
            if detection in matched:
                continue
            if point <= detection < limit:
                matched.add(detection)
                delays.append(detection - point)
                break

    n_false = len(detections) - len(matched)
    return DriftMetrics(
        n_change_points=len(change_points),
        n_detected=len(delays),
        n_missed=len(change_points) - len(delays),
        n_false_alarms=n_false,
        false_alarms_per_1k=1000.0 * n_false / n_steps,
        mean_delay=float(np.mean(delays)) if delays else float("nan"),
        median_delay=float(np.median(delays)) if delays else float("nan"),
        delays=tuple(delays),
    )


def summarise(
    strategy: str,
    stream: str,
    anomaly: AnomalyMetrics,
    drift: DriftMetrics | None = None,
    **extra,
) -> dict:
    """Flatten one strategy's results into a row for the summary table."""
    row: dict = {"stream": stream, "strategy": strategy}
    row.update(anomaly.as_dict())
    if drift is not None:
        row.update(drift.as_dict())
    row.update(extra)
    return row


def results_table(rows: list[dict]) -> pd.DataFrame:
    """Assemble result rows into the table that gets written to results/metrics.csv."""
    if not rows:
        raise ValueError("no result rows to tabulate")
    return pd.DataFrame(rows)
