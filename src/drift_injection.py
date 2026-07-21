"""Injecting concept drift into a clean stream at known locations.

The labelled SKAB recordings tell us where the *anomalies* are, but not where
the underlying process changed. To measure how quickly a drift detector reacts,
I need streams where the change points are known exactly. So I take the
anomaly-free SKAB recording, which contains no annotated faults, and shift its
distribution myself at locations I choose.

Four shapes are implemented, covering the standard taxonomy:

- **sudden** — the distribution steps to a new level at one instant.
- **incremental** — it ramps from old to new over a window.
- **gradual** — old and new regimes interleave, the new one winning out over time.
- **recurring** — the shifted regime comes and goes in alternating blocks.

Every injector returns the modified frame together with the exact indices where
drift begins and a per-row mask of which regime each row belongs to. Shifts are
expressed in units of each column's own standard deviation, so one magnitude
setting means the same thing across sensors on wildly different scales.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DRIFT_KINDS = ("sudden", "incremental", "gradual", "recurring")


@dataclass(frozen=True)
class DriftStream:
    """A stream with drift injected at known points.

    `drift_points` are the row positions where a change *begins* — what a drift
    detector should ideally flag. `drift_mask` is per-row and says whether that
    row was actually shifted, which is what the false-alarm accounting uses.
    """

    name: str
    kind: str
    frame: pd.DataFrame
    labels: pd.Series
    drift_points: list[int]
    drift_mask: np.ndarray
    affected_columns: list[str] = field(default_factory=list)
    magnitude: float = 0.0

    @property
    def n_rows(self) -> int:
        return len(self.frame)

    def describe(self) -> str:
        return (
            f"{self.name} [{self.kind}]: {self.n_rows} rows, "
            f"drift points at {self.drift_points}, "
            f"{int(self.drift_mask.sum())} shifted rows "
            f"({self.drift_mask.mean():.1%}), "
            f"magnitude {self.magnitude} sd on {len(self.affected_columns)} columns"
        )


def _validate(frame: pd.DataFrame, columns: list[str] | None) -> list[str]:
    if frame.empty:
        raise ValueError("cannot inject drift into an empty frame")
    if columns is None:
        return list(frame.columns)
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"columns not in frame: {missing}")
    return list(columns)


def _column_shifts(frame: pd.DataFrame, columns: list[str], magnitude: float) -> pd.Series:
    """Per-column offset of `magnitude` standard deviations.

    A column that never varies gets no shift; scaling zero variance by anything
    still leaves nothing to detect, and silently shifting it by an absolute
    amount would make the magnitude parameter mean different things per column.
    """
    sd = frame[columns].std()
    return (sd.fillna(0.0) * magnitude).replace([np.inf, -np.inf], 0.0)


def _apply_shift(
    frame: pd.DataFrame,
    columns: list[str],
    shifts: pd.Series,
    weights: np.ndarray,
) -> pd.DataFrame:
    """Add `shifts * weights[row]` to `columns`, leaving other columns untouched."""
    out = frame.copy()
    for column in columns:
        out[column] = out[column].to_numpy() + shifts[column] * weights
    return out


def inject_sudden(
    frame: pd.DataFrame,
    labels: pd.Series,
    at: float = 0.5,
    magnitude: float = 3.0,
    columns: list[str] | None = None,
    name: str = "sudden",
) -> DriftStream:
    """Step the distribution to a new level at a single point and leave it there.

    `at` is a fraction of the stream length. This is the easiest shape to detect
    and the clearest to look at, which makes it the natural demo default.
    """
    columns = _validate(frame, columns)
    n = len(frame)
    point = int(round(at * n))
    if not 0 < point < n:
        raise ValueError(f"drift point {point} falls outside the stream (n={n})")

    weights = np.zeros(n)
    weights[point:] = 1.0

    shifted = _apply_shift(frame, columns, _column_shifts(frame, columns, magnitude), weights)
    return DriftStream(
        name=name,
        kind="sudden",
        frame=shifted,
        labels=labels.copy(),
        drift_points=[point],
        drift_mask=weights > 0,
        affected_columns=columns,
        magnitude=magnitude,
    )


def inject_incremental(
    frame: pd.DataFrame,
    labels: pd.Series,
    start: float = 0.4,
    end: float = 0.7,
    magnitude: float = 3.0,
    columns: list[str] | None = None,
    name: str = "incremental",
) -> DriftStream:
    """Ramp linearly from the old level to the new one over a window.

    The drift point reported is where the ramp starts, since that is the first
    moment any change is present. Detectors typically fire somewhere inside the
    ramp rather than at its start, which is exactly the timing effect worth
    measuring.
    """
    columns = _validate(frame, columns)
    n = len(frame)
    i0, i1 = int(round(start * n)), int(round(end * n))
    if not 0 < i0 < i1 <= n:
        raise ValueError(f"invalid ramp window ({i0}, {i1}) for n={n}")

    weights = np.zeros(n)
    weights[i0:i1] = np.linspace(0.0, 1.0, i1 - i0, endpoint=False)
    weights[i1:] = 1.0

    shifted = _apply_shift(frame, columns, _column_shifts(frame, columns, magnitude), weights)
    return DriftStream(
        name=name,
        kind="incremental",
        frame=shifted,
        labels=labels.copy(),
        drift_points=[i0],
        drift_mask=weights > 0,
        affected_columns=columns,
        magnitude=magnitude,
    )


def inject_gradual(
    frame: pd.DataFrame,
    labels: pd.Series,
    start: float = 0.4,
    end: float = 0.7,
    magnitude: float = 3.0,
    columns: list[str] | None = None,
    seed: int = 0,
    name: str = "gradual",
) -> DriftStream:
    """Interleave the old and new regimes, with the new one taking over.

    Unlike the incremental ramp, individual rows are fully in one regime or the
    other; what changes over the window is the probability of drawing the new
    one. This is the shape that trips up detectors relying on a smooth mean
    shift, because early on the new regime looks like sporadic outliers.
    """
    columns = _validate(frame, columns)
    n = len(frame)
    i0, i1 = int(round(start * n)), int(round(end * n))
    if not 0 < i0 < i1 <= n:
        raise ValueError(f"invalid transition window ({i0}, {i1}) for n={n}")

    rng = np.random.default_rng(seed)
    probability = np.zeros(n)
    probability[i0:i1] = np.linspace(0.0, 1.0, i1 - i0, endpoint=False)
    probability[i1:] = 1.0
    weights = (rng.random(n) < probability).astype(float)

    shifted = _apply_shift(frame, columns, _column_shifts(frame, columns, magnitude), weights)
    return DriftStream(
        name=name,
        kind="gradual",
        frame=shifted,
        labels=labels.copy(),
        drift_points=[i0],
        drift_mask=weights > 0,
        affected_columns=columns,
        magnitude=magnitude,
    )


def inject_recurring(
    frame: pd.DataFrame,
    labels: pd.Series,
    period: float = 0.25,
    magnitude: float = 3.0,
    columns: list[str] | None = None,
    name: str = "recurring",
) -> DriftStream:
    """Alternate between the base regime and a shifted one in equal blocks.

    Every boundary is a drift point, in both directions: a detector should react
    when the shifted regime arrives *and* when it departs. This is the shape that
    punishes blind periodic retraining, because a detector that retrains on the
    shifted regime has to unlearn it a block later.
    """
    columns = _validate(frame, columns)
    n = len(frame)
    block = int(round(period * n))
    if block < 1 or block >= n:
        raise ValueError(f"period {period} gives block size {block} for n={n}")

    # Truncate to a whole number of blocks. A trailing partial block would
    # otherwise contribute a boundary a handful of rows from the end, which no
    # detector could fairly be scored on.
    n = (n // block) * block
    frame = frame.iloc[:n]
    labels = labels.iloc[:n]

    index = np.arange(n)
    weights = ((index // block) % 2 == 1).astype(float)
    boundaries = [int(b) for b in range(block, n, block)]

    shifted = _apply_shift(frame, columns, _column_shifts(frame, columns, magnitude), weights)
    return DriftStream(
        name=name,
        kind="recurring",
        frame=shifted,
        labels=labels.copy(),
        drift_points=boundaries,
        drift_mask=weights > 0,
        affected_columns=columns,
        magnitude=magnitude,
    )


INJECTORS = {
    "sudden": inject_sudden,
    "incremental": inject_incremental,
    "gradual": inject_gradual,
    "recurring": inject_recurring,
}


def inject(kind: str, frame: pd.DataFrame, labels: pd.Series, **kwargs) -> DriftStream:
    """Dispatch to one injector by name."""
    if kind not in INJECTORS:
        raise ValueError(f"unknown drift kind {kind!r}; expected one of {DRIFT_KINDS}")
    return INJECTORS[kind](frame, labels, **kwargs)
