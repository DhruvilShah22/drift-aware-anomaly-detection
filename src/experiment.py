"""The three-strategy comparison: replay a stream and score each adaptation policy.

Every strategy sees the same stream, the same warm-up length, and the same
threshold quantile. The *only* thing that differs is when — or whether — the
model is rebuilt:

- **static** — an Isolation Forest fitted once on the warm-up window. Never
  updated at all. This is what most deployed detectors actually are.
- **online-no-reset** — the online model, learning continuously, but never
  explicitly rebuilt. This is a control, not one of the headline three, and it
  matters more than it first appears. Without it, "static" and "periodic" would
  differ in both the model *and* the policy, with no way to tell which caused
  the gap. It also turns out to demonstrate the opposite failure mode from
  "static": it absorbs a shift into its own notion of normal within a few
  hundred rows and goes quiet.
- **periodic** — rebuilt every `retrain_every` steps on the most recent window,
  whether or not anything changed.
- **drift-triggered** — rebuilt only when a drift monitor fires on the model's
  own anomaly score.

Those two controls bracket the problem. A detector that never updates floods the
operator with alarms once normal moves; one that updates constantly stops
reporting anything. The question the comparison asks is whether reacting to
*detected* change lands somewhere better than either.

Rebuilding always means the same operation: `warm_up` on the most recent
`warmup_size` rows, which regenerates the model, its feature limits, and its
threshold together.

## Two stream families, measuring different things

The injected-drift streams are built on SKAB's anomaly-free recording, so they
contain **no true anomalies at all**. That is not a limitation; it is what makes
them useful. Change points are known to the row, so detection timing is exact,
and since nothing is genuinely anomalous, every flag raised is by definition a
false alarm. That measures the "naive detector spams alerts after normal shifts"
failure mode directly.

F1 is not reported on those streams — with no positives it would be zero for
everyone and tell us nothing. Anomaly quality is measured separately, on the
real labelled SKAB recordings.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data_loader import load_anomaly_free, load_skab_stream
from src.detectors import build_detector
from src.drift_injection import DRIFT_KINDS, inject
from src.evaluate import anomaly_metrics, drift_metrics, summarise
from src.models import DEFAULT_THRESHOLD_QUANTILE, build_model


@dataclass(frozen=True)
class Strategy:
    """One adaptation policy."""

    name: str
    model_kind: str
    adapt: str  # "never" | "periodic" | "drift"
    retrain_every: int = 1000
    detector_kind: str = "adwin"
    detector_kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.adapt not in ("never", "periodic", "drift"):
            raise ValueError(f"unknown adaptation policy {self.adapt!r}")


def default_strategies(detector_kind: str = "adwin") -> list[Strategy]:
    return [
        Strategy("static", model_kind="iforest", adapt="never"),
        Strategy("online-no-reset", model_kind="hst", adapt="never"),
        Strategy("periodic", model_kind="hst", adapt="periodic", retrain_every=1000),
        Strategy("drift-triggered", model_kind="hst", adapt="drift", detector_kind=detector_kind),
    ]


@dataclass
class RunConfig:
    """Settings shared by every strategy in a comparison."""

    warmup_size: int = 1000
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE
    hst_n_trees: int = 25
    hst_height: int = 8
    hst_window_size: int = 250
    cooldown: int = 250
    seed: int = 42


@dataclass
class StrategyRun:
    """Everything one strategy did over one stream.

    Arrays are full stream length; positions inside the warm-up window are NaN
    for scores and 0 for flags, since the strategy had not started predicting
    yet. `eval_start` marks where scoring actually begins.
    """

    strategy: str
    scores: np.ndarray
    flags: np.ndarray
    thresholds: np.ndarray
    adaptations: list[int]
    detections: list[int]
    eval_start: int
    seconds: float

    @property
    def n_adaptations(self) -> int:
        return len(self.adaptations)


def run_strategy(
    frame: pd.DataFrame,
    strategy: Strategy,
    config: RunConfig | None = None,
) -> StrategyRun:
    """Replay `frame` through one strategy, one row at a time."""
    config = config or RunConfig()
    n = len(frame)
    warmup = config.warmup_size
    if n <= warmup + 10:
        raise ValueError(
            f"stream of {n} rows is too short for a warm-up of {warmup}"
        )

    model_kwargs: dict = {"threshold_quantile": config.threshold_quantile, "seed": config.seed}
    if strategy.model_kind == "hst":
        model_kwargs.update(
            n_trees=config.hst_n_trees,
            height=config.hst_height,
            window_size=config.hst_window_size,
        )
    model = build_model(strategy.model_kind, **model_kwargs)

    monitor = None
    if strategy.adapt == "drift":
        monitor = build_detector(
            strategy.detector_kind,
            cooldown=config.cooldown,
            seed=config.seed,
            **strategy.detector_kwargs,
        )

    started = time.perf_counter()
    model.warm_up(frame.iloc[:warmup])

    scores = np.full(n, np.nan)
    flags = np.zeros(n, dtype=int)
    thresholds = np.full(n, np.nan)
    adaptations: list[int] = []
    detections: list[int] = []

    # A model that never adapts and never learns produces the same score for a
    # given row no matter when it is asked, so the whole remainder can be scored
    # in one vectorised call instead of a Python loop over 30,000 rows.
    if strategy.adapt == "never" and not model.is_online:
        block = frame.iloc[warmup:]
        scores[warmup:] = model.score_many(block)
        flags[warmup:] = (scores[warmup:] > model.threshold).astype(int)
        thresholds[warmup:] = model.threshold
        return StrategyRun(
            strategy=strategy.name,
            scores=scores,
            flags=flags,
            thresholds=thresholds,
            adaptations=adaptations,
            detections=detections,
            eval_start=warmup,
            seconds=time.perf_counter() - started,
        )

    records = frame.to_dict(orient="records")
    recent: deque[dict] = deque(records[:warmup], maxlen=warmup)

    for i in range(warmup, n):
        x = records[i]

        score = model.score_one(x)
        scores[i] = score
        thresholds[i] = model.threshold
        flags[i] = int(score > model.threshold)

        model.learn_one(x)
        recent.append(x)

        should_adapt = False
        if strategy.adapt == "periodic":
            should_adapt = (i - warmup) > 0 and (i - warmup) % strategy.retrain_every == 0
        elif strategy.adapt == "drift":
            assert monitor is not None
            if monitor.update(score):
                detections.append(i)
                should_adapt = True

        if should_adapt:
            model.warm_up(pd.DataFrame(list(recent), columns=frame.columns))
            adaptations.append(i)

    return StrategyRun(
        strategy=strategy.name,
        scores=scores,
        flags=flags,
        thresholds=thresholds,
        adaptations=adaptations,
        detections=detections,
        eval_start=warmup,
        seconds=time.perf_counter() - started,
    )


# --------------------------------------------------------------------------
# Stream construction
# --------------------------------------------------------------------------


@dataclass
class Scenario:
    """A stream to run the comparison over, plus what can be measured on it."""

    name: str
    frame: pd.DataFrame
    labels: pd.Series
    change_points: list[int]
    has_true_anomalies: bool
    note: str = ""


def injected_scenarios(
    magnitude: float = 3.0,
    max_rows: int | None = None,
    kinds: tuple[str, ...] = DRIFT_KINDS,
) -> list[Scenario]:
    """The four injected-drift streams, built on SKAB's anomaly-free recording.

    No true anomalies here by construction, so every flag is a false alarm and
    change points are known exactly.
    """
    clean = load_anomaly_free()
    frame, labels = clean.frame, clean.labels
    if max_rows is not None:
        frame, labels = frame.iloc[:max_rows], labels.iloc[:max_rows]

    scenarios = []
    for kind in kinds:
        stream = inject(kind, frame, labels, magnitude=magnitude)
        scenarios.append(
            Scenario(
                name=kind,
                frame=stream.frame,
                labels=stream.labels,
                change_points=stream.drift_points,
                has_true_anomalies=False,
                note=f"{magnitude} sd shift injected into the anomaly-free recording",
            )
        )
    return scenarios


# The labelled SKAB recordings are each only ~1000 rows, too short to warm up on
# and still have stream left to measure. Concatenating a group gives a usable
# length. The joins are genuine regime changes, but I do not know their
# magnitude, so this stream is used only for anomaly quality — never for drift
# timing, where ground truth has to be exact.
LABELLED_GROUP = "valve1"


def labelled_scenario(max_files: int | None = None) -> Scenario:
    """A long labelled stream, concatenated from one SKAB fault group."""
    from src.data_loader import list_skab_files

    filenames = list_skab_files(LABELLED_GROUP)
    if max_files is not None:
        filenames = filenames[:max_files]

    frames, label_parts = [], []
    for filename in filenames:
        stream = load_skab_stream(LABELLED_GROUP, filename)
        frames.append(stream.frame)
        label_parts.append(stream.labels)

    frame = pd.concat(frames, ignore_index=True)
    labels = pd.concat(label_parts, ignore_index=True)

    return Scenario(
        name=f"skab-{LABELLED_GROUP}",
        frame=frame,
        labels=labels,
        change_points=[],
        has_true_anomalies=True,
        note=(
            f"{len(filenames)} {LABELLED_GROUP} recordings concatenated; "
            "used for anomaly quality only, not drift timing"
        ),
    )


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


def score_run(scenario: Scenario, run: StrategyRun, horizon: int = 750) -> dict:
    """Turn one strategy's run over one scenario into a summary row."""
    start = run.eval_start
    y_true = scenario.labels.to_numpy()[start:]
    y_pred = run.flags[start:]
    n_steps = len(y_pred)

    anomaly = anomaly_metrics(y_true, y_pred)

    drift = None
    if scenario.change_points:
        # Only change points inside the evaluated region can be detected.
        points = [c for c in scenario.change_points if c >= start]
        drift = drift_metrics(
            detections=[d - start for d in run.detections],
            change_points=[c - start for c in points],
            n_steps=n_steps,
            horizon=horizon,
        )

    row = summarise(
        strategy=run.strategy,
        stream=scenario.name,
        anomaly=anomaly,
        drift=drift,
        n_adaptations=run.n_adaptations,
        alarm_rate=float(np.mean(y_pred)),
        n_steps=n_steps,
        seconds=round(run.seconds, 2),
    )

    # F1 on a stream with no true anomalies is zero for everyone and says
    # nothing; blank it out rather than publish a misleading column.
    if not scenario.has_true_anomalies:
        for key in ("precision", "recall", "f1"):
            row[key] = np.nan
    return row


def compare(
    scenario: Scenario,
    strategies: list[Strategy] | None = None,
    config: RunConfig | None = None,
    horizon: int = 750,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, StrategyRun]]:
    """Run every strategy over one scenario and return the summary plus raw runs."""
    strategies = strategies or default_strategies()
    config = config or RunConfig()

    rows, runs = [], {}
    for strategy in strategies:
        run = run_strategy(scenario.frame, strategy, config)
        runs[strategy.name] = run
        rows.append(score_run(scenario, run, horizon=horizon))
        if verbose:
            row = rows[-1]
            detail = f"alarm rate {row['alarm_rate']:.3f}"
            if scenario.has_true_anomalies:
                detail = f"F1 {row['f1']:.3f}, " + detail
            if "mean_delay" in row and not np.isnan(row["mean_delay"]):
                detail += f", mean delay {row['mean_delay']:.0f}"
            print(
                f"  {strategy.name:<16} {detail}, "
                f"{run.n_adaptations} adaptations, {run.seconds:.1f}s"
            )

    return pd.DataFrame(rows), runs
