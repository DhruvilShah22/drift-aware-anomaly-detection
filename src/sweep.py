"""A wider robustness sweep: drift shape x magnitude x detector.

The local comparison in `experiment.py` runs one magnitude with one detector.
This asks the question that actually matters for judging the approach: **how big
does a change have to be before the detector notices it, and what does that cost
in false alarms?** Sweeping four shapes, several magnitudes and both detectors is
tedious on a weak laptop but trivial on free CPU, which is what the Kaggle
notebook is for.

## The magnitude-0.0 control

Every sweep includes `magnitude=0.0` — no drift injected at all. This is not a
formality. SKAB's anomaly-free recording has no annotated faults but is *not*
stationary, and the detector fires on it regardless: on the full stream it
adapts 18 times with nothing injected. Without that control, those 18 firings
would be silently attributed to the injected change and the method would look
far better than it is.

So the number to read is not "detections", it is **detections near the injected
change point, against what the same run produces with no change injected**.
Both are reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data_loader import StreamData
from src.drift_injection import DRIFT_KINDS, inject
from src.evaluate import drift_metrics
from src.experiment import RunConfig, Strategy, run_strategy

# 0.0 is the control. The rest span "barely there" to "unmissable" in units of
# each sensor's own standard deviation.
DEFAULT_MAGNITUDES = (0.0, 1.0, 2.0, 3.0, 5.0)
# adwin_var feeds ADWIN the trailing variance instead of the location signal. It
# is here because the other two are flat against magnitude on gradual drift,
# where a change shows up as rising spread before rising mean; see src/detectors.py.
DEFAULT_DETECTORS = ("adwin", "kswin", "adwin_var")

# A detection counts as attributable to the injected change if it lands within
# this many rows after it. Same horizon the main evaluation uses.
ATTRIBUTION_HORIZON = 750


@dataclass
class SweepConfig:
    magnitudes: tuple[float, ...] = DEFAULT_MAGNITUDES
    detectors: tuple[str, ...] = DEFAULT_DETECTORS
    kinds: tuple[str, ...] = DRIFT_KINDS
    warmup_size: int = 1000
    threshold_quantile: float = 0.98
    max_rows: int | None = None
    seed: int = 42


def sweep(
    base: StreamData,
    config: SweepConfig | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run every (shape, magnitude, detector) combination and tabulate it.

    `base` should be the anomaly-free recording: no annotated faults, so any
    flag raised is a false alarm and change points are known exactly.
    """
    config = config or SweepConfig()

    frame, labels = base.frame, base.labels
    if config.max_rows is not None:
        frame, labels = frame.iloc[: config.max_rows], labels.iloc[: config.max_rows]

    run_config = RunConfig(
        warmup_size=config.warmup_size,
        threshold_quantile=config.threshold_quantile,
        seed=config.seed,
    )

    rows: list[dict] = []
    total = len(config.kinds) * len(config.magnitudes) * len(config.detectors)
    done = 0

    for kind in config.kinds:
        for magnitude in config.magnitudes:
            stream = inject(kind, frame, labels, magnitude=magnitude)

            for detector in config.detectors:
                strategy = Strategy(
                    "drift-triggered",
                    model_kind="hst",
                    adapt="drift",
                    detector_kind=detector,
                )
                run = run_strategy(stream.frame, strategy, run_config)
                start = run.eval_start

                points = [c for c in stream.drift_points if c >= start]
                metrics = drift_metrics(
                    detections=[d - start for d in run.detections],
                    change_points=[c - start for c in points],
                    n_steps=len(stream.frame) - start,
                    horizon=ATTRIBUTION_HORIZON,
                )

                flags = run.flags[start:]
                rows.append({
                    "kind": kind,
                    "magnitude": magnitude,
                    "detector": detector,
                    "n_change_points": metrics.n_change_points,
                    "n_detected": metrics.n_detected,
                    "detection_rate": metrics.detection_rate,
                    "mean_delay": metrics.mean_delay,
                    "median_delay": metrics.median_delay,
                    "n_detections_total": len(run.detections),
                    "false_alarms_per_1k": metrics.false_alarms_per_1k,
                    "alarm_rate": float(flags.mean()),
                    "n_adaptations": run.n_adaptations,
                    "n_steps": len(flags),
                    "seconds": round(run.seconds, 2),
                })

                done += 1
                if verbose:
                    row = rows[-1]
                    delay = (
                        f"{row['mean_delay']:.0f}"
                        if not np.isnan(row["mean_delay"]) else "—"
                    )
                    print(
                        f"  [{done:>3}/{total}] {kind:<12} mag={magnitude:<4} "
                        f"{detector:<5} detected {row['n_detected']}/"
                        f"{row['n_change_points']}, delay {delay:>4}, "
                        f"{row['n_detections_total']:>3} total firings"
                    )

    return pd.DataFrame(rows)


def attribution_table(table: pd.DataFrame) -> pd.DataFrame:
    """Compare each magnitude against the magnitude-0.0 control.

    `excess_firings` is total detections minus what the same shape and detector
    produced with nothing injected. It is the honest measure of what the
    injected change actually caused, given the base recording drifts on its own.
    """
    control = (
        table[table.magnitude == 0.0]
        .set_index(["kind", "detector"])["n_detections_total"]
        .rename("control_firings")
    )
    if control.empty:
        raise ValueError(
            "the sweep has no magnitude=0.0 control, so nothing can be attributed"
        )

    merged = table.join(control, on=["kind", "detector"])
    merged["excess_firings"] = merged["n_detections_total"] - merged["control_firings"]
    return merged


def summarise_sweep(table: pd.DataFrame) -> pd.DataFrame:
    """Detection rate and delay by magnitude and detector, averaged over shapes."""
    real = table[table.magnitude > 0]
    if real.empty:
        raise ValueError("no non-zero magnitudes to summarise")
    return (
        real.groupby(["detector", "magnitude"])
        .agg(
            detection_rate=("detection_rate", "mean"),
            mean_delay=("mean_delay", "mean"),
            false_alarms_per_1k=("false_alarms_per_1k", "mean"),
            alarm_rate=("alarm_rate", "mean"),
        )
        .round(3)
        .reset_index()
    )
