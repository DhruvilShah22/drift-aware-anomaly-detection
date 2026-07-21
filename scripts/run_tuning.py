"""A small hyperparameter sweep to pick sensible defaults.

    python scripts/run_tuning.py

Three knobs, swept one dimension at a time rather than as a full grid — the aim
is a defensible default, not an optimum:

- **threshold quantile** — how selective the alarm is. Measured on the labelled
  stream, where F1 is meaningful.
- **ADWIN delta** — how eager the drift detector is. Measured on the injected
  streams, where change points are known exactly, trading detection delay
  against false alarms.
- **HST n_trees** — capacity, against runtime.

Writes `results/tuning.csv`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluate import anomaly_metrics, drift_metrics  # noqa: E402
from src.experiment import (  # noqa: E402
    RunConfig,
    Strategy,
    default_strategies,
    injected_scenarios,
    labelled_scenario,
    run_strategy,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WARMUP = 1000
SWEEP_ROWS = 6000

THRESHOLD_QUANTILES = (0.50, 0.70, 0.85, 0.90, 0.95, 0.98)
ADWIN_DELTAS = (0.0002, 0.002, 0.02, 0.2)
N_TREES = (10, 25, 50)


def sweep_threshold(scenario) -> list[dict]:
    """Threshold selectivity, on the labelled stream where F1 means something."""
    rows = []
    base_rate = float(scenario.labels.mean())
    # Flagging every single point yields this F1. Any strategy that does not beat
    # it is not doing useful work, however respectable its F1 looks in isolation.
    flag_everything_f1 = 2 * base_rate / (1 + base_rate)
    print(f"labelled stream base rate {base_rate:.3f} "
          f"-> flag-everything F1 = {flag_everything_f1:.3f}\n")

    for quantile in THRESHOLD_QUANTILES:
        config = RunConfig(warmup_size=WARMUP, threshold_quantile=quantile)
        for strategy in default_strategies():
            run = run_strategy(scenario.frame, strategy, config)
            y_true = scenario.labels.to_numpy()[run.eval_start:]
            metrics = anomaly_metrics(y_true, run.flags[run.eval_start:])
            rows.append({
                "sweep": "threshold_quantile",
                "value": quantile,
                "strategy": strategy.name,
                "f1": metrics.f1,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "alarm_rate": float(run.flags[run.eval_start:].mean()),
                "beats_flag_everything": metrics.f1 > flag_everything_f1,
                "n_adaptations": run.n_adaptations,
                "seconds": round(run.seconds, 2),
            })
        best = max(rows[-4:], key=lambda r: r["f1"])
        print(f"  q={quantile:.2f}: best {best['strategy']} F1={best['f1']:.3f} "
              f"(alarm rate {best['alarm_rate']:.3f})")
    return rows


def sweep_adwin(scenarios) -> list[dict]:
    """Detector eagerness, on the injected streams where timing is exact."""
    rows = []
    for delta in ADWIN_DELTAS:
        strategy = Strategy(
            "drift-triggered", model_kind="hst", adapt="drift",
            detector_kind="adwin", detector_kwargs={"delta": delta},
        )
        delays, false_alarms, detected, total = [], [], 0, 0
        for scenario in scenarios:
            config = RunConfig(warmup_size=WARMUP)
            run = run_strategy(scenario.frame, strategy, config)
            start = run.eval_start
            metrics = drift_metrics(
                detections=[d - start for d in run.detections],
                change_points=[c - start for c in scenario.change_points if c >= start],
                n_steps=len(scenario.frame) - start,
            )
            if not np.isnan(metrics.mean_delay):
                delays.append(metrics.mean_delay)
            false_alarms.append(metrics.false_alarms_per_1k)
            detected += metrics.n_detected
            total += metrics.n_change_points

        rows.append({
            "sweep": "adwin_delta",
            "value": delta,
            "strategy": "drift-triggered",
            "mean_delay": float(np.mean(delays)) if delays else float("nan"),
            "false_alarms_per_1k": float(np.mean(false_alarms)),
            "detection_rate": detected / total if total else float("nan"),
        })
        row = rows[-1]
        print(f"  delta={delta:<7}: delay {row['mean_delay']:7.1f}, "
              f"false alarms/1k {row['false_alarms_per_1k']:.2f}, "
              f"detected {detected}/{total}")
    return rows


def sweep_trees(scenario) -> list[dict]:
    """Capacity against runtime."""
    rows = []
    strategy = [s for s in default_strategies() if s.name == "drift-triggered"][0]
    for n_trees in N_TREES:
        config = RunConfig(warmup_size=WARMUP, hst_n_trees=n_trees, threshold_quantile=0.70)
        run = run_strategy(scenario.frame, strategy, config)
        y_true = scenario.labels.to_numpy()[run.eval_start:]
        metrics = anomaly_metrics(y_true, run.flags[run.eval_start:])
        rows.append({
            "sweep": "hst_n_trees",
            "value": n_trees,
            "strategy": "drift-triggered",
            "f1": metrics.f1,
            "seconds": round(run.seconds, 2),
        })
        print(f"  n_trees={n_trees:<3}: F1 {metrics.f1:.3f}, {run.seconds:.1f}s")
    return rows


def main() -> int:
    started = time.perf_counter()

    labelled = labelled_scenario()
    injected = injected_scenarios(magnitude=3.0, max_rows=SWEEP_ROWS)

    print("Sweeping alarm threshold quantile (labelled stream)...")
    rows = sweep_threshold(labelled)

    print("\nSweeping ADWIN delta (injected streams)...")
    rows += sweep_adwin(injected)

    print("\nSweeping HST n_trees (labelled stream)...")
    rows += sweep_trees(labelled)

    table = pd.DataFrame(rows)
    out = PROJECT_ROOT / "results" / "tuning.csv"
    table.to_csv(out, index=False)
    print(f"\nWrote {out.relative_to(PROJECT_ROOT)} ({len(table)} rows)")
    print(f"Done in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
