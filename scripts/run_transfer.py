"""Do the settings chosen on SKAB transfer to a dataset they were not tuned on?

    python scripts/run_transfer.py

Everything in this project was calibrated on SKAB: the alarm threshold quantile,
ADWIN's delta, the Half-Space Trees geometry. That raises an obvious objection —
those numbers may simply be fitted to one recording of one industrial rig, in
which case none of the results mean much beyond it.

NAB is the test. It differs from SKAB in exactly the ways that should break a
transferred calibration:

- **univariate** (one channel) rather than 8 sensors
- roughly **10% anomalous** rather than valve1's 35%
- different domains entirely: server metrics, taxi demand, ambient temperature

The experiment is a straight comparison:

1. **Transferred** — apply the SKAB-chosen settings to NAB with no retuning.
2. **Retuned** — sweep the alarm threshold on NAB and take the best.

The gap between them is the cost of not retuning. A small gap means the
calibration carries; a large one means it was fitted to SKAB.

Every series is also scored against **flag-everything**, which at NAB's ~10%
base rate is F1 0.18 — a far more demanding reference than SKAB's 0.51.

Writes `results/transfer.csv`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluate import anomaly_metrics  # noqa: E402
from src.experiment import (  # noqa: E402
    NAB_SERIES,
    RunConfig,
    default_strategies,
    nab_scenario,
    run_strategy,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The value chosen on SKAB's injected streams for the rare-anomaly operating
# point. This is the number under test.
SKAB_THRESHOLD = 0.98

# What a NAB-native sweep is allowed to pick from.
CANDIDATE_THRESHOLDS = (0.50, 0.70, 0.80, 0.90, 0.95, 0.98)

WARMUP = 1000


def flag_everything_f1(labels: np.ndarray) -> float:
    base = float(np.mean(labels))
    return 2 * base / (1 + base) if base else 0.0


def main() -> int:
    started = time.perf_counter()
    strategies = default_strategies()
    rows: list[dict] = []

    for series_key in NAB_SERIES:
        scenario = nab_scenario(series_key, threshold_quantile=SKAB_THRESHOLD)
        n = len(scenario.frame)
        if n <= WARMUP + 500:
            print(f"skipping {scenario.name}: only {n} rows")
            continue

        warm_contamination = float(scenario.labels.iloc[:WARMUP].mean())
        print(f"\n{scenario.name}: {n} rows, "
              f"{scenario.labels.mean():.1%} anomalous, "
              f"warm-up window {warm_contamination:.1%} anomalous")

        for quantile in CANDIDATE_THRESHOLDS:
            config = RunConfig(warmup_size=WARMUP, threshold_quantile=quantile)
            for strategy in strategies:
                run = run_strategy(scenario.frame, strategy, config)
                start = run.eval_start
                y_true = scenario.labels.to_numpy()[start:]
                metrics = anomaly_metrics(y_true, run.flags[start:])

                rows.append({
                    "series": scenario.name,
                    "strategy": strategy.name,
                    "threshold_quantile": quantile,
                    "is_skab_setting": quantile == SKAB_THRESHOLD,
                    "f1": metrics.f1,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "alarm_rate": float(run.flags[start:].mean()),
                    "flag_everything_f1": flag_everything_f1(y_true),
                    "n_adaptations": run.n_adaptations,
                    "n_steps": len(y_true),
                    "seconds": round(run.seconds, 2),
                })

        at_skab = [r for r in rows if r["series"] == scenario.name and r["is_skab_setting"]]
        best_skab = max(at_skab, key=lambda r: r["f1"])
        all_series = [r for r in rows if r["series"] == scenario.name]
        best_any = max(all_series, key=lambda r: r["f1"])
        print(
            f"  transferred (q=0.98): {best_skab['f1']:.3f} ({best_skab['strategy']})   "
            f"retuned (q={best_any['threshold_quantile']}): {best_any['f1']:.3f} "
            f"({best_any['strategy']})   "
            f"flag-everything: {best_skab['flag_everything_f1']:.3f}"
        )

    table = pd.DataFrame(rows)
    out = PROJECT_ROOT / "results" / "transfer.csv"
    table.to_csv(out, index=False)
    print(f"\nWrote {out.relative_to(PROJECT_ROOT)} ({len(table)} rows)")

    # ---- the headline: cost of not retuning -------------------------------
    print("\n" + "=" * 78)
    print("TRANSFER: best F1 per series, SKAB setting vs retuned on NAB")
    print("=" * 78)

    summary = []
    for series in table.series.unique():
        subset = table[table.series == series]
        transferred = subset[subset.is_skab_setting].f1.max()
        retuned = subset.f1.max()
        best_q = subset.loc[subset.f1.idxmax(), "threshold_quantile"]
        baseline = subset.flag_everything_f1.iloc[0]
        summary.append({
            "series": series,
            "flag_everything": round(baseline, 3),
            "transferred_q098": round(transferred, 3),
            "retuned": round(retuned, 3),
            "best_q": best_q,
            "cost_of_transfer": round(retuned - transferred, 3),
            "transferred_beats_baseline": transferred > baseline,
        })

    summary_frame = pd.DataFrame(summary)
    print(summary_frame.to_string(index=False))

    summary_frame.to_csv(PROJECT_ROOT / "results" / "transfer_summary.csv", index=False)

    print(
        f"\nmean cost of not retuning: "
        f"{summary_frame.cost_of_transfer.mean():.3f} F1"
    )
    print(
        f"series where the transferred setting still beats flag-everything: "
        f"{int(summary_frame.transferred_beats_baseline.sum())}/{len(summary_frame)}"
    )
    print(f"\nDone in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
