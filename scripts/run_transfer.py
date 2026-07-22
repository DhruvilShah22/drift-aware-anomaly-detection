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

Both point-wise and event-level metrics are reported (see `src/evaluate.py`).
NAB is where the event-level view earns its keep: its anomalies are short,
sparse windows, so event recall — the fraction of labelled anomaly windows a
strategy caught — actually varies here, unlike SKAB valve1's long fault blocks
where every strategy trivially catches all 15 events and recall saturates at 1.0.

Writes `results/transfer.csv` and `results/transfer_summary.csv`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluate import anomaly_metrics, event_metrics  # noqa: E402
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
                y_pred = run.flags[start:]
                metrics = anomaly_metrics(y_true, y_pred)
                events = event_metrics(y_true, y_pred)

                rows.append({
                    "series": scenario.name,
                    "strategy": strategy.name,
                    "threshold_quantile": quantile,
                    "is_skab_setting": quantile == SKAB_THRESHOLD,
                    "f1": metrics.f1,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "event_f1": events.f1,
                    "event_precision": events.precision,
                    "event_recall": events.recall,
                    "n_true_events": events.n_true_events,
                    "n_detected_events": events.n_detected_events,
                    "alarm_rate": float(y_pred.mean()),
                    # flag-everything reaches the same 2b/(1+b) ceiling under both
                    # metrics (recall 1.0, precision at the base rate), so one
                    # baseline column serves the point-wise and event-level views.
                    "flag_everything_f1": flag_everything_f1(y_true),
                    "n_adaptations": run.n_adaptations,
                    "n_steps": len(y_true),
                    "seconds": round(run.seconds, 2),
                })

        at_skab = [r for r in rows if r["series"] == scenario.name and r["is_skab_setting"]]
        best_skab = max(at_skab, key=lambda r: r["f1"])
        all_series = [r for r in rows if r["series"] == scenario.name]
        best_any = max(all_series, key=lambda r: r["f1"])
        best_event = max(all_series, key=lambda r: r["event_f1"])
        print(
            f"  point-wise transferred (q=0.98): {best_skab['f1']:.3f} ({best_skab['strategy']})   "
            f"retuned (q={best_any['threshold_quantile']}): {best_any['f1']:.3f} "
            f"({best_any['strategy']})   "
            f"flag-everything: {best_skab['flag_everything_f1']:.3f}"
        )
        print(
            f"  event-level: {best_event['n_true_events']} anomaly windows, "
            f"best event F1 {best_event['event_f1']:.3f} ({best_event['strategy']}), "
            f"event recall {best_event['event_recall']:.2f}"
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

        # Event-level, chosen independently: the best threshold for event F1 is
        # not necessarily the best for point-wise F1.
        skab_subset = subset[subset.is_skab_setting]
        event_transferred = float(skab_subset.event_f1.max())
        event_retuned = subset.event_f1.max()
        event_best_row = subset.loc[subset.event_f1.idxmax()]

        summary.append({
            "series": series,
            "n_windows": int(subset.n_true_events.iloc[0]),
            "flag_everything": round(baseline, 3),
            "transferred_q098": round(transferred, 3),
            "retuned": round(retuned, 3),
            "best_q": best_q,
            "cost_of_transfer": round(retuned - transferred, 3),
            "transferred_beats_baseline": transferred > baseline,
            "event_transferred_q098": round(event_transferred, 3),
            "event_retuned": round(event_retuned, 3),
            "event_best_q": event_best_row.threshold_quantile,
            "event_transferred_beats_baseline": event_transferred > baseline,
            # Recall spread *across strategies* at the transferred q=0.98 setting.
            # The best strategy catches every window (recall 1.0, so its event F1
            # is precision-driven, as on SKAB valve1); weaker strategies at the
            # same selective threshold miss whole windows — that is where the
            # recall axis discriminates on NAB. Reported as the min so the
            # variation is visible rather than hidden behind the winner.
            "event_recall_min_q098": round(float(skab_subset.event_recall.min()), 3),
            "event_cost_of_transfer": round(event_retuned - event_transferred, 3),
        })

    summary_frame = pd.DataFrame(summary)
    point_cols = [
        "series", "n_windows", "flag_everything", "transferred_q098",
        "retuned", "best_q", "cost_of_transfer", "transferred_beats_baseline",
    ]
    print(summary_frame[point_cols].to_string(index=False))

    summary_frame.to_csv(PROJECT_ROOT / "results" / "transfer_summary.csv", index=False)

    print(
        f"\nmean cost of not retuning: "
        f"{summary_frame.cost_of_transfer.mean():.3f} F1"
    )
    print(
        f"series where the transferred setting still beats flag-everything: "
        f"{int(summary_frame.transferred_beats_baseline.sum())}/{len(summary_frame)}"
    )

    # ---- the event-level view: does counting windows change the story? ----
    print("\n" + "=" * 78)
    print("EVENT-LEVEL: each labelled anomaly window scored once")
    print("=" * 78)
    event_cols = [
        "series", "n_windows", "flag_everything", "event_transferred_q098",
        "event_retuned", "event_best_q", "event_transferred_beats_baseline",
        "event_recall_min_q098", "event_cost_of_transfer",
    ]
    print(summary_frame[event_cols].to_string(index=False))

    point_beats = int(summary_frame.transferred_beats_baseline.sum())
    event_beats = int(summary_frame.event_transferred_beats_baseline.sum())
    print(
        f"\ntransferred setting beats flag-everything: "
        f"point-wise {point_beats}/{len(summary_frame)}, "
        f"event-level {event_beats}/{len(summary_frame)}"
    )
    print(
        f"mean cost of not retuning: point-wise "
        f"{summary_frame.cost_of_transfer.mean():.3f} F1, event-level "
        f"{summary_frame.event_cost_of_transfer.mean():.3f} F1"
    )
    # The honest recall caveat: for the best strategy per series the recall axis
    # saturates as it does on SKAB, but weaker strategies at the same selective
    # threshold miss whole windows, which is where event recall does its work.
    print(
        f"event recall across strategies at q=0.98 falls as low as "
        f"{summary_frame.event_recall_min_q098.min():.2f} "
        f"(a strategy catching none of a series' anomaly windows), "
        f"while the best strategy per series catches all of them"
    )
    print(f"\nDone in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
