"""The full comparison run: every strategy over every stream, with figures.

    python scripts/run_experiment.py            # full run
    python scripts/run_experiment.py --quick    # smaller, for a weak machine

Writes `results/metrics.csv` and the figures under `results/figures/`. Runs are
cached: the raw per-step output of each strategy is saved to `data/cache/`, so
regenerating figures after a plotting tweak does not re-run the models. Pass
`--force` to recompute anyway.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluate import rolling_f1  # noqa: E402
from src.experiment import (  # noqa: E402
    RunConfig,
    Scenario,
    compare,
    default_strategies,
    injected_scenarios,
    labelled_scenario,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# A consistent colour per strategy across every figure.
COLOURS = {
    "static": "#c1440e",
    "online-no-reset": "#8a8a8a",
    "periodic": "#1f6f8b",
    "drift-triggered": "#2a7f4f",
}


def build_scenarios(quick: bool) -> list[Scenario]:
    max_rows = 4000 if quick else None
    scenarios = injected_scenarios(magnitude=3.0, max_rows=max_rows)
    scenarios.append(labelled_scenario(max_files=4 if quick else None))
    return scenarios


def run_all(quick: bool, force: bool) -> tuple[pd.DataFrame, dict]:
    config = RunConfig(warmup_size=500 if quick else 1000)
    cache_path = CACHE_DIR / f"runs_{'quick' if quick else 'full'}.pkl"

    if cache_path.exists() and not force:
        print(f"Loading cached runs from {cache_path.name} (pass --force to recompute)\n")
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        return payload["table"], payload["runs"]

    tables, all_runs = [], {}
    for scenario in build_scenarios(quick):
        print(f"{scenario.name}: {len(scenario.frame)} rows — {scenario.note}")
        table, runs = compare(scenario, default_strategies(), config)
        tables.append(table)
        all_runs[scenario.name] = (scenario, runs)
        print()

    table = pd.concat(tables, ignore_index=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump({"table": table, "runs": all_runs}, handle)
    return table, all_runs


def figure_stream_walkthrough(scenario: Scenario, runs: dict, path: Path) -> None:
    """The signal, the injected change points, and what each strategy flagged."""
    strategies = ["static", "online-no-reset", "periodic", "drift-triggered"]
    fig, axes = plt.subplots(
        len(strategies) + 1, 1, figsize=(11, 9), sharex=True,
        gridspec_kw={"height_ratios": [1.4] + [1] * len(strategies)},
    )

    signal_column = scenario.frame.columns[0]
    signal = scenario.frame[signal_column].to_numpy()
    x = np.arange(len(signal))

    axes[0].plot(x, signal, lw=0.6, color="#333333")
    axes[0].set_ylabel(signal_column, fontsize=8)
    axes[0].set_title(
        f"{scenario.name} — {scenario.note}", fontsize=10, loc="left"
    )

    for ax in axes:
        for point in scenario.change_points:
            ax.axvline(point, color="#d62728", ls="--", lw=1.0, alpha=0.7)

    for ax, name in zip(axes[1:], strategies):
        run = runs[name]
        flagged = np.where(run.flags == 1)[0]
        ax.plot(x, signal, lw=0.4, color="#cccccc")
        ax.scatter(
            flagged, signal[flagged], s=3, color=COLOURS[name], label=f"flagged ({len(flagged)})"
        )
        for adaptation in run.adaptations:
            ax.axvline(adaptation, color=COLOURS[name], lw=0.9, alpha=0.55)
        ax.set_ylabel(name, fontsize=8)
        ax.legend(loc="upper left", fontsize=7, frameon=False)

    axes[-1].set_xlabel("stream position")
    fig.suptitle(
        "Dashed red: injected change point.  Solid: model rebuild.",
        fontsize=8, y=0.995,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def figure_alarm_rates(table: pd.DataFrame, path: Path) -> None:
    """Alarm rate per strategy on the injected streams, where every flag is false."""
    injected = table[table.stream.isin(["sudden", "incremental", "gradual", "recurring"])]
    pivot = injected.pivot(index="stream", columns="strategy", values="alarm_rate")
    order = [c for c in COLOURS if c in pivot.columns]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    pivot[order].plot.bar(ax=ax, color=[COLOURS[c] for c in order], width=0.78)
    ax.set_ylabel("fraction of points flagged")
    ax.set_xlabel("")
    ax.set_title(
        "False-alarm rate on injected-drift streams\n"
        "(these streams contain no true anomalies, so every flag is a false alarm)",
        fontsize=10, loc="left",
    )
    ax.tick_params(axis="x", rotation=0)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def figure_rolling_f1(scenario: Scenario, runs: dict, path: Path, window: int = 300) -> None:
    """Anomaly F1 over a trailing window, on the labelled stream."""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    start = next(iter(runs.values())).eval_start
    y_true = scenario.labels.to_numpy()[start:]

    for name, run in runs.items():
        series = rolling_f1(y_true, run.flags[start:], window=window)
        ax.plot(
            np.arange(start, start + len(series)), series,
            lw=1.1, color=COLOURS.get(name), label=name,
        )

    ax.set_xlabel("stream position")
    ax.set_ylabel(f"F1 over trailing {window} points")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(
        f"Anomaly detection quality over time — {scenario.name}", fontsize=10, loc="left"
    )
    ax.legend(fontsize=8, frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def figure_event_vs_point_f1(table: pd.DataFrame, path: Path) -> None:
    """Point-wise vs event-level F1 on the labelled stream, against the baseline.

    The whole argument for event-level scoring in one picture: point-wise F1
    clusters every strategy around the flag-everything line, while event-level F1
    lifts the strategies that catch every fault with fewer flags clear of it.
    """
    labelled = table[table.stream.str.startswith("skab-")]
    if labelled.empty or "event_f1" not in labelled.columns:
        print("  (no event-level rows to plot)")
        return

    order = [s for s in COLOURS if s in set(labelled.strategy)]
    rows = labelled.set_index("strategy").loc[order]
    baseline = float(rows["flag_everything_f1"].iloc[0])

    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(x - 0.2, rows["f1"], width=0.38, color="#b0b0b0", label="point-wise F1")
    ax.bar(x + 0.2, rows["event_f1"], width=0.38,
           color=[COLOURS[s] for s in order], label="event-level F1")
    ax.axhline(baseline, color="#d62728", ls="--", lw=1.1,
               label=f"flag-everything ({baseline:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=9)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 0.75)
    ax.set_title(
        "Point-wise F1 buries the strategies at the baseline; event-level F1 separates them\n"
        "(skab-valve1 — every strategy catches all 15 fault events, so they differ only in alarm volume)",
        fontsize=9, loc="left",
    )
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def figure_detection_delay(table: pd.DataFrame, path: Path) -> None:
    """How long the drift-triggered strategy took to notice each change."""
    rows = table[(table.strategy == "drift-triggered") & table.mean_delay.notna()]
    if rows.empty:
        print("  (no drift-timing rows to plot)")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(rows.stream, rows.mean_delay, color=COLOURS["drift-triggered"], width=0.6)
    for stream, delay, detected, total in zip(
        rows.stream, rows.mean_delay, rows.n_detected, rows.n_change_points
    ):
        ax.text(stream, delay, f"{delay:.0f}\n({int(detected)}/{int(total)})",
                ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("mean steps from change to detection")
    ax.set_title(
        "Drift detection delay by change shape\n(detected / total change points in brackets)",
        fontsize=10, loc="left",
    )
    ax.margins(y=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smaller run for a weak machine")
    parser.add_argument("--force", action="store_true", help="recompute even if cached")
    args = parser.parse_args()

    started = time.perf_counter()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    table, all_runs = run_all(quick=args.quick, force=args.force)

    metrics_path = RESULTS_DIR / "metrics.csv"
    table.to_csv(metrics_path, index=False)
    print(f"Wrote {metrics_path.relative_to(PROJECT_ROOT)} ({len(table)} rows)")

    print("\nGenerating figures...")
    for name in ("sudden", "recurring"):
        if name in all_runs:
            scenario, runs = all_runs[name]
            out = FIGURES_DIR / f"walkthrough_{name}.png"
            figure_stream_walkthrough(scenario, runs, out)
            print(f"  {out.relative_to(PROJECT_ROOT)}")

    figure_alarm_rates(table, FIGURES_DIR / "alarm_rates.png")
    print(f"  {(FIGURES_DIR / 'alarm_rates.png').relative_to(PROJECT_ROOT)}")

    figure_detection_delay(table, FIGURES_DIR / "detection_delay.png")
    print(f"  {(FIGURES_DIR / 'detection_delay.png').relative_to(PROJECT_ROOT)}")

    labelled = [k for k in all_runs if k.startswith("skab-")]
    if labelled:
        scenario, runs = all_runs[labelled[0]]
        out = FIGURES_DIR / "rolling_f1.png"
        figure_rolling_f1(scenario, runs, out)
        print(f"  {out.relative_to(PROJECT_ROOT)}")

    figure_event_vs_point_f1(table, FIGURES_DIR / "event_vs_point_f1.png")
    print(f"  {(FIGURES_DIR / 'event_vs_point_f1.png').relative_to(PROJECT_ROOT)}")

    print(f"\nDone in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
