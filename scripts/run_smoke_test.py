"""A fast end-to-end run proving the pipeline works, on a small slice of data.

    python scripts/run_smoke_test.py

This is deliberately small so it finishes in well under a minute on a modest
machine. It is not the real experiment — for that see `scripts/run_experiment.py`
or the Kaggle notebook. What it proves is that data loads, drift injects,
strategies run, and metrics come out non-degenerate.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.experiment import (  # noqa: E402
    RunConfig,
    Scenario,
    compare,
    default_strategies,
    injected_scenarios,
)

# Small enough to be fast, long enough that a 400-row warm-up leaves real stream
# behind it and HST's 150-row window fills several times over.
SMOKE_ROWS = 3000
SMOKE_CONFIG = RunConfig(warmup_size=400, hst_n_trees=15, hst_window_size=150, cooldown=200)


def check(condition: bool, message: str) -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {message}")
    return condition


def main() -> int:
    started = time.perf_counter()
    print(f"Smoke test: {SMOKE_ROWS} rows, warm-up {SMOKE_CONFIG.warmup_size}\n")

    scenario: Scenario = injected_scenarios(
        magnitude=3.0, max_rows=SMOKE_ROWS, kinds=("sudden",)
    )[0]
    print(f"Stream '{scenario.name}': {len(scenario.frame)} rows, "
          f"change points {scenario.change_points}\n")

    table, runs = compare(scenario, default_strategies(), SMOKE_CONFIG)
    print()

    ok = True
    ok &= check(len(table) == len(default_strategies()), "every strategy produced a row")
    ok &= check(
        all(np.isfinite(runs[s].scores[SMOKE_CONFIG.warmup_size:]).all() for s in runs),
        "all post-warm-up scores are finite",
    )

    drift_run = runs["drift-triggered"]
    ok &= check(drift_run.n_adaptations > 0, "the drift-triggered strategy adapted at least once")

    point = scenario.change_points[0]
    after = [d for d in drift_run.detections if d >= point]
    ok &= check(bool(after), f"drift was detected after the injected change at row {point}")
    if after:
        ok &= check(after[0] - point < 750, f"detection delay was {after[0] - point} rows")

    static_alarms = float(table.loc[table.strategy == "static", "alarm_rate"].iloc[0])
    drift_alarms = float(table.loc[table.strategy == "drift-triggered", "alarm_rate"].iloc[0])
    print(f"\n  static alarm rate {static_alarms:.3f} vs "
          f"drift-triggered {drift_alarms:.3f} (all flags here are false alarms)")

    print(f"\nFinished in {time.perf_counter() - started:.1f}s")
    if not ok:
        print("\nSMOKE TEST FAILED", file=sys.stderr)
        return 1
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
