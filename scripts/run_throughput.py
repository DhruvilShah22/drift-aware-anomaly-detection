"""Measure per-strategy throughput on a quiet machine.

`run_multiseed.py` records elapsed time too, but it runs for the better part of
an hour and anything else happening on the machine during it lands in those
numbers. Cost is the one measurement here that is sensitive to what else the CPU
is doing, so it gets its own short run, and the paper's cost table reads this
file rather than the timings embedded in the grid.

Run it with nothing else going on:

    python scripts/run_throughput.py
    python scripts/run_throughput.py --repeats 5

Writes `results/throughput.csv`: one row per strategy per repeat, with the
median reported in the paper because a single slow repeat is far more likely to
be interference than a real property of the code.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment import (  # noqa: E402
    STRATEGY_FACTORS,
    RunConfig,
    factorial_strategies,
    labelled_scenario,
    run_strategy,
)

RESULTS = ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-files", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenario = labelled_scenario(max_files=args.max_files)
    strategies = factorial_strategies()
    base = RunConfig(threshold_quantile=scenario.threshold_quantile)

    print(
        f"Timing {len(strategies)} strategies over {len(scenario.frame):,} rows, "
        f"{args.repeats} repeats each."
    )

    rows = []
    for repeat in range(args.repeats):
        for strategy in strategies:
            config = replace(base, seed=repeat)
            started = time.perf_counter()
            run = run_strategy(scenario.frame, strategy, config)
            elapsed = time.perf_counter() - started
            scored = len(run.flags) - run.eval_start
            model, policy = STRATEGY_FACTORS.get(strategy.name, ("?", "?"))
            rows.append({
                "strategy": strategy.name,
                "model": model,
                "policy": policy,
                "repeat": repeat,
                "n_scored": scored,
                "seconds": round(elapsed, 4),
                "rows_per_second": round(scored / elapsed, 1),
                "n_adaptations": run.n_adaptations,
            })
        print(f"  repeat {repeat}: done")

    table = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / "throughput.csv", index=False)
    print(f"\nWrote results/throughput.csv ({len(table)} rows)\n")

    summary = (
        table.groupby("strategy")
        .agg(
            rows_per_second=("rows_per_second", "median"),
            seconds=("seconds", "median"),
            n_adaptations=("n_adaptations", "median"),
        )
        .sort_values("rows_per_second", ascending=False)
    )
    print(summary.round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
