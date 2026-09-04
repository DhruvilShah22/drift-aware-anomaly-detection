"""Run the robustness sweep locally and write `results/sweep.csv`.

    python scripts/run_sweep.py              # single-seed sweep, several minutes
    python scripts/run_sweep.py --seeds 5    # repeat every cell at five seeds
    python scripts/run_sweep.py --quick      # small, for a sanity check

With `--seeds` the output goes to `results/sweep_multiseed.csv` instead, and
carries a `seed` column plus a per-seed summary with bootstrap intervals in
`results/sweep_multiseed_summary.csv`. The single-seed `sweep.csv` is left alone
so the committed figures and the notebook keep resolving.

The same sweep is packaged for free CPU in `notebooks/kaggle_experiment.ipynb`.
This script exists so the code can be tested and run without a notebook.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_anomaly_free  # noqa: E402
from src.stats import summarise_seeds  # noqa: E402
from src.sweep import SweepConfig, attribution_table, summarise_sweep, sweep  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="repeat every cell at this many seeds, writing sweep_multiseed.csv",
    )
    args = parser.parse_args()

    seeds = tuple(range(args.seeds)) if args.seeds else None
    config = (
        SweepConfig(
            magnitudes=(0.0, 3.0), detectors=("adwin",),
            kinds=("sudden",), max_rows=4000, warmup_size=500, seeds=seeds,
        )
        if args.quick
        else SweepConfig(seeds=seeds)
    )

    started = time.perf_counter()
    base = load_anomaly_free()
    print(f"Base stream: {base.describe()}")
    print(
        f"Sweeping {len(config.kinds)} shapes x {len(config.magnitudes)} "
        f"magnitudes x {len(config.detectors)} detectors\n"
    )

    table = sweep(base, config)
    table = attribution_table(table)

    if args.quick:
        name = "sweep_quick.csv"
    elif seeds:
        name = "sweep_multiseed.csv"
    else:
        name = "sweep.csv"
    out = PROJECT_ROOT / "results" / name
    table.to_csv(out, index=False)
    print(f"\nWrote {out.relative_to(PROJECT_ROOT)} ({len(table)} rows)")

    if seeds and len(config.magnitudes) > 1:
        # Delay is undefined where a detector never fired, and averaging over
        # seeds would quietly drop those rows and flatter the detector. Count
        # them alongside so a missing detection stays visible.
        real = table[table.magnitude > 0].copy()
        real["detected"] = real.n_detected > 0
        summary = summarise_seeds(
            real,
            ["mean_delay", "false_alarms_per_1k", "detected"],
            ["kind", "magnitude", "detector"],
        )
        summary_path = PROJECT_ROOT / "results" / "sweep_multiseed_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Wrote {summary_path.relative_to(PROJECT_ROOT)} ({len(summary)} rows)")

    if len(config.magnitudes) > 1:
        print("\nDetection rate and delay by magnitude (averaged over shapes):")
        print(summarise_sweep(table).to_string(index=False))

        print("\nFirings attributable to the injected change (vs the 0.0 control):")
        columns = ["kind", "magnitude", "detector", "n_detections_total",
                   "control_firings", "excess_firings"]
        print(table[table.magnitude > 0][columns].to_string(index=False))

    print(f"\nDone in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
