"""Run the robustness sweep locally and write `results/sweep.csv`.

    python scripts/run_sweep.py            # full sweep, several minutes
    python scripts/run_sweep.py --quick    # small, for a sanity check

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
from src.sweep import SweepConfig, attribution_table, summarise_sweep, sweep  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    config = (
        SweepConfig(
            magnitudes=(0.0, 3.0), detectors=("adwin",),
            kinds=("sudden",), max_rows=4000, warmup_size=500,
        )
        if args.quick
        else SweepConfig()
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

    out = PROJECT_ROOT / "results" / ("sweep_quick.csv" if args.quick else "sweep.csv")
    table.to_csv(out, index=False)
    print(f"\nWrote {out.relative_to(PROJECT_ROOT)} ({len(table)} rows)")

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
