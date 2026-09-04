"""Sensitivity of the results to the constants we picked by hand.

Three numbers in this project were chosen by judgement rather than derived, and
a reader is entitled to ask whether the conclusions rest on them:

- **warm-up size** (1,000 rows). Long enough to fill Half-Space Trees' window
  and estimate a threshold quantile, short enough to leave stream to measure.
- **matching horizon** (750 steps). A detection further than this after a change
  point is not credited to it. Described in `evaluate.py` as "a few HST windows",
  which is a justification for the order of magnitude and not for the value.
- **cooldown** (250 steps). How long a drift monitor stays quiet after firing,
  which trades responsiveness against chatter on a stream that keeps moving.

The horizon sweep is free: it re-scores detections that were already recorded,
so one run produces every horizon. The other two need re-runs.

    python scripts/run_ablation.py
    python scripts/run_ablation.py --seeds 3

Writes `results/ablation_warmup.csv`, `results/ablation_horizon.csv` and
`results/ablation_cooldown.csv`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_anomaly_free  # noqa: E402
from src.drift_injection import inject  # noqa: E402
from src.evaluate import drift_metrics  # noqa: E402
from src.experiment import (  # noqa: E402
    RunConfig,
    factorial_strategies,
    labelled_scenario,
    run_strategy,
)
from src.metrics_lit import all_metrics  # noqa: E402
from src.stats import summarise_seeds  # noqa: E402

RESULTS = ROOT / "results"

WARMUP_SIZES = (500, 1000, 2000, 3000)
HORIZONS = (250, 500, 750, 1000, 1500, 2000)
COOLDOWNS = (0, 100, 250, 500, 1000)

# The policies worth varying these under. The static baselines have no
# adaptation to be sensitive to, so they are left out of the cooldown sweep.
ABLATION_STRATEGIES = ("hst-static", "hst-periodic", "hst-drift", "iforest-static")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--max-files", type=int, default=None)
    return parser.parse_args()


def warmup_ablation(seeds, max_files=None) -> pd.DataFrame:
    """Does the ranking of the policies survive a different warm-up length?

    Note that changing the warm-up also changes what is scored: a longer warm-up
    means fewer rows evaluated, and on a stream with unevenly spaced faults that
    moves the base rate. `base_rate` and `flag_everything_f1` are carried in the
    output so the comparison stays honest across rows.
    """
    scenario = labelled_scenario(max_files=max_files)
    strategies = [s for s in factorial_strategies() if s.name in ABLATION_STRATEGIES]

    rows = []
    for warmup in WARMUP_SIZES:
        for seed in seeds:
            config = RunConfig(
                warmup_size=warmup,
                threshold_quantile=scenario.threshold_quantile,
                seed=seed,
            )
            for strategy in strategies:
                run = run_strategy(scenario.frame, strategy, config)
                start = run.eval_start
                y_true = scenario.labels.to_numpy()[start:]
                y_pred = run.flags[start:]
                base_rate = float(np.mean(y_true))
                row = {
                    "warmup_size": warmup,
                    "seed": seed,
                    "strategy": strategy.name,
                    "n_steps": len(y_pred),
                    "base_rate": base_rate,
                    "flag_everything_f1": 2 * base_rate / (1 + base_rate),
                    "alarm_rate": float(np.mean(y_pred)),
                    "n_adaptations": run.n_adaptations,
                }
                row.update(all_metrics(y_true, y_pred))
                row["composite_lift"] = row["composite_f1"] - row["flag_everything_f1"]
                rows.append(row)
        print(f"  warm-up {warmup}: done")
    return pd.DataFrame(rows)


def horizon_ablation(seeds) -> pd.DataFrame:
    """How much of the drift-timing result is the 750-step matching horizon?

    One run per (stream, seed) produces every horizon, since the horizon only
    enters when detections are matched to change points after the fact.
    """
    clean = load_anomaly_free()
    strategy = [s for s in factorial_strategies() if s.name == "hst-drift"][0]

    rows = []
    for seed in seeds:
        config = RunConfig(seed=seed, threshold_quantile=0.98)
        for kind in ("sudden", "incremental", "gradual", "recurring"):
            kwargs = {"magnitude": 3.0}
            if kind == "gradual":
                kwargs["seed"] = seed
            stream = inject(kind, clean.frame, clean.labels, **kwargs)
            run = run_strategy(stream.frame, strategy, config)

            start = run.eval_start
            detections = [d - start for d in run.detections]
            points = [c - start for c in stream.drift_points if c >= start]
            n_steps = len(run.flags) - start

            for horizon in HORIZONS:
                metrics = drift_metrics(
                    detections=detections,
                    change_points=points,
                    n_steps=n_steps,
                    horizon=horizon,
                )
                rows.append(
                    {
                        "stream": kind,
                        "seed": seed,
                        "horizon": horizon,
                        **metrics.as_dict(),
                    }
                )
        print(f"  horizon seed {seed}: done")
    return pd.DataFrame(rows)


def cooldown_ablation(seeds) -> pd.DataFrame:
    """The cooldown suppresses repeat firings. What does its value buy?

    Reported on the injected streams, where a firing can be checked against a
    known change point, so "chatty" and "responsive" are separable.
    """
    clean = load_anomaly_free()
    strategy = [s for s in factorial_strategies() if s.name == "hst-drift"][0]

    rows = []
    for cooldown in COOLDOWNS:
        for seed in seeds:
            config = RunConfig(seed=seed, threshold_quantile=0.98, cooldown=cooldown)
            for kind in ("sudden", "incremental", "gradual", "recurring"):
                kwargs = {"magnitude": 3.0}
                if kind == "gradual":
                    kwargs["seed"] = seed
                stream = inject(kind, clean.frame, clean.labels, **kwargs)
                run = run_strategy(stream.frame, strategy, config)

                start = run.eval_start
                metrics = drift_metrics(
                    detections=[d - start for d in run.detections],
                    change_points=[c - start for c in stream.drift_points if c >= start],
                    n_steps=len(run.flags) - start,
                    horizon=750,
                )
                rows.append(
                    {
                        "cooldown": cooldown,
                        "stream": kind,
                        "seed": seed,
                        "n_adaptations": run.n_adaptations,
                        "alarm_rate": float(np.mean(run.flags[start:])),
                        **metrics.as_dict(),
                    }
                )
        print(f"  cooldown {cooldown}: done")
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    seeds = tuple(range(args.seeds))
    RESULTS.mkdir(exist_ok=True)
    started = time.perf_counter()

    print(f"Ablations over {len(seeds)} seeds.\n")

    print("Warm-up size:")
    warmup = warmup_ablation(seeds, max_files=args.max_files)
    warmup.to_csv(RESULTS / "ablation_warmup.csv", index=False)

    print("\nMatching horizon:")
    horizon = horizon_ablation(seeds)
    horizon.to_csv(RESULTS / "ablation_horizon.csv", index=False)

    print("\nDetector cooldown:")
    cooldown = cooldown_ablation(seeds)
    cooldown.to_csv(RESULTS / "ablation_cooldown.csv", index=False)

    print(f"\nDone in {(time.perf_counter() - started) / 60:.1f} min.")

    print("\nComposite F1 by warm-up size (mean over seeds):")
    view = summarise_seeds(warmup, ["composite_f1"], ["warmup_size", "strategy"])
    pivot = view.pivot(index="strategy", columns="warmup_size", values="composite_f1")
    print(pivot.round(3).to_string())

    print("\nMean detection delay by horizon (hst-drift):")
    delay = horizon.groupby(["stream", "horizon"]).mean_delay.mean().unstack()
    print(delay.round(0).to_string())

    print("\nAdaptations by cooldown:")
    adapt = cooldown.groupby(["stream", "cooldown"]).n_adaptations.mean().unstack()
    print(adapt.round(1).to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
