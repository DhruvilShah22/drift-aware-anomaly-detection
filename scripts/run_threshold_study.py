"""Can one threshold rule work on both datasets without being retuned?

    python scripts/run_threshold_study.py

This is the follow-up to the transfer test. That test showed a hand-set quantile
does not carry from SKAB to NAB — costing 0.093 F1 on average and beating a
trivial baseline on only half of NAB's series. The diagnosis was that a quantile
depends on something that differs between the datasets: **how contaminated the
warm-up window is**. NAB's is 0% anomalous; SKAB valve1's is about 35%.

So the question here is narrow and falsifiable:

> Is there a single threshold rule, with a single parameter setting, that does
> well on SKAB *and* NAB without per-dataset tuning?

Each candidate rule is applied unchanged to all seven streams. The rule to beat
is the status quo: the hand-tuned per-dataset quantile, which gets to keep its
advantage of having been fitted to each dataset separately.

Scored against **flag-everything** per stream, since base rates differ wildly
(35% on valve1, ~10% on NAB) and raw F1 is not comparable across them.

Writes `results/threshold_study.csv`.
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
    labelled_scenario,
    nab_scenario,
    run_strategy,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WARMUP = 1000

# One setting each, applied unchanged everywhere. No per-dataset tuning.
CANDIDATE_RULES = [
    "robust_z@2",
    "robust_z@3",
    "robust_z@5",
    "tukey@1.5",
    "tukey@3",
    "target_rate@0.05",
    "target_rate@0.10",
    "quantile@0.9",
    "quantile@0.98",
]

# The status quo it has to beat: the per-dataset quantiles chosen by hand in
# earlier phases. This one is allowed to differ per dataset, which is precisely
# the advantage a transferable rule should not need.
HAND_TUNED = {"skab-valve1": 0.50}
HAND_TUNED_DEFAULT = 0.98


def build_scenarios():
    scenarios = [labelled_scenario()]
    for series_key in NAB_SERIES:
        scenario = nab_scenario(series_key)
        if len(scenario.frame) > WARMUP + 500:
            scenarios.append(scenario)
    return scenarios


def flag_everything(labels: np.ndarray) -> float:
    base = float(np.mean(labels))
    return 2 * base / (1 + base) if base else 0.0


def main() -> int:
    started = time.perf_counter()
    scenarios = build_scenarios()
    strategies = default_strategies()
    rows: list[dict] = []

    print(f"{len(scenarios)} streams x {len(CANDIDATE_RULES) + 1} rules "
          f"x {len(strategies)} strategies\n")

    for scenario in scenarios:
        labels = scenario.labels.to_numpy()
        warm_contamination = float(scenario.labels.iloc[:WARMUP].mean())
        print(f"{scenario.name}: {len(scenario.frame)} rows, "
              f"{labels.mean():.1%} anomalous, "
              f"warm-up {warm_contamination:.1%} contaminated")

        specs = [(rule, rule) for rule in CANDIDATE_RULES]
        hand = HAND_TUNED.get(scenario.name, HAND_TUNED_DEFAULT)
        specs.append(("hand-tuned", f"quantile@{hand}"))

        for label, spec in specs:
            for strategy in strategies:
                config = RunConfig(warmup_size=WARMUP, threshold_rule=spec)
                run = run_strategy(scenario.frame, strategy, config)
                start = run.eval_start
                y_true = labels[start:]
                metrics = anomaly_metrics(y_true, run.flags[start:])
                baseline = flag_everything(y_true)

                rows.append({
                    "stream": scenario.name,
                    "dataset": "SKAB" if scenario.name.startswith("skab") else "NAB",
                    "rule": label,
                    "strategy": strategy.name,
                    "f1": metrics.f1,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "alarm_rate": float(run.flags[start:].mean()),
                    "flag_everything_f1": baseline,
                    "lift": metrics.f1 - baseline,
                    "warmup_contamination": warm_contamination,
                })

        best = max(
            (r for r in rows if r["stream"] == scenario.name),
            key=lambda r: r["lift"],
        )
        print(f"  best: {best['rule']} / {best['strategy']} "
              f"lift {best['lift']:+.3f} (F1 {best['f1']:.3f})\n")

    table = pd.DataFrame(rows)
    out = PROJECT_ROOT / "results" / "threshold_study.csv"
    table.to_csv(out, index=False)
    print(f"Wrote {out.relative_to(PROJECT_ROOT)} ({len(table)} rows)\n")

    # For each rule, take its best strategy per stream, then average the lift
    # over streams. A rule that only works on one dataset scores badly here.
    best_per_stream = (
        table.groupby(["rule", "stream", "dataset"])["lift"].max().reset_index()
    )

    print("=" * 76)
    print("Mean lift over flag-everything, best strategy per stream")
    print("=" * 76)
    summary = (
        best_per_stream.groupby("rule")
        .agg(
            mean_lift=("lift", "mean"),
            worst_stream=("lift", "min"),
            streams_beaten=("lift", lambda s: int((s > 0).sum())),
            n_streams=("lift", "size"),
        )
        .sort_values("mean_lift", ascending=False)
        .round(3)
    )
    print(summary.to_string())

    by_dataset = (
        best_per_stream.groupby(["rule", "dataset"])["lift"].mean().unstack().round(3)
    )
    print("\nMean lift split by dataset:")
    print(by_dataset.sort_values("NAB", ascending=False).to_string())

    summary.to_csv(PROJECT_ROOT / "results" / "threshold_study_summary.csv")

    # Taking the best strategy per stream flatters any rule that suits one model
    # type, even if it ruins the others. Holding the strategy fixed exposes that.
    print("\nMean lift with the strategy held fixed (no best-of cherry-picking):")
    per_strategy = (
        table.groupby(["rule", "strategy"])["lift"].mean().unstack().round(3)
    )
    print(per_strategy.loc[summary.index].to_string())
    per_strategy.to_csv(PROJECT_ROOT / "results" / "threshold_study_by_strategy.csv")

    best_rule = summary.index[0]
    hand_lift = summary.loc["hand-tuned", "mean_lift"]
    print(
        f"\nBest transferable rule: {best_rule} "
        f"(mean lift {summary.loc[best_rule, 'mean_lift']:+.3f})\n"
        f"Hand-tuned per-dataset baseline: {hand_lift:+.3f}"
    )
    print(f"\nDone in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
