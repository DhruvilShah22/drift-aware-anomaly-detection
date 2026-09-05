"""Repeated runs across seeds, which is what turns the single-run tables in the
earlier phases of this project into something a claim can rest on.

Every number reported before this module existed came from one run at seed 42.
That is enough to demonstrate a system and not enough to say that one policy
beats another, because nothing in it separates a real gap from the spread a
policy shows when only the seed moves. Here a seed drives three things at once:

1. the model — Half-Space Trees' tree construction, Isolation Forest's subsample;
2. the stream, wherever the stream itself is stochastic (the gradual injection
   interleaves the two regimes at random, so it genuinely varies; sudden,
   incremental and recurring are deterministic given the base recording, and we
   say so rather than implying more replication than exists);
3. the drift detector, for KSWIN, which samples its reference window.

Runs are paired: at seed *s* every strategy sees the same stream and the same
warm-up, so a difference between two strategies at that seed is attributable to
the policy. `src/stats.py` does the aggregation and the paired tests.

There are two units of replication here and they answer different questions.
Seeds measure how much a strategy moves when the randomness moves. The 16 source
recordings behind the labelled SKAB stream measure how much it moves when the
*data* moves, and since every strategy sees all 16, they support a paired test
with real independent units. We report both, because a strategy that is stable
across seeds and wildly unstable across recordings has not been shown to work.
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pandas as pd

from src.data_loader import load_anomaly_free
from src.drift_injection import inject
from src.evaluate import drift_metrics
from src.experiment import (
    STRATEGY_FACTORS,
    RunConfig,
    Scenario,
    Strategy,
    run_strategy,
)
from src.models import DegenerateWarmUp
from src.metrics_lit import all_metrics, top_one_hit

# Ten seeds is the floor we were willing to report. Below six the signed-rank
# test cannot reach significance at any effect size (see stats.MIN_PAIRS), and
# ten keeps the bootstrap interval stable while the full grid still finishes in
# well under an hour on one laptop core.
DEFAULT_SEEDS = tuple(range(10))


def _flag_everything_f1(y_true: np.ndarray) -> float:
    """The F1 a detector gets by flagging every single row.

    At a base rate b this is 2b/(1+b), and it is the number every result in this
    project is quoted against. Reporting an F1 of 0.51 without it beside a base
    rate of 0.34 would be meaningless.
    """
    base_rate = float(np.mean(y_true))
    return 2 * base_rate / (1 + base_rate) if base_rate else 0.0


def _stream_row(
    scenario: Scenario,
    strategy_name: str,
    seed: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_adaptations: int,
    seconds: float,
    scores: np.ndarray | None = None,
) -> dict:
    model, policy = STRATEGY_FACTORS.get(strategy_name, ("?", "?"))
    row: dict = {
        "stream": scenario.name,
        "strategy": strategy_name,
        "model": model,
        "policy": policy,
        "seed": seed,
        "n_steps": int(len(y_pred)),
        "n_adaptations": int(n_adaptations),
        "alarm_rate": float(np.mean(y_pred)),
        "base_rate": float(np.mean(y_true)),
        "flag_everything_f1": _flag_everything_f1(y_true),
        "seconds": round(seconds, 3),
        "rows_per_second": round(len(y_pred) / seconds, 1) if seconds > 0 else np.nan,
    }
    row.update(all_metrics(y_true, y_pred))
    # Lift over the trivial baseline, on the two metrics the paper leads with.
    row["point_lift"] = row["point_f1"] - row["flag_everything_f1"]
    row["composite_lift"] = row["composite_f1"] - row["flag_everything_f1"]
    # The UCR archive's own criterion, which needs no threshold. Recorded for
    # every stream so the two protocols can be compared on the same runs.
    if scores is not None:
        row["top1_hit"] = top_one_hit(y_true, scores)
    return row


def labelled_seed_runs(
    scenario: Scenario,
    strategies: list[Strategy],
    seeds=DEFAULT_SEEDS,
    config: RunConfig | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every strategy over a labelled stream once per seed.

    Returns the stream-level rows and the per-segment rows. The segment frame is
    empty for a scenario with no recorded segments, which is every NAB series.
    """
    config = config or RunConfig()
    config = replace(
        config,
        threshold_quantile=scenario.threshold_quantile,
        threshold_rule=scenario.threshold_rule or config.threshold_rule,
    )

    stream_rows: list[dict] = []
    segment_rows: list[dict] = []

    for seed in seeds:
        seeded = replace(config, seed=seed)
        for strategy in strategies:
            started = time.perf_counter()
            try:
                run = run_strategy(scenario.frame, strategy, seeded)
            except DegenerateWarmUp as failure:
                # One baseline being unfittable on one stream is a fact about
                # that baseline, and the grid should record it and carry on
                # instead of throwing away every other result in the run.
                print(f"    {strategy.name} on {scenario.name}: {failure}")
                model, policy = STRATEGY_FACTORS.get(strategy.name, ("?", "?"))
                stream_rows.append({
                    "stream": scenario.name,
                    "strategy": strategy.name,
                    "model": model,
                    "policy": policy,
                    "seed": seed,
                    "fit_failed": True,
                })
                continue
            elapsed = time.perf_counter() - started

            start = run.eval_start
            y_true = scenario.labels.to_numpy()[start:]
            y_pred = run.flags[start:]

            row = _stream_row(
                scenario, strategy.name, seed, y_true, y_pred,
                run.n_adaptations, elapsed, scores=run.scores[start:],
            )
            row["fit_failed"] = False
            stream_rows.append(row)

            for seg_start, seg_end, seg_name in scenario.segments:
                # Only the part of a recording past the warm-up was ever scored.
                lo, hi = max(seg_start, start), seg_end
                if hi - lo < 50:
                    continue
                seg_true = scenario.labels.to_numpy()[lo:hi]
                seg_pred = run.flags[lo:hi]
                # A recording with no fault at all cannot produce a meaningful
                # F1 for anyone, so it is dropped rather than scored as zero.
                if seg_true.sum() == 0:
                    continue
                model, policy = STRATEGY_FACTORS.get(strategy.name, ("?", "?"))
                seg_row = {
                    "stream": scenario.name,
                    "segment": seg_name,
                    "strategy": strategy.name,
                    "model": model,
                    "policy": policy,
                    "seed": seed,
                    "n_steps": int(hi - lo),
                    "alarm_rate": float(np.mean(seg_pred)),
                    "base_rate": float(np.mean(seg_true)),
                    "flag_everything_f1": _flag_everything_f1(seg_true),
                }
                seg_row.update(all_metrics(seg_true, seg_pred))
                seg_row["point_lift"] = seg_row["point_f1"] - seg_row["flag_everything_f1"]
                seg_row["composite_lift"] = (
                    seg_row["composite_f1"] - seg_row["flag_everything_f1"]
                )
                segment_rows.append(seg_row)

        if verbose:
            done = [
                r for r in stream_rows
                if r["seed"] == seed and not r.get("fit_failed")
            ]
            if not done:
                continue
            best = max(done, key=lambda r: r["composite_f1"])
            print(
                f"  seed {seed:>2}: best composite F1 {best['composite_f1']:.3f} "
                f"({best['strategy']}), {len(done)} strategies"
            )

    return pd.DataFrame(stream_rows), pd.DataFrame(segment_rows)


def injected_seed_runs(
    strategies: list[Strategy],
    seeds=DEFAULT_SEEDS,
    magnitude: float = 3.0,
    kinds: tuple[str, ...] = ("sudden", "incremental", "gradual", "recurring"),
    config: RunConfig | None = None,
    horizon: int = 750,
    verbose: bool = True,
) -> pd.DataFrame:
    """Drift timing and false alarms on the injected streams, once per seed.

    These streams contain no true anomalies by construction, so every flag is a
    false alarm and there is no F1 to report. What they measure is how long a
    detector takes to notice a change whose position is known exactly.

    The gradual injection is re-drawn per seed because its interleaving is
    random. The other three shapes are deterministic given the base recording,
    so their across-seed spread reflects the model and detector only. That is a
    real limit on what the intervals mean and it is recorded in the output as
    `stream_is_stochastic`.
    """
    config = config or RunConfig()
    clean = load_anomaly_free()

    rows: list[dict] = []
    for seed in seeds:
        seeded = replace(config, seed=seed, threshold_quantile=0.98)
        for kind in kinds:
            # Only the gradual injection takes a seed; passing one to the others
            # would be a lie about where the variation comes from.
            kwargs = {"magnitude": magnitude}
            if kind == "gradual":
                kwargs["seed"] = seed
            stream = inject(kind, clean.frame, clean.labels, **kwargs)

            scenario = Scenario(
                name=kind,
                frame=stream.frame,
                labels=stream.labels,
                change_points=stream.drift_points,
                has_true_anomalies=False,
                threshold_quantile=0.98,
            )

            for strategy in strategies:
                started = time.perf_counter()
                run = run_strategy(scenario.frame, strategy, seeded)
                elapsed = time.perf_counter() - started

                start = run.eval_start
                y_pred = run.flags[start:]
                points = [c - start for c in scenario.change_points if c >= start]
                drift = drift_metrics(
                    detections=[d - start for d in run.detections],
                    change_points=points,
                    n_steps=len(y_pred),
                    horizon=horizon,
                )
                row = {
                    "stream": kind,
                    "strategy": strategy.name,
                    "detector": strategy.detector_kind if strategy.adapt == "drift" else "",
                    "seed": seed,
                    "magnitude": magnitude,
                    "stream_is_stochastic": kind == "gradual",
                    "n_steps": len(y_pred),
                    "alarm_rate": float(np.mean(y_pred)),
                    "n_adaptations": run.n_adaptations,
                    "seconds": round(elapsed, 3),
                }
                row.update(drift.as_dict())
                rows.append(row)

        if verbose:
            print(f"  seed {seed:>2}: {len(kinds)} streams x {len(strategies)} strategies")

    return pd.DataFrame(rows)
