"""The multi-seed grid: every strategy, every stream, repeated across seeds.

This is the experiment the paper reports. Everything before it in this project
ran once at seed 42, which is enough to show a system working and not enough to
say that one adaptation policy beats another.

What it writes into `results/`:

    multiseed.csv           one row per stream x strategy x seed
    multiseed_segments.csv  the same, split by source recording (SKAB only)
    multiseed_summary.csv   means with bootstrap intervals
    significance.csv        paired comparisons, Holm-corrected
    multiseed_drift.csv     drift timing on the injected streams

Runtime is about 12 minutes for the default ten seeds on one laptop core. Almost
all of it is the labelled SKAB stream; the NAB series are short.

    python scripts/run_multiseed.py                # the full grid
    python scripts/run_multiseed.py --seeds 3      # a quick look
    python scripts/run_multiseed.py --skip-nab     # SKAB only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment import (  # noqa: E402
    NAB_SERIES,
    RunConfig,
    factorial_strategies,
    labelled_scenario,
    nab_scenario,
)
from src.multiseed import injected_seed_runs, labelled_seed_runs  # noqa: E402
from src.stats import holm_bonferroni, paired_compare, summarise_seeds  # noqa: E402

RESULTS = ROOT / "results"

# The metrics carried through to the summary table. Point-wise and composite are
# the two the paper argues between; range-based is the tie-breaker we trust most;
# point-adjust is reported to show what it does, not because we rely on it.
SUMMARY_METRICS = [
    "point_f1",
    "composite_f1",
    "range_f1",
    "pa_f1",
    "pak_f1",
    "point_lift",
    "composite_lift",
    "alarm_rate",
    "n_adaptations",
    "rows_per_second",
]

# Comparisons the paper actually makes, and therefore the family the multiplicity
# correction is applied over. Testing all 45 pairs and correcting for all 45
# would be honest but needlessly weak; these are the questions being asked.
KEY_COMPARISONS = [
    ("hst-periodic", "hst-static"),
    ("hst-drift", "hst-static"),
    ("hst-periodic", "hst-drift"),
    ("hst-periodic", "hst-online"),
    ("iforest-periodic", "iforest-static"),
    ("iforest-drift", "iforest-static"),
    ("hst-periodic", "iforest-periodic"),
    ("hst-static", "iforest-static"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10, help="how many seeds to run")
    parser.add_argument("--skip-nab", action="store_true", help="SKAB only")
    parser.add_argument("--skip-drift", action="store_true", help="skip injected streams")
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="limit the SKAB recordings used, for a fast smoke run",
    )
    return parser.parse_args()


def significance_table(frame: pd.DataFrame, metric: str, unit: str) -> pd.DataFrame:
    """Paired comparisons for the questions the paper asks, Holm-corrected.

    `unit` is what the pairing is over — "seed" or "segment". Both are reported
    because they answer different questions: seeds ask whether the gap survives
    the randomness, recordings ask whether it survives the data.
    """
    # Pairing on recordings needs exactly one value per (strategy, recording),
    # and the raw frame still carries one row per seed. Averaging the seeds away
    # first is what makes the recording the unit of replication; leaving them in
    # would pair a strategy's seed 0 against another's seed 7 and count it as
    # sixteen independent observations when it is not.
    if unit == "segment" and "seed" in frame.columns:
        frame = (
            frame.groupby(["stream", "strategy", "segment"], as_index=False)[metric]
            .mean()
        )

    rows = []
    for stream, block in frame.groupby("stream"):
        available = set(block.strategy.unique())
        for a, b in KEY_COMPARISONS:
            if a not in available or b not in available:
                continue
            comparison = paired_compare(block, metric, a, b, pair_on=unit)
            row = {"stream": stream, "unit": unit, **comparison.as_dict()}
            rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    # Correct within each stream: the family is "the claims made about this
    # stream", not "every claim in the paper", since the streams are analysed
    # and reported separately.
    table["reject_holm"] = False
    for stream, block in table.groupby("stream"):
        decisions = holm_bonferroni(block.p_value.to_numpy())
        table.loc[block.index, "reject_holm"] = decisions
    table["significant_holm"] = table.reject_holm & ~table.underpowered
    return table


def main() -> int:
    args = parse_args()
    seeds = tuple(range(args.seeds))
    strategies = factorial_strategies()
    RESULTS.mkdir(exist_ok=True)

    print(
        f"Multi-seed grid: {len(strategies)} strategies x {len(seeds)} seeds"
        f"{'' if args.skip_nab else f' x {1 + len(NAB_SERIES)} streams'}"
    )
    started = time.perf_counter()

    stream_frames, segment_frames = [], []

    def checkpoint() -> None:
        """Write what has been computed so far.

        The full grid takes the better part of an hour, and writing only at the
        end means a crash in the last stream throws away every stream before it.
        Each completed stream is flushed to the same files the run would have
        written anyway, so an interrupted run leaves usable partial results.
        """
        if stream_frames:
            pd.concat(stream_frames, ignore_index=True).to_csv(
                RESULTS / "multiseed.csv", index=False
            )
        usable = [f for f in segment_frames if not f.empty]
        if usable:
            pd.concat(usable, ignore_index=True).to_csv(
                RESULTS / "multiseed_segments.csv", index=False
            )

    print("\nSKAB valve1 (labelled):")
    skab = labelled_scenario(max_files=args.max_files)
    skab_stream, skab_segments = labelled_seed_runs(
        skab, strategies, seeds=seeds, config=RunConfig()
    )
    stream_frames.append(skab_stream)
    segment_frames.append(skab_segments)
    checkpoint()

    if not args.skip_nab:
        for key in NAB_SERIES:
            name = key.split("/")[-1].replace(".csv", "")
            print(f"\nNAB {name}:")
            # 0.98 is the SKAB-family setting carried over unchanged. Holding it
            # fixed is the point: this measures what transfer costs, so retuning
            # per series here would answer a different question.
            scenario = nab_scenario(key, threshold_quantile=0.98)
            rows, _ = labelled_seed_runs(
                scenario, strategies, seeds=seeds, config=RunConfig()
            )
            stream_frames.append(rows)
            checkpoint()

    stream_rows = pd.concat(stream_frames, ignore_index=True)
    segment_rows = (
        pd.concat(segment_frames, ignore_index=True) if segment_frames else pd.DataFrame()
    )

    stream_rows.to_csv(RESULTS / "multiseed.csv", index=False)
    if not segment_rows.empty:
        segment_rows.to_csv(RESULTS / "multiseed_segments.csv", index=False)

    summary = summarise_seeds(stream_rows, SUMMARY_METRICS, ["stream", "strategy"])
    summary = summary.sort_values(["stream", "composite_f1"], ascending=[True, False])
    summary.to_csv(RESULTS / "multiseed_summary.csv", index=False)

    tables = []
    for metric in ("point_f1", "composite_f1", "range_f1"):
        tables.append(significance_table(stream_rows, metric, "seed"))
        if not segment_rows.empty:
            tables.append(significance_table(segment_rows, metric, "segment"))
    significance = pd.concat([t for t in tables if not t.empty], ignore_index=True)
    significance.to_csv(RESULTS / "significance.csv", index=False)

    if not args.skip_drift:
        print("\nInjected streams (drift timing):")
        drift_strategies = [s for s in strategies if s.adapt == "drift"]
        drift_rows = injected_seed_runs(drift_strategies, seeds=seeds)
        drift_rows.to_csv(RESULTS / "multiseed_drift.csv", index=False)

    elapsed = time.perf_counter() - started
    print(f"\nDone in {elapsed / 60:.1f} min. Wrote:")
    for name in (
        "multiseed.csv",
        "multiseed_segments.csv",
        "multiseed_summary.csv",
        "significance.csv",
        "multiseed_drift.csv",
    ):
        path = RESULTS / name
        if path.exists():
            print(f"  results/{name}")

    print("\nSKAB valve1, composite F1 (mean [95% CI]):")
    view = summary[summary.stream.str.startswith("skab")]
    for _, row in view.iterrows():
        # A star marks a strategy whose mean composite F1 clears the
        # flag-everything baseline, which most of them do not.
        marker = "*" if row.composite_lift_mean > 0 else " "
        print(
            f" {marker} {row.strategy:<17} "
            f"{row.composite_f1_mean:.3f} "
            f"[{row.composite_f1_ci_low:.3f}, {row.composite_f1_ci_high:.3f}]"
            f"   point {row.point_f1_mean:.3f}   alarm {row.alarm_rate_mean:.3f}"
        )

    if not significance.empty:
        held = significance[
            (significance.unit == "seed")
            & (significance.metric == "composite_f1")
            & significance.stream.str.startswith("skab")
        ]
        print("\nPaired over seeds, SKAB valve1, composite F1:")
        for _, row in held.iterrows():
            verdict = "significant" if row.significant_holm else "not significant"
            print(
                f"  {row.strategy_a:<17} vs {row.strategy_b:<17} "
                f"{row.mean_diff:+.3f} [{row.ci_low:+.3f}, {row.ci_high:+.3f}]  "
                f"p={row.p_value:.4f}  {verdict}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
