"""Generate the paper's LaTeX tables from the result CSVs.

The manuscript does not contain a hand-typed number anywhere. Every table is
written here, straight out of `results/*.csv`, and `paper/main.tex` includes the
files. That removes the failure mode where a table is edited and a sentence
about it is not, and it means regenerating the experiments regenerates the
paper.

    python scripts/make_paper_tables.py

Writes into `paper/tables/`. Also prints a short digest of the numbers the prose
quotes, so the text can be checked against them in one glance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
TABLES = ROOT / "paper" / "tables"

STRATEGY_ORDER = [
    "hst-static",
    "hst-online",
    "hst-periodic",
    "hst-drift",
    "iforest-static",
    "iforest-periodic",
    "iforest-drift",
    "lof-static",
    "ocsvm-static",
    "elliptic-static",
]

PRETTY = {
    "hst-static": r"HST, never",
    "hst-online": r"HST, online",
    "hst-periodic": r"HST, periodic",
    "hst-drift": r"HST, drift",
    "iforest-static": r"iForest, never",
    "iforest-periodic": r"iForest, periodic",
    "iforest-drift": r"iForest, drift",
    "lof-static": r"LOF, never",
    "ocsvm-static": r"OC-SVM, never",
    "elliptic-static": r"MCD, never",
}


def load(name: str) -> pd.DataFrame | None:
    path = RESULTS / name
    if not path.exists():
        print(f"  missing: results/{name}")
        return None
    return pd.read_csv(path)


def write(name: str, body: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / name).write_text(body, encoding="utf-8")
    print(f"  paper/tables/{name}")


def ci(row: pd.Series, metric: str, digits: int = 3) -> str:
    """A mean with its interval, formatted for a table cell."""
    mean = row[f"{metric}_mean"]
    low, high = row[f"{metric}_ci_low"], row[f"{metric}_ci_high"]
    if np.isnan(mean):
        return "--"
    # A degenerate interval means the estimator is deterministic given the data,
    # which is true of LOF, OC-SVM and MCD. Saying so is more useful than
    # printing a zero-width interval that looks like a rounding artefact.
    if np.isclose(low, high):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} \\ci{{{low:.{digits}f}}}{{{high:.{digits}f}}}"


def trivial_baseline_scores(warmup: int = 1000) -> dict[str, float]:
    """Score the flag-everything strategy under every convention.

    Its point-wise and composite F1 are both 2b/(1+b) at base rate b, but its
    range-based F1 is not: flagging every row produces a single predicted range
    spanning the stream, which the cardinality factor divides by the number of
    true events it swallows. That number has to be computed rather than assumed,
    and getting it wrong would flatter every detector in the table.

    Computed from the labels directly, so it needs no run.
    """
    from src.experiment import labelled_scenario
    from src.metrics_lit import all_metrics

    scenario = labelled_scenario()
    y_true = scenario.labels.to_numpy()[warmup:]
    y_pred = np.ones_like(y_true)
    return all_metrics(y_true, y_pred)


def table_main(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """The headline table: every strategy under every scoring convention."""
    skab = summary[summary.stream.str.startswith("skab")].set_index("strategy")
    present = [s for s in STRATEGY_ORDER if s in skab.index]
    baseline = float(
        rows[rows.stream.str.startswith("skab")].flag_everything_f1.iloc[0]
    )
    n_seeds = int(skab.n_seeds.iloc[0])

    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r" & \multicolumn{4}{c}{F1 under each convention} & \\",
        r"\cmidrule(lr){2-5}",
        r"Model, policy & Point-wise & Composite & Range & Point-adj. "
        r"& Alarm rate \\",
        r"\midrule",
    ]
    for name in present:
        row = skab.loc[name]
        lines.append(
            f"{PRETTY[name]} & {ci(row, 'point_f1')} & {ci(row, 'composite_f1')} "
            f"& {ci(row, 'range_f1')} & {ci(row, 'pa_f1')} "
            f"& {row['alarm_rate_mean']:.3f} \\\\"
        )
    trivial = trivial_baseline_scores()
    # Figures need the same reference values, and they read CSVs only, so put it
    # where they can find it rather than recomputing it there.
    pd.DataFrame([trivial]).to_csv(RESULTS / "trivial_baseline.csv", index=False)
    lines += [
        r"\midrule",
        f"Flag everything & {trivial['point_f1']:.3f} & {trivial['composite_f1']:.3f}"
        f" & {trivial['range_f1']:.3f} & {trivial['pa_f1']:.3f} & 1.000 \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write("main_results.tex", "\n".join(lines) + "\n")
    return n_seeds, baseline, trivial


def table_significance(significance: pd.DataFrame) -> None:
    """Paired comparisons on the labelled stream, over seeds and recordings."""
    skab = significance[significance.stream.str.startswith("skab")]
    if skab.empty:
        return

    lines = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Comparison & Metric & Unit & Mean diff. [95\% CI] & $p$ & Holm \\",
        r"\midrule",
    ]
    for metric in ("point_f1", "composite_f1"):
        block = skab[skab.metric == metric]
        for unit in ("seed", "segment"):
            rows = block[block.unit == unit]
            for _, row in rows.iterrows():
                verdict = r"\yes" if row.significant_holm else r"\no"
                label = f"{PRETTY.get(row.strategy_a, row.strategy_a)} vs.\\ {PRETTY.get(row.strategy_b, row.strategy_b)}"
                metric_name = "point" if metric == "point_f1" else "composite"
                lines.append(
                    f"{label} & {metric_name} & {unit} & "
                    f"${row.mean_diff:+.3f}$ "
                    f"$[{row.ci_low:+.3f}, {row.ci_high:+.3f}]$ & "
                    f"{row.p_value:.4f} & {verdict} \\\\"
                )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write("significance.tex", "\n".join(lines) + "\n")


def table_transfer(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """NAB, SKAB threshold carried over unchanged, best strategy per series."""
    nab = summary[~summary.stream.str.startswith(("skab", "ucr-"))]
    if nab.empty:
        return
    baselines = rows.groupby("stream").flag_everything_f1.first()

    best = (
        nab.sort_values("composite_f1_mean", ascending=False).groupby("stream").head(1)
    )
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"NAB series & Best strategy & Flag-all & Point-wise F1 & Composite F1 & Beats \\",
        r"\midrule",
    ]
    for _, row in best.sort_values("stream").iterrows():
        base = baselines[row.stream]
        beats = r"\yes" if row.composite_f1_mean > base else r"\no"
        short = (
            row.stream.replace("_system_failure", "")
            .replace("_misconfiguration", "")
            .replace("_", r"\_")
        )
        lines.append(
            f"{short} & {PRETTY.get(row.strategy, row.strategy)} & {base:.3f} & "
            f"{ci(row, 'point_f1')} & {ci(row, 'composite_f1')} & {beats} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("transfer.tex", "\n".join(lines) + "\n")


def table_detectors(sweep_summary: pd.DataFrame) -> None:
    """Detection delay per drift shape, at a 3 sd shift, per detector."""
    if sweep_summary is None or sweep_summary.empty:
        return
    at3 = sweep_summary[np.isclose(sweep_summary.magnitude, 3.0)]
    if at3.empty:
        return

    shapes = [k for k in ("sudden", "incremental", "gradual", "recurring") if k in set(at3.kind)]
    detectors = sorted(at3.detector.unique())

    lines = [
        r"\begin{tabular}{l" + "c" * len(shapes) + "c}",
        r"\toprule",
        "Detector & " + " & ".join(shapes) + r" & False alarms / 1k \\",
        r"\midrule",
    ]
    for detector in detectors:
        cells = []
        for shape in shapes:
            row = at3[(at3.detector == detector) & (at3.kind == shape)]
            if row.empty or np.isnan(row.mean_delay_mean.iloc[0]):
                cells.append("--")
            else:
                cells.append(ci(row.iloc[0], "mean_delay", digits=0))
        alarms = at3[at3.detector == detector].false_alarms_per_1k_mean.mean()
        lines.append(
            f"\\texttt{{{detector.replace('_', chr(92) + '_')}}} & "
            + " & ".join(cells)
            + f" & {alarms:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("detectors.tex", "\n".join(lines) + "\n")


def table_ucr(rows: pd.DataFrame) -> pd.DataFrame | None:
    """The UCR subset, where event recall finally discriminates.

    Every series carries exactly one anomaly, so `composite_recall` is 0 or 1
    per run and its mean over series and seeds is the proportion of anomalies a
    strategy actually found. On SKAB and NAB that number is 1.0 for everything,
    which is why the composite score collapses to a precision ranking there.
    """
    ucr = rows[rows.stream.str.startswith("ucr-")]
    if ucr.empty:
        return None

    aggregations = dict(
        found=("composite_recall", "mean"),
        composite=("composite_f1", "mean"),
        point=("point_f1", "mean"),
        alarm=("alarm_rate", "mean"),
        baseline=("flag_everything_f1", "mean"),
    )
    has_top1 = "top1_hit" in ucr.columns and ucr.top1_hit.notna().any()
    if has_top1:
        aggregations["top1"] = ("top1_hit", "mean")
    grouped = ucr.groupby("strategy").agg(**aggregations)
    grouped = grouped.reindex([s for s in STRATEGY_ORDER if s in grouped.index])

    columns = "lcccc" + ("c" if has_top1 else "")
    header = r"Model, policy & "
    if has_top1:
        header += r"Peak in window & "
    header += r"Anomalies found & Composite F1 & Point-wise F1 & Alarm rate \\"

    lines = [
        r"\begin{tabular}{" + columns + "}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for name, row in grouped.iterrows():
        cells = [PRETTY.get(name, name)]
        if has_top1:
            cells.append(f"{row.top1:.2f}")
        cells += [f"{row.found:.2f}", f"{row.composite:.3f}",
                  f"{row.point:.3f}", f"{row.alarm:.3f}"]
        lines.append(" & ".join(cells) + r" \\")
    trivial_cells = ["Flag everything"]
    if has_top1:
        # Flagging every row assigns one score to everything, so there is no
        # peak to land anywhere. The criterion simply does not apply to it.
        trivial_cells.append("--")
    trivial_cells += ["1.00", f"{grouped.baseline.iloc[0]:.3f}",
                      f"{grouped.baseline.iloc[0]:.3f}", "1.000"]
    lines += [
        r"\midrule",
        " & ".join(trivial_cells) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write("ucr.tex", "\n".join(lines) + "\n")
    return grouped


def table_cost(summary: pd.DataFrame, throughput: pd.DataFrame | None) -> None:
    """Throughput, because a streaming method that cannot keep up is not one.

    Prefers `results/throughput.csv`, which is measured on a quiet machine by
    `scripts/run_throughput.py`. The timings recorded inside the multi-seed grid
    are the fallback, and they are noisier: that run takes the better part of an
    hour, so anything else on the machine during it lands in the numbers.
    """
    skab = summary[summary.stream.str.startswith("skab")].set_index("strategy")
    present = [s for s in STRATEGY_ORDER if s in skab.index]

    if throughput is not None and not throughput.empty:
        speed = throughput.groupby("strategy").rows_per_second.median()
        source = "quiet"
    else:
        speed = skab["rows_per_second_mean"]
        source = "grid"

    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Model, policy & Rows / second & Rebuilds \\",
        r"\midrule",
    ]
    for name in present:
        rate = speed.get(name, float("nan"))
        rebuilds = skab.loc[name, "n_adaptations_mean"]
        rendered = "--" if np.isnan(rate) else f"{rate:,.0f}"
        lines.append(f"{PRETTY[name]} & {rendered} & {rebuilds:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("cost.tex", "\n".join(lines) + "\n")
    return speed, source


def ordinal(n: int) -> str:
    """1 -> 1st, 2 -> 2nd, 11 -> 11th. The prose says "places \\MoverPointRank",
    so the suffix has to come from here; appending a literal "th" in the text
    produces "2th" for every rank that is not a teen."""
    n = int(n)
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def table_numbers(summary, rows, significance, segments, speed=None,
                  speed_source="grid", trivial=None, ucr=None) -> dict:
    """Emit every number the prose quotes as a LaTeX macro.

    The manuscript writes \\FlagAll rather than 0.512, so a sentence cannot end
    up disagreeing with the table beside it. If a macro is undefined the paper
    fails to build, which is the behaviour we want: it is louder than a stale
    number sitting quietly in a paragraph.
    """
    skab = summary[summary.stream.str.startswith("skab")].set_index("strategy")
    skab_rows = rows[rows.stream.str.startswith("skab")]
    values: dict[str, str] = {}

    def put(name: str, value) -> None:
        values[name] = value if isinstance(value, str) else f"{value}"

    put("NSeeds", int(skab.n_seeds.iloc[0]))
    put("NStrategies", len(skab))
    put("FlagAll", f"{float(skab_rows.flag_everything_f1.iloc[0]):.3f}")
    put("BaseRate", f"{100 * float(skab_rows.base_rate.iloc[0]):.1f}")
    put("NRows", f"{int(skab_rows.n_steps.iloc[0]):,}")

    metric_macro = {
        "point_f1": "Point",
        "composite_f1": "Composite",
        "range_f1": "Range",
        "pa_f1": "PA",
    }
    winners = {}
    for metric, tag in metric_macro.items():
        ordered = skab[f"{metric}_mean"].sort_values(ascending=False)
        winners[metric] = ordered.index[0]
        put(f"Best{tag}", PRETTY.get(ordered.index[0], ordered.index[0]))
        put(f"Best{tag}Val", f"{ordered.iloc[0]:.3f}")
        put(f"Worst{tag}", PRETTY.get(ordered.index[-1], ordered.index[-1]))
        put(f"Worst{tag}Val", f"{ordered.iloc[-1]:.3f}")

    put("NDistinctWinners", len(set(winners.values())))

    # How many strategies change rank between point-wise and composite scoring,
    # which is the paper's headline quantity.
    point_rank = skab["point_f1_mean"].rank(ascending=False, method="min")
    composite_rank = skab["composite_f1_mean"].rank(ascending=False, method="min")
    moved = (point_rank != composite_rank).sum()
    put("NRankChanges", int(moved))
    put("MaxRankJump", int((point_rank - composite_rank).abs().max()))

    # The single clearest example of a reordering, for the abstract.
    jumps = (point_rank - composite_rank).sort_values(ascending=False)
    mover = jumps.index[0]
    put("MoverName", PRETTY.get(mover, mover))
    put("MoverPointRank", ordinal(point_rank[mover]))
    put("MoverCompositeRank", ordinal(composite_rank[mover]))
    put("MoverPoint", f"{skab.loc[mover, 'point_f1_mean']:.3f}")
    put("MoverComposite", f"{skab.loc[mover, 'composite_f1_mean']:.3f}")

    # How many strategies clear the trivial baseline under each convention.
    baseline = float(skab_rows.flag_everything_f1.iloc[0])
    put("NBeatPoint", int((skab["point_f1_mean"] > baseline).sum()))
    put("NBeatComposite", int((skab["composite_f1_mean"] > baseline).sum()))
    if trivial is not None:
        put("FlagAllRange", f"{trivial['range_f1']:.3f}")
        put("FlagAllPA", f"{trivial['pa_f1']:.3f}")
        put("NBeatRange", int((skab["range_f1_mean"] > trivial["range_f1"]).sum()))
        put("NBeatPA", int((skab["pa_f1_mean"] > trivial["pa_f1"]).sum()))
        # A mean that clears the baseline by less than its own interval is not
        # a clear win, so count the strategies whose whole interval clears it.
        above = skab[skab["range_f1_mean"] > trivial["range_f1"]]
        clear = above[above["range_f1_ci_low"] > trivial["range_f1"]]
        marginal = above.drop(clear.index)
        put("NBeatRangeClear", int(len(clear)))
        put("NBeatRangeMarginal", int(len(marginal)))
        if len(marginal):
            name = marginal.index[0]
            put("RangeMarginalName", PRETTY.get(name, name))
            put("RangeMarginalVal", f"{marginal.iloc[0]['range_f1_mean']:.4f}")
            put("FlagAllRangeFine", f"{trivial['range_f1']:.4f}")

    if segments is not None and not segments.empty:
        put("NRecordings", int(segments.segment.nunique()))

        # How often each strategy fails to beat the trivial baseline on an
        # individual recording. The aggregate winner is not necessarily the
        # most reliable one, and the mean in the headline table cannot show it.
        per_recording = (
            segments.groupby(["strategy", "segment"]).composite_lift.mean().reset_index()
        )
        losses = (
            per_recording.assign(lost=per_recording.composite_lift < 0)
            .groupby("strategy")
            .lost.sum()
        )
        winner = winners["composite_f1"]
        put("BestCompositeLosses", int(losses.get(winner, 0)))
        adaptive = [s for s in losses.index if s.split("-")[-1] in ("periodic", "drift")]
        if adaptive:
            steadiest = losses.loc[adaptive].idxmin()
            put("SteadiestName", PRETTY.get(steadiest, steadiest))
            put("SteadiestLosses", int(losses.loc[steadiest]))

    if significance is not None and not significance.empty:
        block = significance[significance.stream.str.startswith("skab")]
        put("NComparisons", int(len(block)))
        put("NSignificant", int(block.significant_holm.sum()))
        seed_block = block[(block.unit == "seed") & (block.metric == "composite_f1")]
        put("NSeedComparisons", int(len(seed_block)))
        put("NSeedSignificant", int(seed_block.significant_holm.sum()))

    if ucr is not None and not ucr.empty:
        ucr_rows = rows[rows.stream.str.startswith("ucr-")]
        put("NUcrSeries", int(ucr_rows.stream.nunique()))
        put("UcrRows", f"{int(ucr_rows.n_steps.mean()):,}")
        put("UcrBaseRate", f"{100 * ucr_rows.base_rate.mean():.2f}")
        put("UcrFlagAll", f"{ucr_rows.flag_everything_f1.mean():.3f}")

        found = ucr["found"].sort_values(ascending=False)
        put("BestUcrName", PRETTY.get(found.index[0], found.index[0]))
        put("BestUcrFound", f"{found.iloc[0]:.2f}")
        put("WorstUcrName", PRETTY.get(found.index[-1], found.index[-1]))
        put("WorstUcrFound", f"{found.iloc[-1]:.2f}")
        put("UcrRecallSpread", f"{found.iloc[0] - found.iloc[-1]:.2f}")

        best_composite = ucr["composite"].idxmax()
        put("BestUcrComposite", PRETTY.get(best_composite, best_composite))
        policy_words = {"never": "never rebuild", "online": "learn continuously",
                        "periodic": "rebuild on a schedule", "drift": "rebuild on drift"}
        best_policy = rows[rows.strategy == best_composite].policy.iloc[0]
        put("BestUcrPolicy", policy_words.get(best_policy, best_policy))
        found_policy = rows[rows.strategy == found.index[0]].policy.iloc[0]
        put("BestUcrFoundPolicy", policy_words.get(found_policy, found_policy))
        adaptive = ucr.index[ucr.index.str.contains("periodic|drift")]
        put("NUcrAdaptive", int(len(adaptive)))
        put(
            "NUcrAdaptiveAboveBest",
            int((ucr.loc[adaptive, "composite"] > ucr["composite"].max() - 1e-12).sum()),
        )
        policy_words = {"never": "never rebuild", "online": "learn continuously",
                        "periodic": "rebuild on a schedule", "drift": "rebuild on drift"}
        best_policy = rows[rows.strategy == best_composite].policy.iloc[0]
        put("BestUcrPolicy", policy_words.get(best_policy, best_policy))
        found_policy = rows[rows.strategy == found.index[0]].policy.iloc[0]
        put("BestUcrFoundPolicy", policy_words.get(found_policy, found_policy))
        adaptive = ucr.index[ucr.index.str.contains("periodic|drift")]
        put("NUcrAdaptive", int(len(adaptive)))
        put(
            "NUcrAdaptiveAboveBest",
            int((ucr.loc[adaptive, "composite"] > ucr["composite"].max() - 1e-12).sum()),
        )
        put("BestUcrCompositeVal", f"{ucr['composite'].max():.3f}")
        best_point = ucr["point"].idxmax()
        put("BestUcrPoint", PRETTY.get(best_point, best_point))

        if "top1" in ucr.columns:
            top1 = ucr["top1"].sort_values(ascending=False)
            # Several deterministic strategies can score the same fraction of
            # series, so name every strategy at the top rather than one of them.
            tied = [PRETTY.get(n, n) for n in top1.index[top1 >= top1.iloc[0] - 1e-12]]
            put("BestUcrTopOne", "; ".join(tied[:-1]) + "; and " + tied[-1] if len(tied) > 1 else tied[0])
            put("NBestUcrTopOne", len(tied))
            put("BestUcrTopOneVal", f"{top1.iloc[0]:.2f}")
            put("WorstUcrTopOneVal", f"{top1.iloc[-1]:.2f}")
            put("MeanUcrTopOne", f"{ucr['top1'].mean():.2f}")
            # Do the two protocols agree on who is best?
            put(
                "TopOneAgreesWithComposite",
                "yes" if ucr["composite"].idxmax() in top1.index[top1 >= top1.iloc[0] - 1e-12] else "no",
            )
        put("NUcrBeatComposite", int((ucr["composite"] > ucr["baseline"]).sum()))
        put("NUcrBeatPoint", int((ucr["point"] > ucr["baseline"]).sum()))

        # Does the recall axis actually carry information here, unlike the other
        # two datasets? This is the number that answers it.
        for prefix, frame in (("Skab", rows[rows.stream.str.startswith("skab")]),
                              ("Nab", rows[rows.stream.str.startswith(("machine", "cpu", "nyc", "ambient", "rogue", "ec2"))])):
            if not frame.empty:
                put(f"{prefix}EventRecallMin", f"{frame.composite_recall.min():.2f}")

    nab = summary[~summary.stream.str.startswith(("skab", "ucr-"))]
    if not nab.empty:
        baselines = rows.groupby("stream").flag_everything_f1.first()
        best = (
            nab.sort_values("composite_f1_mean", ascending=False)
            .groupby("stream")
            .head(1)
        )
        beats_composite = sum(
            1 for _, r in best.iterrows() if r.composite_f1_mean > baselines[r.stream]
        )
        beats_point = sum(
            1 for _, r in best.iterrows() if r.point_f1_mean > baselines[r.stream]
        )
        put("NNabSeries", int(best.stream.nunique()))
        put("NNabBeatsComposite", int(beats_composite))
        put("NNabBeatsPoint", int(beats_point))
        put("NNabDistinctWinners", int(best.strategy.nunique()))

    rates = speed if speed is not None else skab["rows_per_second_mean"]
    rates = rates.dropna()
    put("FastestRows", f"{rates.max():,.0f}")
    put("SlowestRows", f"{rates.min():,.0f}")
    put("SlowestModel", PRETTY.get(rates.idxmin(), rates.idxmin()))
    put("FastestModel", PRETTY.get(rates.idxmax(), rates.idxmax()))
    put("SpeedSource", "a quiet run" if speed_source == "quiet" else "the grid")
    put("SpeedRatio", f"{rates.max() / rates.min():.0f}")

    body = [
        "% Generated by scripts/make_paper_tables.py. Do not edit by hand.",
        "% Every number quoted in the prose is defined here, straight from",
        "% results/*.csv, so the text cannot drift from the experiments.",
    ]
    for name, value in sorted(values.items()):
        body.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    write("numbers.tex", "\n".join(body) + "\n")
    return values


def digest(summary, rows, significance, segments) -> None:
    """Print the numbers the prose quotes, so the text can be checked."""
    print("\n--- digest for the prose ---")
    skab = summary[summary.stream.str.startswith("skab")].set_index("strategy")
    baseline = float(rows[rows.stream.str.startswith("skab")].flag_everything_f1.iloc[0])
    print(f"flag-everything F1 on valve1: {baseline:.4f}")
    print(f"seeds: {int(skab.n_seeds.iloc[0])}")

    for metric in ("point_f1", "composite_f1", "range_f1", "pa_f1"):
        ordered = skab[f"{metric}_mean"].sort_values(ascending=False)
        print(f"\nranking by {metric}:")
        for name, value in ordered.items():
            print(f"   {name:<18} {value:.4f}")

    if segments is not None and not segments.empty:
        print(f"\nrecordings scored: {segments.segment.nunique()}")

    if significance is not None and not significance.empty:
        sig = significance[
            significance.stream.str.startswith("skab") & significance.significant_holm
        ]
        print(f"\nsignificant after Holm ({len(sig)} of {len(significance[significance.stream.str.startswith('skab')])}):")
        for _, row in sig.iterrows():
            print(
                f"   [{row.metric}/{row.unit}] {row.strategy_a} > {row.strategy_b}: "
                f"{row.mean_diff:+.4f} p={row.p_value:.4g}"
            )


def main() -> int:
    print("Building paper tables:")
    summary = load("multiseed_summary.csv")
    rows = load("multiseed.csv")
    significance = load("significance.csv")
    segments = load("multiseed_segments.csv")
    sweep_summary = load("sweep_multiseed_summary.csv")

    if summary is None or rows is None:
        print("Run scripts/run_multiseed.py first.")
        return 1

    throughput = load("throughput.csv")

    _, _, trivial = table_main(summary, rows)
    ucr = table_ucr(rows)
    speed, speed_source = table_cost(summary, throughput)
    table_transfer(summary, rows)
    if significance is not None:
        table_significance(significance)
    if sweep_summary is not None:
        table_detectors(sweep_summary)
    table_numbers(summary, rows, significance, segments, speed, speed_source,
                  trivial, ucr)

    digest(summary, rows, significance, segments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
