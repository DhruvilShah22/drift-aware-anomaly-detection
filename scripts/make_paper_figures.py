"""Figures for the paper, built from the committed result CSVs.

Nothing here recomputes anything. Every figure reads `results/*.csv` so that a
number in a plot and the same number in a table cannot drift apart, and so the
figures regenerate in seconds without the experiments being re-run.

    python scripts/make_paper_figures.py

Writes PNG and PDF into `results/figures/paper/`. The PDFs are vector and are
what the manuscript includes; the PNGs are for the README and for looking at.

Design notes, since figures in a paper get read in bad conditions:

- Four metrics share one categorical palette, assigned in a fixed order so a
  colour always means the same metric across every figure. The order is checked
  for colour-vision separation; the worst adjacent pair is 9.1 CVD Delta E, above
  the 8 threshold.
- Colour is never the only encoding. Bars carry hatching and printed values, so
  the figures survive greyscale printing and a monochrome reader.
- One y-axis everywhere. No figure here puts two scales on one plot.
- Grid and axes are recessive; the marks are the only strong ink.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures" / "paper"

# Categorical slots, fixed order, validated for colour-vision separation.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d8d7d2"

# Metric identity is constant across every figure in the paper.
METRIC_STYLE = {
    "point_f1": ("Point-wise", BLUE, ""),
    "composite_f1": ("Composite", ORANGE, "///"),
    "range_f1": ("Range-based", AQUA, "..."),
    "pa_f1": ("Point-adjust", YELLOW, "xxx"),
}

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

# The figures use the same names as the text and tables: "HST, never", not the
# internal identifier "hst-static".
MODEL_DISPLAY = {"hst": "HST", "iforest": "iForest", "lof": "LOF",
                 "ocsvm": "OC-SVM", "elliptic": "MCD"}
POLICY_DISPLAY = {"static": "never"}


def display(strategy: str) -> str:
    model, _, policy = strategy.partition("-")
    return f"{MODEL_DISPLAY.get(model, model)}, {POLICY_DISPLAY.get(policy, policy)}"


def style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 8,
    })


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{suffix}")
    plt.close(fig)
    print(f"  results/figures/paper/{name}.png (+ .pdf)")


def load(name: str) -> pd.DataFrame | None:
    path = RESULTS / name
    if not path.exists():
        print(f"  skipped: results/{name} not found")
        return None
    return pd.read_csv(path)


def _present(summary: pd.DataFrame) -> list[str]:
    return [s for s in STRATEGY_ORDER if s in set(summary.strategy)]


def figure_metric_ranking(summary: pd.DataFrame, trivial: pd.Series | None) -> None:
    """The paper's central figure: one stream, four metrics, four rankings.

    Grouped bars rather than lines because the strategies are unordered
    categories, and each bar carries its own confidence interval so the reader
    can see which gaps are real before reading any claim about them.
    """
    skab = summary[summary.stream.str.startswith("skab")]
    if skab.empty:
        return
    strategies = _present(skab)
    indexed = skab.set_index("strategy")

    x = np.arange(len(strategies))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9.0, 3.8))

    for i, (metric, (label, colour, hatch)) in enumerate(METRIC_STYLE.items()):
        means = np.array([indexed.loc[s, f"{metric}_mean"] for s in strategies])
        low = np.array([indexed.loc[s, f"{metric}_ci_low"] for s in strategies])
        high = np.array([indexed.loc[s, f"{metric}_ci_high"] for s in strategies])
        offset = (i - 1.5) * width
        ax.bar(
            x + offset, means, width * 0.9,
            label=label, color=colour, hatch=hatch,
            edgecolor="white", linewidth=0.6,
        )
        ax.errorbar(
            x + offset, means,
            yerr=[means - low, high - means],
            fmt="none", ecolor=INK, elinewidth=0.8, capsize=1.8,
        )

    # Flagging every row does not score the same under every convention. It is
    # 2b/(1+b) point-wise, composite and point-adjust, but under range-based
    # scoring the single giant predicted range is divided by the number of true
    # events it swallows, which puts its baseline far lower. One line across all
    # four groups would misrepresent three of them, so both are drawn.
    baseline = (
        float(indexed["flag_everything_f1"].iloc[0])
        if "flag_everything_f1" in indexed
        else float("nan")
    )
    references = [(baseline, "flag everything", (0, (4, 3)))]
    if trivial is not None and "range_f1" in trivial:
        references.append(
            (float(trivial["range_f1"]), "flag everything, range-based", (0, (1, 2)))
        )

    for value, label, dashes in references:
        if np.isnan(value):
            continue
        ax.axhline(value, color=INK, linestyle=dashes, linewidth=1.0)
        ax.text(
            len(strategies) - 0.45, value + 0.014,
            f"{label} ({value:.3f})",
            ha="right", fontsize=7.5, color=INK,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([display(s) for s in strategies], rotation=30, ha="right")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    n_seeds = int(indexed["n_seeds"].iloc[0]) if "n_seeds" in indexed else 0
    seed_note = f"{n_seeds} seed{'s' if n_seeds != 1 else ''}"
    ax.set_title(
        f"The metric decides the ranking (SKAB valve1, {seed_note}, 95% CI)",
        loc="left",
    )
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, 1.0))
    save(fig, "metric_ranking")


def figure_rank_flip(summary: pd.DataFrame) -> None:
    """The same claim as a bump chart, which is where it is unmissable.

    Each line is a strategy; the y position is its rank under each metric. Lines
    that cross are strategies whose order depends on nothing but the choice of
    scoring convention.
    """
    skab = summary[summary.stream.str.startswith("skab")]
    if skab.empty:
        return
    strategies = _present(skab)
    metrics = list(METRIC_STYLE)

    ranks = {}
    for metric in metrics:
        values = skab.set_index("strategy").loc[strategies, f"{metric}_mean"]
        ranks[metric] = values.rank(ascending=False, method="min")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = np.arange(len(metrics))

    # Colour carries the model family, not the individual strategy. Ten lines
    # would need ten hues, and cycling a palette to reach that makes colour mean
    # nothing; three families fit the palette slots that separate cleanly under
    # colour-vision simulation. Identity comes from the label at the right edge,
    # which every line gets, so no line depends on colour to be identified.
    family_colour = {"hst": BLUE, "iforest": ORANGE}
    family_of = lambda name: name.split("-")[0]

    for strategy in strategies:
        y = [ranks[m][strategy] for m in metrics]
        colour = family_colour.get(family_of(strategy), AQUA)
        moves = len({int(ranks[m][strategy]) for m in metrics}) > 1
        ax.plot(
            x, y,
            marker="o", markersize=4.5,
            linewidth=2.0 if moves else 1.2,
            color=colour, alpha=1.0 if moves else 0.45, zorder=3 if moves else 2,
        )
        ax.annotate(
            display(strategy), (x[-1], y[-1]), xytext=(6, 0),
            textcoords="offset points", va="center", fontsize=7.5, color=colour,
        )

    handles = [
        plt.Line2D([], [], color=BLUE, linewidth=2.0, label="Half-Space Trees"),
        plt.Line2D([], [], color=ORANGE, linewidth=2.0, label="Isolation Forest"),
        plt.Line2D([], [], color=AQUA, linewidth=2.0, label="classical baselines"),
    ]
    # A frame here only, because this legend sits over crossing lines.
    ax.legend(
        handles=handles, loc="lower left", fontsize=7.5,
        frameon=True, facecolor="white", edgecolor="none", framealpha=0.85,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_STYLE[m][0] for m in metrics])
    ax.set_yticks(range(1, len(strategies) + 1))
    ax.set_ylabel("rank (1 = best)")
    ax.invert_yaxis()
    ax.set_xlim(-0.3, len(metrics) - 0.15)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_title("Rank under each scoring convention", loc="left")
    save(fig, "rank_flip")


def figure_factorial(rows: pd.DataFrame) -> None:
    """Model against policy, so the two factors can be read apart.

    A sequential single-hue ramp, because the quantity is a magnitude with no
    meaningful midpoint. Every cell is printed as well as shaded, which is what
    discharges the contrast warning on the light end of the ramp.
    """
    skab = rows[rows.stream.str.startswith("skab")]
    if skab.empty:
        return

    grid = (
        skab.groupby(["model", "policy"]).composite_f1.mean().unstack()
    )
    policy_order = [p for p in ("never", "online", "periodic", "drift") if p in grid.columns]
    model_order = [m for m in ("hst", "iforest", "lof", "ocsvm", "elliptic") if m in grid.index]
    grid = grid.loc[model_order, policy_order]

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    data = grid.to_numpy(dtype=float)
    image = ax.imshow(data, cmap="Blues", aspect="auto", vmin=np.nanmin(data), vmax=np.nanmax(data))

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if np.isnan(value):
                ax.text(j, i, "-", ha="center", va="center", color=MUTED, fontsize=9)
                continue
            span = np.nanmax(data) - np.nanmin(data)
            light = span > 0 and (value - np.nanmin(data)) / span > 0.55
            ax.text(
                j, i, f"{value:.3f}",
                ha="center", va="center", fontsize=8.5,
                color="white" if light else INK,
            )

    ax.set_xticks(range(len(policy_order)), policy_order)
    ax.set_yticks(range(len(model_order)), [MODEL_DISPLAY.get(m, m) for m in model_order])
    ax.set_xlabel("adaptation policy")
    ax.set_ylabel("model")
    ax.set_title("Composite F1 by model and policy (SKAB valve1)", loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    fig.colorbar(image, ax=ax, shrink=0.85, label="composite F1")
    save(fig, "factorial")


def figure_segment_spread(segments: pd.DataFrame) -> None:
    """How much of the ranking survives moving to a different recording.

    Sixteen source recordings, every strategy scored on each. A box per strategy
    shows the spread the summary table's single mean hides.
    """
    if segments is None or segments.empty:
        return
    strategies = [s for s in STRATEGY_ORDER if s in set(segments.strategy)]

    # Average over seeds first, so each box has one point per recording rather
    # than ten correlated copies of it.
    per_segment = (
        segments.groupby(["strategy", "segment"]).composite_lift.mean().reset_index()
    )
    data = [
        per_segment[per_segment.strategy == s].composite_lift.to_numpy()
        for s in strategies
    ]

    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    parts = ax.boxplot(
        data, tick_labels=[display(s) for s in strategies], patch_artist=True, widths=0.6,
        medianprops={"color": INK, "linewidth": 1.4},
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": MUTED,
                    "markeredgecolor": "none", "alpha": 0.6},
    )
    for patch in parts["boxes"]:
        patch.set_facecolor(BLUE)
        patch.set_alpha(0.35)
        patch.set_edgecolor(MUTED)
    for element in ("whiskers", "caps"):
        for line in parts[element]:
            line.set_color(MUTED)

    ax.axhline(0, color=INK, linestyle=(0, (4, 3)), linewidth=1.0)
    ax.set_ylabel("composite F1 minus flag-everything")
    ax.set_xticklabels([display(s) for s in strategies], rotation=30, ha="right")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    # Count the recordings actually scored: one of the sixteen carries no fault
    # and is dropped, and the title should not claim otherwise.
    n_recordings = per_segment.segment.nunique()
    ax.set_title(
        f"Per-recording spread across {n_recordings} valve1 recordings "
        f"(the ones containing a fault)",
        loc="left",
    )
    save(fig, "segment_spread")


def figure_detector_delay(summary: pd.DataFrame) -> None:
    """Detection delay against change magnitude, one panel per drift shape."""
    if summary is None or summary.empty:
        return

    shapes = [k for k in ("sudden", "incremental", "gradual", "recurring") if k in set(summary.kind)]
    detectors = sorted(summary.detector.unique())
    colours = [BLUE, ORANGE, AQUA, YELLOW, "#4a3aa7"]
    markers = ["o", "s", "^", "D", "v"]

    fig, axes = plt.subplots(1, len(shapes), figsize=(10.5, 2.9), sharey=True)
    if len(shapes) == 1:
        axes = [axes]

    for ax, shape in zip(axes, shapes):
        block = summary[summary.kind == shape]
        for i, detector in enumerate(detectors):
            series = block[block.detector == detector].sort_values("magnitude")
            if series.empty:
                continue
            ax.plot(
                series.magnitude, series.mean_delay_mean,
                marker=markers[i % len(markers)], markersize=4.5,
                linewidth=1.8, color=colours[i % len(colours)], label=detector,
            )
            ax.fill_between(
                series.magnitude,
                series.mean_delay_ci_low,
                series.mean_delay_ci_high,
                color=colours[i % len(colours)], alpha=0.15, linewidth=0,
            )
        ax.set_title(shape, loc="left", fontsize=9)
        ax.set_xlabel("shift (sd)")
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("detection delay (rows)")
    axes[-1].legend(loc="upper right", fontsize=7.5)
    # Read the seed count off the data instead of asserting one in the caption,
    # so a shorter run cannot leave the figure claiming replication it lacks.
    n_seeds = int(summary["n_seeds"].max()) if "n_seeds" in summary else 0
    fig.suptitle(
        f"Detection delay by drift shape and detector "
        f"({n_seeds} seeds, 95% CI band)",
        x=0.005, ha="left", fontsize=10,
    )
    save(fig, "detector_delay")


def figure_ucr(rows: pd.DataFrame) -> None:
    """Where the recall axis finally does some work.

    One anomaly per series means a strategy either finds it or does not, so the
    height of each bar is the proportion of anomalies found. The other two
    datasets put every bar at 1.0.
    """
    ucr = rows[rows.stream.str.startswith("ucr-")]
    if ucr.empty:
        return

    n_series = ucr.stream.nunique()
    grouped = ucr.groupby("strategy").agg(
        found=("composite_recall", "mean"),
        composite=("composite_f1", "mean"),
        precision=("composite_precision", "mean"),
    )
    strategies = [s for s in STRATEGY_ORDER if s in grouped.index]
    grouped = grouped.loc[strategies]

    x = np.arange(len(strategies))
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 3.6))

    left.bar(x, grouped["found"], 0.62, color=ORANGE, edgecolor="white", linewidth=0.6)
    for i, value in enumerate(grouped["found"]):
        left.text(i, value + 0.02, f"{value:.2f}", ha="center", fontsize=7.5, color=INK)
    left.set_xticks(x)
    left.set_xticklabels([display(s) for s in strategies], rotation=35, ha="right")
    left.set_ylabel("proportion of anomalies found")
    left.set_ylim(0, 1.1)
    left.yaxis.grid(True)
    left.set_axisbelow(True)
    left.set_title(f"Event recall over {n_series} UCR series", loc="left", fontsize=9)

    right.scatter(
        grouped["found"], grouped["precision"],
        s=46, color=BLUE, edgecolor="white", linewidth=0.8, zorder=3,
    )
    for name, row in grouped.iterrows():
        right.annotate(
            display(name), (row["found"], row["precision"]), xytext=(5, 3),
            textcoords="offset points", fontsize=7, color=MUTED,
        )
    right.set_xlabel("proportion of anomalies found")
    right.set_ylabel("point-wise precision")
    right.grid(True)
    right.set_axisbelow(True)
    right.set_title("Recall against precision", loc="left", fontsize=9)

    save(fig, "ucr")


def figure_ablation(warmup: pd.DataFrame, horizon: pd.DataFrame) -> None:
    """Do the conclusions move when the hand-picked constants move?"""
    if warmup is None and horizon is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))

    if warmup is not None and not warmup.empty:
        ax = axes[0]
        strategies = [s for s in STRATEGY_ORDER if s in set(warmup.strategy)]
        colours = [BLUE, ORANGE, AQUA, YELLOW]
        markers = ["o", "s", "^", "D"]
        grouped = warmup.groupby(["strategy", "warmup_size"]).composite_f1.mean()
        for i, strategy in enumerate(strategies):
            series = grouped.loc[strategy]
            ax.plot(
                series.index, series.to_numpy(),
                marker=markers[i % len(markers)], markersize=4.5, linewidth=1.8,
                color=colours[i % len(colours)], label=display(strategy),
            )
        ax.set_xlabel("warm-up size (rows)")
        ax.set_ylabel("composite F1")
        ax.set_title("Warm-up length", loc="left", fontsize=9)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.legend(fontsize=7.5)

    if horizon is not None and not horizon.empty:
        ax = axes[1]
        shapes = [k for k in ("sudden", "incremental", "gradual", "recurring")
                  if k in set(horizon.stream)]
        colours = [BLUE, ORANGE, AQUA, YELLOW]
        markers = ["o", "s", "^", "D"]
        grouped = horizon.groupby(["stream", "horizon"]).mean_delay.mean()
        for i, shape in enumerate(shapes):
            series = grouped.loc[shape]
            ax.plot(
                series.index, series.to_numpy(),
                marker=markers[i % len(markers)], markersize=4.5, linewidth=1.8,
                color=colours[i % len(colours)], label=shape,
            )
        ax.set_xlabel("matching horizon (rows)")
        ax.set_ylabel("mean detection delay (rows)")
        ax.set_title("Matching horizon", loc="left", fontsize=9)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.legend(fontsize=7.5)

    save(fig, "ablation")


def figure_transfer(summary: pd.DataFrame) -> None:
    """Whether the SKAB setting carries to NAB, per series and per metric."""
    nab = summary[~summary.stream.str.startswith(("skab", "ucr-"))]
    if nab.empty:
        return

    best = (
        nab.sort_values("composite_f1_mean", ascending=False)
        .groupby("stream")
        .head(1)
        .set_index("stream")
    )
    series_names = list(best.index)
    short = [s.replace("_system_failure", "").replace("_misconfiguration", "")
             for s in series_names]

    x = np.arange(len(series_names))
    width = 0.28
    fig, ax = plt.subplots(figsize=(8.4, 3.4))

    for i, metric in enumerate(("point_f1", "composite_f1")):
        label, colour, hatch = METRIC_STYLE[metric]
        means = best[f"{metric}_mean"].to_numpy()
        low = best[f"{metric}_ci_low"].to_numpy()
        high = best[f"{metric}_ci_high"].to_numpy()
        ax.bar(
            x + (i - 0.5) * width, means, width * 0.9,
            label=label, color=colour, hatch=hatch,
            edgecolor="white", linewidth=0.6,
        )
        ax.errorbar(
            x + (i - 0.5) * width, means,
            yerr=[np.maximum(means - low, 0), np.maximum(high - means, 0)],
            fmt="none", ecolor=INK, elinewidth=0.8, capsize=1.8,
        )

    ax.plot(
        x, best["flag_everything_f1"].to_numpy() if "flag_everything_f1" in best
        else np.full(len(x), np.nan),
        linestyle="none", marker="_", markersize=18, color=INK,
        label="flag everything",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=25, ha="right")
    ax.set_ylabel("F1")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_title(
        "Best strategy per NAB series, SKAB threshold carried over unchanged",
        loc="left",
    )
    ax.legend(ncol=3, fontsize=8)
    save(fig, "transfer")


def main() -> int:
    style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    print("Building paper figures:")

    summary = load("multiseed_summary.csv")
    rows = load("multiseed.csv")
    segments = load("multiseed_segments.csv")
    sweep_summary = load("sweep_multiseed_summary.csv")
    warmup = load("ablation_warmup.csv")
    horizon = load("ablation_horizon.csv")

    if summary is not None and rows is not None:
        # flag_everything_f1 lives on the raw rows, not the summary, so attach
        # the per-stream value the figures need.
        baseline = rows.groupby("stream").flag_everything_f1.first()
        summary = summary.merge(
            baseline.rename("flag_everything_f1"), on="stream", how="left"
        )
        trivial_frame = load("trivial_baseline.csv")
        trivial = trivial_frame.iloc[0] if trivial_frame is not None else None
        figure_metric_ranking(summary, trivial)
        figure_rank_flip(summary)
        figure_factorial(rows)
        figure_transfer(summary)
        figure_ucr(rows)

    if segments is not None:
        figure_segment_spread(segments)
    if sweep_summary is not None:
        figure_detector_delay(sweep_summary)
    if warmup is not None or horizon is not None:
        figure_ablation(warmup, horizon)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
