"""Aggregating repeated runs, and testing whether two strategies actually differ.

Everything this project reported up to this point came from one run at one seed.
That is fine for a demo and useless for a claim: a 0.04 gap between two policies
means nothing until you know how much a policy moves when only the seed changes.
This module is what turns a pile of per-seed rows into an interval and a p-value.

Two design choices worth stating.

**Runs are paired by seed.** Seed *s* builds the model, and where the stream
itself is stochastic it builds the stream too, so every strategy at seed *s*
sees the same world. Comparing strategies within a seed and then aggregating the
differences removes the shared stream-to-stream variance, which is much larger
here than the gap between policies. An unpaired test on this data would be
underpowered to the point of being misleading.

**We report a bootstrap interval and a Wilcoxon p-value, not a t-test.** With
10-30 seeds and metrics that are bounded, skewed, and occasionally pinned at 0
or 1, normality is not a safe assumption. The bootstrap makes none, and the
signed-rank test only assumes a symmetric difference distribution. Where the
sample is too small for Wilcoxon to reach significance at any effect size we say
so rather than reporting a p-value that could never have been small.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Resamples for every bootstrap interval. 10,000 is enough that the interval is
# stable to about the third decimal, which is finer than anything we report.
N_BOOTSTRAP = 10_000
DEFAULT_ALPHA = 0.05


@dataclass(frozen=True)
class Interval:
    """A point estimate with a bootstrap confidence interval."""

    mean: float
    low: float
    high: float
    sd: float
    n: int

    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.low:.3f}, {self.high:.3f}]"

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        return {
            f"{prefix}mean": self.mean,
            f"{prefix}ci_low": self.low,
            f"{prefix}ci_high": self.high,
            f"{prefix}sd": self.sd,
            f"{prefix}n": self.n,
        }


@dataclass(frozen=True)
class Comparison:
    """The paired difference between two strategies on one metric."""

    name_a: str
    name_b: str
    metric: str
    diff: Interval
    p_value: float
    n_pairs: int
    # True when the sample is too small for the signed-rank test to produce a
    # significant result at any effect size, which is a property of n, not of
    # the data. Reporting p without this is how underpowered tests get read as
    # evidence of no difference.
    underpowered: bool

    @property
    def significant(self) -> bool:
        return (not self.underpowered) and self.p_value < DEFAULT_ALPHA

    def as_dict(self) -> dict:
        return {
            "strategy_a": self.name_a,
            "strategy_b": self.name_b,
            "metric": self.metric,
            "mean_diff": self.diff.mean,
            "ci_low": self.diff.low,
            "ci_high": self.diff.high,
            "p_value": self.p_value,
            "n_pairs": self.n_pairs,
            "underpowered": self.underpowered,
            "significant": self.significant,
        }


def bootstrap_ci(
    values,
    alpha: float = DEFAULT_ALPHA,
    n_resamples: int = N_BOOTSTRAP,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap interval for the mean of `values`."""
    a = np.asarray(values, dtype=float)
    a = a[~np.isnan(a)]
    n = a.size
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"), float("nan"), 0)
    if n == 1:
        v = float(a[0])
        return Interval(v, v, v, 0.0, 1)

    rng = np.random.default_rng(seed)
    draws = rng.choice(a, size=(n_resamples, n), replace=True).mean(axis=1)
    low, high = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(
        mean=float(a.mean()),
        low=float(low),
        high=float(high),
        sd=float(a.std(ddof=1)) if n > 1 else 0.0,
        n=int(n),
    )


def _min_n_for_wilcoxon(alpha: float = DEFAULT_ALPHA) -> int:
    """Smallest paired sample where the signed-rank test can reach `alpha`.

    The two-sided exact test's smallest attainable p-value is 2 / 2**n, so six
    pairs is the first n where that drops below 0.05. Below it, a
    non-significant result carries no information about the effect.
    """
    n = 1
    while 2.0 / (2**n) > alpha:
        n += 1
    return n


MIN_PAIRS_FOR_WILCOXON = _min_n_for_wilcoxon()


def paired_compare(
    frame: pd.DataFrame,
    metric: str,
    name_a: str,
    name_b: str,
    by: str = "strategy",
    pair_on: str = "seed",
    seed: int = 0,
) -> Comparison:
    """Compare two strategies on `metric`, pairing rows that share `pair_on`.

    The returned difference is a-minus-b, so a positive mean means `name_a`
    scored higher.
    """
    for column in (by, pair_on, metric):
        if column not in frame.columns:
            raise KeyError(f"frame has no column {column!r}")

    a = frame[frame[by] == name_a].set_index(pair_on)[metric]
    b = frame[frame[by] == name_b].set_index(pair_on)[metric]

    # Duplicate labels would align as a cross product and silently manufacture
    # pairs that do not exist. That is exactly the shape of a per-recording
    # frame that still has one row per seed, so refuse it here and make the
    # caller aggregate first.
    for name, series in ((name_a, a), (name_b, b)):
        if series.index.has_duplicates:
            offenders = series.index[series.index.duplicated()].unique().tolist()
            raise ValueError(
                f"{name} has more than one row per {pair_on} "
                f"(e.g. {offenders[:3]}); aggregate before pairing"
            )

    shared = a.index.intersection(b.index)
    if len(shared) == 0:
        raise ValueError(f"no shared {pair_on} values between {name_a} and {name_b}")

    diffs = (a.loc[shared] - b.loc[shared]).to_numpy(dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    n_pairs = int(diffs.size)

    interval = bootstrap_ci(diffs, seed=seed)

    # An all-zero difference vector means the two strategies were identical on
    # every seed. Wilcoxon cannot rank that and raises; the honest p is 1.0.
    if n_pairs == 0 or np.allclose(diffs, 0.0):
        p_value = 1.0
    else:
        p_value = float(scipy_stats.wilcoxon(diffs).pvalue)

    return Comparison(
        name_a=name_a,
        name_b=name_b,
        metric=metric,
        diff=interval,
        p_value=p_value,
        n_pairs=n_pairs,
        underpowered=n_pairs < MIN_PAIRS_FOR_WILCOXON,
    )


def summarise_seeds(
    frame: pd.DataFrame,
    metrics: list[str],
    group: list[str],
    seed: int = 0,
) -> pd.DataFrame:
    """Collapse per-seed rows into one row per group, with intervals.

    `group` is whatever identifies a condition — typically ["stream",
    "strategy"]. Every column in `metrics` gains mean / ci_low / ci_high / sd
    columns, and the seed count is carried so a reader can see what n each
    interval rests on.
    """
    missing = [m for m in metrics if m not in frame.columns]
    if missing:
        raise KeyError(f"frame has no columns {missing}")

    rows = []
    for key, block in frame.groupby(group, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group, key))
        row["n_seeds"] = int(len(block))
        for metric in metrics:
            interval = bootstrap_ci(block[metric], seed=seed)
            row.update(interval.as_dict(prefix=f"{metric}_"))
            # The bare metric name keeps holding the mean, so existing readers
            # and plotting code that expect `f1` still work unchanged.
            row[metric] = interval.mean
        rows.append(row)
    return pd.DataFrame(rows)


def compare_all_pairs(
    frame: pd.DataFrame,
    metric: str,
    by: str = "strategy",
    pair_on: str = "seed",
    within: list[str] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Every pairwise comparison on `metric`, optionally within each stream.

    No multiplicity correction is applied here. The paper applies Holm-Bonferroni
    over the family of comparisons it actually reports, which is the smaller and
    more defensible family; correcting inside this function would double-count.
    """
    blocks = [((), frame)] if not within else list(frame.groupby(within, dropna=False))

    rows = []
    for key, block in blocks:
        key = key if isinstance(key, tuple) else (key,)
        names = sorted(block[by].dropna().unique())
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                comparison = paired_compare(
                    block, metric, a, b, by=by, pair_on=pair_on, seed=seed
                )
                row = dict(zip(within or [], key))
                row.update(comparison.as_dict())
                rows.append(row)
    return pd.DataFrame(rows)


def holm_bonferroni(p_values, alpha: float = DEFAULT_ALPHA) -> np.ndarray:
    """Holm-Bonferroni step-down: returns the reject/keep decision per test.

    Used instead of plain Bonferroni because it is uniformly more powerful at
    the same family-wise error rate, and instead of Benjamini-Hochberg because
    with a family this small we would rather control the chance of any false
    claim than the expected proportion of them.
    """
    p = np.asarray(p_values, dtype=float)
    n = p.size
    if n == 0:
        return np.zeros(0, dtype=bool)

    order = np.argsort(p)
    reject_sorted = np.zeros(n, dtype=bool)
    for rank, index in enumerate(order):
        if p[index] <= alpha / (n - rank):
            reject_sorted[rank] = True
        else:
            break

    reject = np.zeros(n, dtype=bool)
    reject[order[: int(reject_sorted.sum())]] = True
    return reject
