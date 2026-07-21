"""Deriving the alarm threshold from the warm-up score distribution.

Until now the threshold was a hand-set quantile of warm-up scores, and it had to
be set *per dataset*: 0.98 where anomalies are rare, 0.50 on SKAB's valve1
stream. The transfer test showed what that costs — applying SKAB's choice to NAB
loses 0.093 F1 on average and beats a trivial baseline on only half its series.

## Why one quantile cannot serve both

The two datasets differ in something the quantile silently depends on: **how
contaminated the warm-up window is**.

- NAB's warm-up windows are 0% anomalous. The warm-up scores describe normal
  behaviour, so the 0.98 quantile means "the top 2% of normal", which is what a
  selective alarm should be.
- SKAB's valve1 warm-up is ~35% anomalous. Its upper tail is *made of anomalies*,
  so the 0.98 quantile sits far above anything a detector should require, and
  recall collapses. Dropping to 0.50 was a workaround for contamination, not a
  principled operating point.

A fixed quantile therefore measures different things on different data. The
rules here are attempts to define the threshold in a way that does not.

## The rules

- `fixed_quantile(q)` — the existing behaviour, kept as the baseline to beat.
- `robust_z(k)` — median plus `k` robust standard deviations, scale estimated by
  MAD. The median and MAD have a 50% breakdown point, so up to half the warm-up
  can be anomalous before the estimate moves. This is the rule designed for the
  contamination problem above.
- `tukey(k)` — the classic Q3 + k x IQR outlier fence. Robust to about 25%
  contamination, which is less than valve1 has, included to test whether that
  matters in practice.
- `target_rate(rate)` — flag the top `rate` fraction outright. Not robust at all,
  but it makes the deployment parameter explicit and honest: "I expect this
  fraction to be anomalous."

## What the study actually found

`scripts/run_threshold_study.py` applied each rule unchanged to all seven
labelled streams (SKAB valve1 plus six NAB series). Three results, none of them
what I expected when writing this module:

1. **`robust_z` lost.** The contamination argument above is real — the unit
   tests show a fixed quantile shifts ~81% under 35% contamination while
   `robust_z` shifts ~34% — but it did not translate into better F1. Averaged
   over streams, `robust_z@3` scored *below* the hand-tuned baseline.
   Contamination robustness was simply not the binding constraint. Only one of
   seven streams is contaminated at all; on the other six, robustness buys
   nothing and costs calibration accuracy.
2. **`target_rate` is not a new rule.** `target_rate(r)` is exactly
   `fixed_quantile(1 - r)` and produces bit-identical results. It survives
   because the parameter is meaningful to whoever sets it, not because it adds
   capability.
3. **A single uniform quantile beat per-dataset hand tuning**, which is the one
   genuine win. `quantile@0.9` improved *every* strategy over the hand-tuned
   0.98-for-NAB / 0.50-for-SKAB split, by +0.005 to +0.049 mean lift.

The honest limit: with the strategy held fixed, every rule still has negative
mean lift over flag-everything for all three HST-based strategies. A better
threshold narrows the gap; it does not close it. See PROGRESS.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# Scale factor making MAD a consistent estimator of the standard deviation for
# normally distributed data.
MAD_TO_SIGMA = 1.4826

# The best transferable setting found by scripts/run_threshold_study.py: one
# value, applied to both datasets, no per-dataset tuning. It improved every
# strategy over the hand-tuned per-dataset quantiles. Expressed as a rate
# because that is the interpretable form; it equals quantile@0.9.
RECOMMENDED_RULE = "target_rate@0.10"


@dataclass(frozen=True)
class ThresholdRule:
    """A named way of turning warm-up scores into an alarm threshold."""

    name: str
    fn: Callable[[np.ndarray], float]

    def __call__(self, scores: np.ndarray) -> float:
        scores = np.asarray(scores, dtype=float)
        scores = scores[np.isfinite(scores)]
        if scores.size == 0:
            raise ValueError(f"{self.name}: no finite warm-up scores to work from")
        value = float(self.fn(scores))
        if not np.isfinite(value):
            raise ValueError(f"{self.name}: produced a non-finite threshold")
        return value


def fixed_quantile(q: float) -> ThresholdRule:
    """Threshold at a fixed quantile of warm-up scores.

    The original approach, kept so every other rule has something to beat.
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {q}")
    return ThresholdRule(f"quantile@{q:g}", lambda s: np.quantile(s, q))


def robust_z(k: float = 3.0) -> ThresholdRule:
    """Median plus `k` robust standard deviations, scale from MAD.

    Chosen for contaminated warm-up windows: both the median and the MAD have a
    50% breakdown point, so the estimate holds until more than half the warm-up
    is anomalous. A fixed quantile has no such guarantee — its whole upper tail
    is exactly what contamination corrupts.

    If the MAD is zero (a warm-up window where over half the scores are
    identical, which HST can produce), it falls back to the standard deviation,
    and to a small positive epsilon if that is degenerate too, so the rule always
    returns a threshold strictly above the bulk of the data.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    def compute(scores: np.ndarray) -> float:
        centre = float(np.median(scores))
        mad = float(np.median(np.abs(scores - centre)))
        scale = mad * MAD_TO_SIGMA
        if scale <= 0:
            scale = float(np.std(scores))
        if scale <= 0:
            # Degenerate warm-up: every score identical. Anything strictly
            # greater is an anomaly, so nudge just above the constant value.
            scale = max(abs(centre), 1.0) * 1e-9
        return centre + k * scale

    return ThresholdRule(f"robust_z@{k:g}", compute)


def tukey(k: float = 1.5) -> ThresholdRule:
    """Tukey's upper fence: Q3 + k x IQR.

    Robust to roughly 25% contamination — less than valve1's 35%. Included to
    test whether that breakdown point difference shows up in practice.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    def compute(scores: np.ndarray) -> float:
        q1, q3 = np.quantile(scores, [0.25, 0.75])
        iqr = q3 - q1
        if iqr <= 0:
            iqr = float(np.std(scores)) or max(abs(float(q3)), 1.0) * 1e-9
        return float(q3 + k * iqr)

    return ThresholdRule(f"tukey@{k:g}", compute)


def target_rate(rate: float) -> ThresholdRule:
    """Flag the top `rate` fraction of warm-up scores.

    Equivalent to `fixed_quantile(1 - rate)` but parameterised the way a
    deployment would actually think about it. Not robust to contamination; its
    value is that the parameter means something concrete to whoever sets it.
    """
    if not 0.0 < rate < 1.0:
        raise ValueError(f"rate must be in (0, 1), got {rate}")
    return ThresholdRule(f"target_rate@{rate:g}", lambda s: np.quantile(s, 1.0 - rate))


def build_rule(spec: str | float | ThresholdRule) -> ThresholdRule:
    """Coerce a rule, a bare quantile, or a `name@param` string into a rule.

    Accepting a bare float keeps every existing caller that passed a quantile
    working unchanged.
    """
    if isinstance(spec, ThresholdRule):
        return spec
    if isinstance(spec, (int, float)):
        return fixed_quantile(float(spec))

    name, _, raw = str(spec).partition("@")
    factories = {
        "quantile": fixed_quantile,
        "robust_z": robust_z,
        "tukey": tukey,
        "target_rate": target_rate,
    }
    if name not in factories:
        raise ValueError(
            f"unknown threshold rule {name!r}; expected one of {sorted(factories)}"
        )
    if not raw:
        raise ValueError(f"rule {name!r} needs a parameter, e.g. '{name}@3'")
    return factories[name](float(raw))
