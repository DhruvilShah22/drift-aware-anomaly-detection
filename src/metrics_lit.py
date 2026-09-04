"""Published time-series anomaly metrics, so our numbers can be read against
the literature rather than only against each other.

`evaluate.py` reports two views: point-wise precision/recall/F1, and an
event-level view that pairs event recall with point-wise precision. The second
one was arrived at independently here, but it is not new — it is the
**composite F-score** of Garg et al. (2021), and `composite_f` below is kept as
a named alias so the correspondence is explicit rather than implied.

Three further metrics are implemented, because the choice between them changes
the ranking of the strategies, and a paper that reports only one is hiding that:

- `range_metrics` — the range-based precision/recall of Tatbul et al. (2018).
  Rewards detecting a range at all (existence), how much of it is covered
  (overlap), and penalises fragmenting one true range across many predicted
  ones (cardinality). The most defensible of the three, and the slowest.
- `point_adjust_metrics` — the point-adjust convention of Xu et al. (2018):
  hit one point in a true segment and the whole segment counts as detected.
  Included because much of the deep-learning literature reports it, and because
  it is known to inflate scores badly. We report it to show what it does to
  *our* ranking, not because we trust it.
- `point_adjust_k_metrics` — PA%K (Kim et al., 2022), which adjusts a segment
  only when at least K of it was flagged. This is the intended repair for
  point-adjust's inflation, and sweeping K is the honest way to present it.

References, all cited in the paper:

  Tatbul, Lee, Zdonik, Alam, Gottschlich. "Precision and Recall for Time
  Series." NeurIPS 2018.
  Xu et al. "Unsupervised Anomaly Detection via Variational Auto-Encoder for
  Seasonal KPIs in Web Applications." WWW 2018.
  Garg, Zhang, Samaran, Savitha, Foo. "An Evaluation of Anomaly Detection and
  Diagnosis in Multivariate Time Series." IEEE TNNLS 2021.
  Kim, Choi, Choi, Lee, Yoon. "Towards a Rigorous Evaluation of Time-Series
  Anomaly Detection." AAAI 2022.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.evaluate import to_events


@dataclass(frozen=True)
class ScoreTriple:
    """Whatever precision/recall/F1 a given convention produces."""

    precision: float
    recall: float
    f1: float

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        return {f"{prefix}{k}": v for k, v in asdict(self).items()}


def _f1(precision: float, recall: float) -> float:
    total = precision + recall
    return 2 * precision * recall / total if total > 0 else 0.0


def _as_int_array(x) -> np.ndarray:
    return np.asarray(x).astype(int)


def _check_pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y_true = _as_int_array(y_true)
    y_pred = _as_int_array(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return y_true, y_pred


# --------------------------------------------------------------------------
# Tatbul et al. (2018): range-based precision and recall
# --------------------------------------------------------------------------

def _overlap_size(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Number of positions shared by two half-open ranges."""
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def _cardinality_factor(
    target: tuple[int, int],
    others: list[tuple[int, int]],
    gamma: str,
) -> float:
    """Penalty for one range being split across several ranges on the other side.

    `gamma="one"` applies no penalty; `gamma="reciprocal"` divides by the number
    of distinct overlapping ranges, which is the paper's choice for the case
    where fragmentation matters.
    """
    n_overlapping = sum(1 for other in others if _overlap_size(target, other) > 0)
    if n_overlapping <= 1:
        return 1.0
    if gamma == "one":
        return 1.0
    if gamma == "reciprocal":
        return 1.0 / n_overlapping
    raise ValueError(f"unknown gamma {gamma!r}; expected 'one' or 'reciprocal'")


def _positional_weight(length: int, bias: str) -> np.ndarray:
    """Per-position weights inside a range, under one of the paper's biases.

    Flat weights every position equally and is the default. `front` weights the
    start of a range most heavily, which is the right choice when catching a
    fault early is what matters; `back` and `middle` are here for completeness.
    """
    if length <= 0:
        return np.zeros(0)
    positions = np.arange(1, length + 1)
    if bias == "flat":
        return np.ones(length)
    if bias == "front":
        return (length - positions + 1).astype(float)
    if bias == "back":
        return positions.astype(float)
    if bias == "middle":
        mid = (length + 1) / 2
        return (length / 2 - np.abs(positions - mid) + 1).astype(float)
    raise ValueError(f"unknown bias {bias!r}")


def _overlap_reward(
    target: tuple[int, int],
    others: list[tuple[int, int]],
    bias: str,
    gamma: str,
) -> float:
    """How much of `target` is covered by `others`, under a positional bias."""
    start, end = target
    length = end - start
    if length <= 0:
        return 0.0

    weights = _positional_weight(length, bias)
    total_weight = weights.sum()
    if total_weight <= 0:
        return 0.0

    covered = np.zeros(length, dtype=bool)
    for other in others:
        lo = max(start, other[0])
        hi = min(end, other[1])
        if hi > lo:
            covered[lo - start : hi - start] = True

    overlap = float(weights[covered].sum() / total_weight)
    return _cardinality_factor(target, others, gamma) * overlap


def _flat_rewards(
    targets: list[tuple[int, int]],
    others: list[tuple[int, int]],
    other_mask: np.ndarray,
    gamma: str,
) -> np.ndarray:
    """Overlap rewards for every target at once, for the flat bias only.

    Under a flat bias the reward for one target is just the fraction of its rows
    that the other side covers, times the cardinality factor. Both parts are
    computable in bulk:

    - coverage is a prefix sum over the other side's own label array, so the
      count for target [s, e) is one subtraction;
    - the cardinality factor needs the number of other-side ranges overlapping
      [s, e), which two `searchsorted` calls give, since the ranges come out of
      `to_events` already sorted and disjoint.

    That takes the whole function from a Python loop over every predicted range
    to a handful of vector operations. It matters because a permissive detector
    on the SKAB stream produces four thousand predicted ranges, and the loop
    version dominated the runtime of the entire multi-seed grid.
    """
    if not targets:
        return np.zeros(0)

    starts = np.fromiter((t[0] for t in targets), dtype=np.int64, count=len(targets))
    ends = np.fromiter((t[1] for t in targets), dtype=np.int64, count=len(targets))
    lengths = ends - starts

    prefix = np.concatenate(([0], np.cumsum(other_mask, dtype=np.int64)))
    covered = prefix[ends] - prefix[starts]
    with np.errstate(divide="ignore", invalid="ignore"):
        overlap = np.where(lengths > 0, covered / lengths, 0.0)

    if gamma == "one" or not others:
        return overlap
    if gamma != "reciprocal":
        raise ValueError(f"unknown gamma {gamma!r}; expected 'one' or 'reciprocal'")

    other_starts = np.fromiter((o[0] for o in others), dtype=np.int64, count=len(others))
    other_ends = np.fromiter((o[1] for o in others), dtype=np.int64, count=len(others))
    # Ranges overlapping [s, e) are those beginning before e and ending after s.
    n_overlapping = np.searchsorted(other_starts, ends, side="left") - np.searchsorted(
        other_ends, starts, side="right"
    )
    factor = np.where(n_overlapping > 1, 1.0 / np.maximum(n_overlapping, 1), 1.0)
    return overlap * factor


def range_metrics(
    y_true,
    y_pred,
    alpha: float = 0.0,
    bias: str = "flat",
    gamma: str = "reciprocal",
) -> ScoreTriple:
    """Range-based precision/recall (Tatbul et al., 2018).

    `alpha` splits recall between an existence reward (was the range detected at
    all) and an overlap reward (how much of it). We use alpha=0, pure overlap,
    for the headline table, and report alpha=1 as a sensitivity, since that is
    the setting for applications where detecting a range at all is the whole
    requirement.

    Precision carries no existence term by construction: a predicted range that
    barely clips a true one should not be rewarded for merely existing.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    y_true, y_pred = _check_pair(y_true, y_pred)

    real = to_events(y_true)
    predicted = to_events(y_pred)

    if not real:
        return ScoreTriple(0.0, 0.0, 0.0)

    if bias == "flat":
        # The vectorised path. `_flat_rewards` is exactly the loop below
        # specialised to a flat positional weight; the equivalence is tested in
        # tests/test_metrics_lit.py against the general implementation.
        recall_overlap = _flat_rewards(real, predicted, y_pred, gamma)
        if alpha > 0:
            # A true range exists-hits if any row inside it was flagged, which
            # the same prefix sum answers.
            starts = np.fromiter((r[0] for r in real), dtype=np.int64, count=len(real))
            ends = np.fromiter((r[1] for r in real), dtype=np.int64, count=len(real))
            hit_prefix = np.concatenate(([0], np.cumsum(y_pred, dtype=np.int64)))
            exists = ((hit_prefix[ends] - hit_prefix[starts]) > 0).astype(float)
        else:
            exists = np.zeros(len(real))
        recall = float(np.mean(alpha * exists + (1 - alpha) * recall_overlap))
        precision = (
            float(np.mean(_flat_rewards(predicted, real, y_true, gamma)))
            if predicted
            else 0.0
        )
        return ScoreTriple(precision, recall, _f1(precision, recall))

    recall_terms = []
    for r in real:
        exists = 1.0 if any(_overlap_size(r, p) > 0 for p in predicted) else 0.0
        overlap = _overlap_reward(r, predicted, bias, gamma)
        recall_terms.append(alpha * exists + (1 - alpha) * overlap)
    recall = float(np.mean(recall_terms))

    if not predicted:
        precision = 0.0
    else:
        precision = float(
            np.mean([_overlap_reward(p, real, bias, gamma) for p in predicted])
        )

    return ScoreTriple(precision, recall, _f1(precision, recall))


# --------------------------------------------------------------------------
# Xu et al. (2018): point-adjust, and Kim et al. (2022): PA%K
# --------------------------------------------------------------------------

def _adjust(y_true: np.ndarray, y_pred: np.ndarray, k: float) -> np.ndarray:
    """Point-adjust `y_pred`: fill in a true segment that was flagged enough.

    `k` is the fraction of the segment that must already be flagged before the
    whole segment is credited. k=0 is the original point-adjust rule, where one
    hit is enough; k=1 makes the adjustment a no-op.
    """
    adjusted = y_pred.copy()
    for start, end in to_events(y_true):
        segment = y_pred[start:end]
        if segment.size == 0:
            continue
        hit_fraction = float(segment.mean())
        if k <= 0.0:
            credited = bool(segment.any())
        else:
            credited = hit_fraction >= k
        if credited:
            adjusted[start:end] = 1
    return adjusted


def _point_scores(y_true: np.ndarray, y_pred: np.ndarray) -> ScoreTriple:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return ScoreTriple(precision, recall, _f1(precision, recall))


def point_metrics(y_true, y_pred) -> ScoreTriple:
    """Plain point-wise precision/recall/F1, every row scored on its own."""
    y_true, y_pred = _check_pair(y_true, y_pred)
    return _point_scores(y_true, y_pred)


def point_adjust_metrics(y_true, y_pred) -> ScoreTriple:
    """Point-adjust F1 (Xu et al., 2018).

    Reported for comparability with the deep-learning literature and treated as
    an upper bound rather than a measurement. One flagged row anywhere inside a
    fault credits the detector with the entire fault, which on SKAB's ~390-row
    blocks is a very large gift.
    """
    y_true, y_pred = _check_pair(y_true, y_pred)
    return _point_scores(y_true, _adjust(y_true, y_pred, k=0.0))


def point_adjust_k_metrics(y_true, y_pred, k: float = 0.2) -> ScoreTriple:
    """PA%K (Kim et al., 2022): adjust a segment only if K of it was flagged."""
    if not 0.0 <= k <= 1.0:
        raise ValueError("k must be in [0, 1]")
    y_true, y_pred = _check_pair(y_true, y_pred)
    return _point_scores(y_true, _adjust(y_true, y_pred, k=k))


def composite_f(y_true, y_pred) -> ScoreTriple:
    """Composite F-score (Garg et al., 2021): point precision, event recall.

    Identical by construction to `evaluate.event_metrics` at tolerance 0. It is
    defined again here so the literature name and our own name can be shown to
    agree, and so a reader checking one module against the other does not have
    to take it on trust. `tests/test_metrics_lit.py` asserts the equality.
    """
    from src.evaluate import event_metrics

    m = event_metrics(y_true, y_pred)
    return ScoreTriple(m.precision, m.recall, m.f1)


def all_metrics(y_true, y_pred, pa_k: float = 0.2) -> dict[str, float]:
    """Every convention at once, flattened for a results row."""
    y_true, y_pred = _check_pair(y_true, y_pred)
    out: dict[str, float] = {}
    out.update(_point_scores(y_true, y_pred).as_dict("point_"))
    out.update(composite_f(y_true, y_pred).as_dict("composite_"))
    out.update(range_metrics(y_true, y_pred).as_dict("range_"))
    out.update(range_metrics(y_true, y_pred, alpha=1.0).as_dict("range_alpha1_"))
    out.update(point_adjust_metrics(y_true, y_pred).as_dict("pa_"))
    out.update(point_adjust_k_metrics(y_true, y_pred, k=pa_k).as_dict("pak_"))
    return out
