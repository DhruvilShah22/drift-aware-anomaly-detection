"""Anomaly scorers: an online Half-Space Trees model and a static baseline.

Both models expose the same small interface so `experiment.py` can swap them
without caring which is which:

    model.warm_up(frame)     # fit initial state and the alarm threshold
    model.score_one(x)       # higher means more anomalous
    model.predict_one(x)     # 1 if the score clears the threshold
    model.learn_one(x)       # online update; a no-op for the static baseline

Two decisions here are worth explaining, because they are what makes the
three-strategy comparison meaningful.

**Thresholds are fixed at warm-up, not rolling.** Half-Space Trees normalises
its raw mass score against a theoretical maximum that is essentially never
reached, so scores sit in a narrow, offset band — on SKAB, normal points land
near 0.94 and clearly anomalous ones near 0.995. An absolute threshold like 0.5
is therefore meaningless. I set the threshold to a high quantile of the scores
observed during warm-up. A *rolling* quantile would have been the other option,
but it quietly adapts to drift on its own, which would hide the very effect this
project is about.

**Feature ranges are also fixed at warm-up.** HST needs to know each feature's
range up front. `preprocessing.MinMaxScaler` is the usual answer, but it too
adapts continuously and would absorb the drift before the model ever saw it.
Instead I take per-feature min/max from the warm-up window and hold them. Values
that drift outside that range saturate into the boundary leaves of the trees,
which is the behaviour I want: they score as anomalous until the model is told
to adapt.

Adaptation, for every strategy, means the same thing: re-run `warm_up` on a
recent window, which rebuilds the model, its feature ranges, and its threshold.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
from river import anomaly
from sklearn.ensemble import IsolationForest

from src.thresholds import ThresholdRule, build_rule

# Warm-up min/max are widened by this fraction of the observed range. Without a
# margin, ordinary variation just past the edge of the warm-up window would
# saturate the boundary leaves and read as anomalous from the first step.
LIMIT_MARGIN = 0.10

# Default alarm threshold: a point is flagged if it scores above this quantile
# of the warm-up scores. Tuned in Phase 7.
DEFAULT_THRESHOLD_QUANTILE = 0.98


class AnomalyScorer(Protocol):
    """The interface `experiment.py` codes against."""

    threshold: float

    def warm_up(self, frame: pd.DataFrame) -> None: ...
    def score_one(self, x: dict[str, float]) -> float: ...
    def score_many(self, frame: pd.DataFrame) -> np.ndarray: ...
    def learn_one(self, x: dict[str, float]) -> None: ...

    @property
    def is_online(self) -> bool: ...


def _feature_limits(frame: pd.DataFrame, margin: float = LIMIT_MARGIN) -> dict[str, tuple[float, float]]:
    """Per-feature (min, max) from a warm-up window, widened by `margin`.

    A feature that never varies gets a unit-width band so the tree builder has
    something to split on rather than a degenerate zero-width interval.
    """
    limits: dict[str, tuple[float, float]] = {}
    for column in frame.columns:
        low = float(frame[column].min())
        high = float(frame[column].max())
        span = high - low
        if span <= 0:
            limits[column] = (low - 0.5, high + 0.5)
        else:
            pad = span * margin
            limits[column] = (low - pad, high + pad)
    return limits


class OnlineHalfSpaceTrees:
    """Half-Space Trees scored and updated one observation at a time.

    This is the model the periodic and drift-triggered strategies use. It keeps
    learning as the stream arrives, so it tracks slow changes on its own; what
    the strategies control is when it gets *reset*, which is the only way it can
    forget an old regime quickly.
    """

    def __init__(
        self,
        n_trees: int = 25,
        height: int = 8,
        window_size: int = 250,
        seed: int = 42,
        threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE,
        threshold_rule: ThresholdRule | str | float | None = None,
    ) -> None:
        self.n_trees = n_trees
        self.height = height
        self.window_size = window_size
        self.seed = seed
        self.threshold_quantile = threshold_quantile
        # `threshold_rule` supersedes `threshold_quantile` when given; the older
        # argument stays so existing callers keep working unchanged.
        self.threshold_rule = build_rule(
            threshold_rule if threshold_rule is not None else threshold_quantile
        )

        self.threshold = float("inf")
        self._model: anomaly.HalfSpaceTrees | None = None
        self._columns: list[str] = []

    @property
    def name(self) -> str:
        return "HalfSpaceTrees"

    @property
    def is_online(self) -> bool:
        return True

    def warm_up(self, frame: pd.DataFrame) -> None:
        """Build a fresh model on `frame` and set the alarm threshold from it.

        Called once at the start and again on every adaptation, so a reset and
        an initial fit follow exactly the same path.
        """
        if frame.empty:
            raise ValueError("cannot warm up on an empty frame")

        self._columns = list(frame.columns)
        self._model = anomaly.HalfSpaceTrees(
            n_trees=self.n_trees,
            height=self.height,
            window_size=self.window_size,
            limits=_feature_limits(frame),
            seed=self.seed,
        )

        records = frame.to_dict(orient="records")
        for record in records:
            self._model.learn_one(record)

        # HST returns 0 until its first window has been filled, so a warm-up
        # shorter than window_size leaves every score at 0 and the threshold
        # meaningless. Score after learning, and only keep the non-zero part.
        scores = np.array([self._model.score_one(r) for r in records])
        informative = scores[scores > 0]
        if informative.size == 0:
            raise ValueError(
                f"warm-up of {len(frame)} rows produced no usable scores; it must "
                f"be longer than window_size={self.window_size}"
            )
        self.threshold = self.threshold_rule(informative)

    def _require_model(self) -> anomaly.HalfSpaceTrees:
        if self._model is None:
            raise RuntimeError("warm_up must be called before scoring")
        return self._model

    def score_one(self, x: dict[str, float]) -> float:
        return float(self._require_model().score_one(x))

    def score_many(self, frame: pd.DataFrame) -> np.ndarray:
        model = self._require_model()
        return np.array([model.score_one(r) for r in frame.to_dict(orient="records")])

    def learn_one(self, x: dict[str, float]) -> None:
        self._require_model().learn_one(x)

    def predict_one(self, x: dict[str, float]) -> int:
        return int(self.score_one(x) > self.threshold)


class StaticIsolationForest:
    """An Isolation Forest fitted once and never updated.

    This is the naive baseline: whatever it learned during warm-up is what it
    believes for the rest of the stream. `learn_one` deliberately does nothing.

    sklearn's `score_samples` returns higher values for *more normal* points, so
    it is negated here to match the convention everywhere else in this project:
    higher means more anomalous.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float | str = "auto",
        seed: int = 42,
        threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE,
        threshold_rule: ThresholdRule | str | float | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.seed = seed
        self.threshold_quantile = threshold_quantile
        self.threshold_rule = build_rule(
            threshold_rule if threshold_rule is not None else threshold_quantile
        )

        self.threshold = float("inf")
        self._model: IsolationForest | None = None
        self._columns: list[str] = []

    @property
    def name(self) -> str:
        return "IsolationForest"

    @property
    def is_online(self) -> bool:
        return False

    def warm_up(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            raise ValueError("cannot warm up on an empty frame")

        self._columns = list(frame.columns)
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.seed,
        )
        self._model.fit(frame.to_numpy())

        scores = -self._model.score_samples(frame.to_numpy())
        self.threshold = self.threshold_rule(scores)

    def _require_model(self) -> IsolationForest:
        if self._model is None:
            raise RuntimeError("warm_up must be called before scoring")
        return self._model

    def score_one(self, x: dict[str, float]) -> float:
        row = np.array([[x[c] for c in self._columns]])
        return float(-self._require_model().score_samples(row)[0])

    def score_many(self, frame: pd.DataFrame) -> np.ndarray:
        """Score a block at once.

        Scoring row by row through sklearn costs milliseconds per call, which is
        wasteful for a model that cannot change between resets anyway. The
        experiment runner uses this for the static and periodic strategies,
        where the model is provably constant across a whole segment.
        """
        return -self._require_model().score_samples(frame[self._columns].to_numpy())

    def learn_one(self, x: dict[str, float]) -> None:
        """No-op. This baseline exists precisely because it does not adapt."""

    def predict_one(self, x: dict[str, float]) -> int:
        return int(self.score_one(x) > self.threshold)


def build_model(kind: str, **kwargs) -> AnomalyScorer:
    """Construct a scorer by name."""
    if kind == "hst":
        return OnlineHalfSpaceTrees(**kwargs)
    if kind == "iforest":
        return StaticIsolationForest(**kwargs)
    raise ValueError(f"unknown model kind {kind!r}; expected 'hst' or 'iforest'")
