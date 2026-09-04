"""The three classical baselines added for the paper's comparison.

`test_models_and_detectors.py` covers the two models the original project was
built on. These are the comparison baselines a reader asks for: is the whole
adaptation result an artefact of using Half-Space Trees and Isolation Forest?
They are tested here rather than folded into that file so the older tests keep
saying exactly what they said.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import (
    MCD_SUPPORT_LADDER,
    MODEL_KINDS,
    SKLEARN_BASELINES,
    DegenerateWarmUp,
    build_model,
)

BASELINES = list(SKLEARN_BASELINES)


@pytest.fixture(scope="module")
def warm() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    return pd.DataFrame(
        rng.normal(0.0, 1.0, size=(600, 5)),
        columns=[f"sensor_{i}" for i in range(5)],
    )


@pytest.fixture(scope="module")
def shifted(warm) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    return pd.DataFrame(
        rng.normal(3.0, 1.0, size=(200, 5)), columns=warm.columns
    )


@pytest.mark.parametrize("kind", BASELINES)
class TestBaselineContract:
    def test_is_registered_and_constructible(self, kind):
        assert kind in MODEL_KINDS
        assert build_model(kind).name

    def test_reports_itself_as_offline(self, kind):
        assert build_model(kind).is_online is False

    def test_scoring_before_warm_up_is_an_error(self, kind, warm):
        with pytest.raises(RuntimeError, match="warm_up"):
            build_model(kind).score_one(warm.iloc[0].to_dict())

    def test_empty_warm_up_is_rejected(self, kind):
        with pytest.raises(ValueError, match="empty"):
            build_model(kind).warm_up(pd.DataFrame())

    def test_score_one_agrees_with_score_many(self, kind, warm, shifted):
        model = build_model(kind, seed=1)
        model.warm_up(warm)
        block = model.score_many(shifted)
        single = model.score_one(shifted.iloc[0].to_dict())
        assert single == pytest.approx(block[0])

    def test_higher_means_more_anomalous(self, kind, warm, shifted):
        """The project-wide convention. sklearn returns the opposite sign, so
        this is checking the negation was not dropped somewhere."""
        model = build_model(kind, seed=1)
        model.warm_up(warm)
        assert model.score_many(shifted).mean() > model.score_many(warm).mean()

    def test_flags_a_shifted_distribution(self, kind, warm, shifted):
        model = build_model(kind, seed=1)
        model.warm_up(warm)
        flagged = (model.score_many(shifted) > model.threshold).mean()
        assert flagged > 0.9

    def test_learn_one_does_nothing(self, kind, warm, shifted):
        """None of these supports an incremental update, so `learn_one` is a
        no-op by design. If one of them silently started updating, the static
        policy would stop being static and the comparison would be wrong."""
        model = build_model(kind, seed=1)
        model.warm_up(warm)
        before = model.score_many(shifted)
        for record in shifted.to_dict(orient="records"):
            model.learn_one(record)
        np.testing.assert_allclose(model.score_many(shifted), before)

    def test_threshold_follows_the_quantile(self, kind, warm):
        """A more permissive quantile must not produce a stricter threshold."""
        strict = build_model(kind, seed=1, threshold_quantile=0.99)
        loose = build_model(kind, seed=1, threshold_quantile=0.50)
        strict.warm_up(warm)
        loose.warm_up(warm)
        assert strict.threshold >= loose.threshold

    def test_column_order_does_not_change_the_score(self, kind, warm, shifted):
        """Scoring goes through a dict, so a caller handing columns back in a
        different order must still get the same answer."""
        model = build_model(kind, seed=1)
        model.warm_up(warm)
        row = shifted.iloc[0].to_dict()
        shuffled = dict(reversed(list(row.items())))
        assert model.score_one(shuffled) == pytest.approx(model.score_one(row))


class TestStandardisation:
    """The baselines are distance-based, so they are standardised on the warm-up
    window. The scaler is fixed there for the same reason the HST feature limits
    are: a rolling one would absorb the drift before the model saw it."""

    def test_scale_is_frozen_at_warm_up(self, warm):
        model = build_model("elliptic", seed=1)
        model.warm_up(warm)
        first = model._scale.copy()
        model.score_many(warm * 100.0)
        np.testing.assert_allclose(model._scale, first)

    def test_a_constant_sensor_does_not_divide_by_zero(self):
        frame = pd.DataFrame({"a": np.linspace(0, 1, 300), "b": np.zeros(300)})
        model = build_model("lof", seed=1)
        model.warm_up(frame)
        assert np.isfinite(model.threshold)
        assert np.isfinite(model.score_one({"a": 0.5, "b": 0.0}))

    def test_units_do_not_change_the_ranking(self, warm, shifted):
        """Rescaling one sensor by a constant is exactly what standardisation is
        meant to absorb. Without it, OC-SVM in particular would be measuring the
        units the sensors happen to be recorded in."""
        model_plain = build_model("ocsvm", seed=1)
        model_plain.warm_up(warm)
        plain = model_plain.score_many(shifted)

        rescaled_warm = warm.copy()
        rescaled_shift = shifted.copy()
        rescaled_warm["sensor_0"] *= 1000.0
        rescaled_shift["sensor_0"] *= 1000.0
        model_scaled = build_model("ocsvm", seed=1)
        model_scaled.warm_up(rescaled_warm)
        rescaled = model_scaled.score_many(rescaled_shift)

        np.testing.assert_allclose(plain, rescaled, rtol=1e-6, atol=1e-6)


class TestDegenerateWarmUp:
    """MCD picks the subset of points with the smallest covariance, and on a
    stream where most rows are identical that subset can be entirely constant.
    sklearn then refuses to fit. This is not hypothetical: on NAB's
    rogue_agent_key_updown, 859 of the 1,000 warm-up rows are exactly 0.0, and
    it killed a 90-minute run before this was handled."""

    @pytest.fixture
    def mostly_constant(self) -> pd.DataFrame:
        rng = np.random.default_rng(2)
        values = np.zeros(1000)
        varied = rng.choice(1000, size=140, replace=False)
        values[varied] = rng.normal(0.0, 3.0, size=140)
        return pd.DataFrame({"sensor_0": values})

    def test_mcd_escalates_its_support_and_fits(self, mostly_constant):
        model = build_model("elliptic", seed=0)
        model.warm_up(mostly_constant)
        assert model.support_fraction_used in MCD_SUPPORT_LADDER
        assert np.isfinite(model.threshold)

    def test_a_well_behaved_stream_keeps_the_default_support(self, warm):
        """Escalation gives up the robustness MCD exists for, so it must only
        happen where the default genuinely fails."""
        model = build_model("elliptic", seed=0)
        model.warm_up(warm)
        assert model.support_fraction_used is None

    def test_an_explicit_support_fraction_is_not_overridden(self, mostly_constant):
        with pytest.raises(DegenerateWarmUp):
            build_model("elliptic", seed=0, support_fraction=0.6).warm_up(mostly_constant)

    def test_a_fully_constant_window_raises_the_typed_error(self):
        """MCD does not fail on a constant window, which is the dangerous part:
        it fits at full support and then returns the same score for every row,
        including an obvious outlier. That would appear in the results as a
        detector that never raises a false alarm, so it has to be caught."""
        frame = pd.DataFrame({"sensor_0": np.zeros(500)})
        with pytest.raises(DegenerateWarmUp, match="identically"):
            build_model("elliptic", seed=0).warm_up(frame)

    @pytest.mark.parametrize("kind", BASELINES)
    def test_no_baseline_accepts_a_window_it_cannot_discriminate_on(self, kind):
        frame = pd.DataFrame({"a": np.zeros(400), "b": np.zeros(400)})
        with pytest.raises((DegenerateWarmUp, ValueError)):
            build_model(kind, seed=0).warm_up(frame)

    def test_the_error_is_a_runtime_error(self):
        assert issubclass(DegenerateWarmUp, RuntimeError)


class TestRegistry:
    def test_unknown_baseline_names_the_valid_ones(self):
        with pytest.raises(ValueError, match="expected one of"):
            build_model("not_a_model")

    def test_model_kinds_lists_everything_buildable(self, warm):
        for kind in MODEL_KINDS:
            model = build_model(kind, seed=0)
            model.warm_up(warm)
            assert np.isfinite(model.threshold)
