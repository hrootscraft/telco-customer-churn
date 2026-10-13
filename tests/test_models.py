import numpy as np
import pytest
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score

from churnprop.propensity.metrics import (
    expected_calibration_error,
    lift_at_top,
    recall_at_precision,
    score_all,
)
from churnprop.propensity.models import MODEL_NAMES, make_model
from churnprop.propensity.nn import TorchMLPClassifier


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_fits_and_beats_chance(name, small_split, cfg):
    m = make_model(name, cfg, resampling="none")
    m.fit(small_split.X_train, small_split.y_train)
    p = m.predict_proba(small_split.X_test)[:, 1]
    assert p.shape == (len(small_split.X_test),)
    assert np.all((p >= 0) & (p <= 1))
    assert roc_auc_score(small_split.y_test, p) > 0.7


def test_smote_lives_inside_the_pipeline(cfg):
    """SMOTE must be a pipeline step (applied in fit only), never applied to the data beforehand."""
    m = make_model("logistic", cfg, resampling="smote")
    assert isinstance(m.named_steps["smote"], SMOTE)
    assert list(m.named_steps)[0] == "prep"  # encode first, then oversample the encoded rows


def test_smote_does_not_change_test_predictions_shape(small_split, cfg):
    m = make_model("lightgbm", cfg, resampling="smote").fit(
        small_split.X_train, small_split.y_train
    )
    assert len(m.predict_proba(small_split.X_test)) == len(small_split.X_test)


def test_class_weight_regime(cfg):
    m = make_model("logistic", cfg, resampling="class_weight")
    assert m.named_steps["clf"].class_weight == "balanced"
    assert "smote" not in m.named_steps


def test_torch_mlp_api():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 5)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=400) > 0).astype(int)
    clf = TorchMLPClassifier(epochs=30, random_state=1).fit(X, y)
    p = clf.predict_proba(X)
    assert p.shape == (400, 2)
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-5)
    assert roc_auc_score(y, p[:, 1]) > 0.85
    assert len(clf.history_) <= 30
    # deterministic given the seed
    p2 = TorchMLPClassifier(epochs=30, random_state=1).fit(X, y).predict_proba(X)
    np.testing.assert_allclose(p, p2, atol=1e-5)


def test_metrics_on_perfect_and_constant_scores():
    y = np.array([0] * 70 + [1] * 30)
    perfect = y.astype(float)
    s = score_all(y, perfect)
    assert s["roc_auc"] == 1.0 and s["pr_auc"] == 1.0
    assert lift_at_top(y, perfect, 0.1) == pytest.approx(1 / 0.3)
    assert recall_at_precision(y, perfect, 0.5) == 1.0
    const = np.full(100, 0.3)
    assert expected_calibration_error(y, const) == pytest.approx(0.0, abs=1e-9)
    assert score_all(y, const)["accuracy"] == 0.7  # "nobody churns" accuracy
