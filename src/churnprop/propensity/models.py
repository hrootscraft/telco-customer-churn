"""The model zoo. Every entry is ``prep -> (resampler) -> classifier`` in one pipeline.

Resampling inside the pipeline is the point: ``imblearn.Pipeline`` applies SMOTE
only in ``fit``, so synthetic rows are made from the training fold alone and the
validation fold is scored as it is. The 2023 notebook oversampled before
splitting, which is how it reported 87% accuracy; this zoo cannot make that
mistake.
"""

from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from churnprop.config import Config
from churnprop.features import make_preprocessor
from churnprop.propensity.nn import TorchMLPClassifier

MODEL_NAMES = (
    "logistic",
    "knn",
    "svm",
    "random_forest",
    "xgboost",
    "lightgbm",
    "ensemble",
    "mlp",
)

PRETTY = {
    "logistic": "Logistic regression",
    "knn": "k-nearest neighbours",
    "svm": "SVM (RBF)",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "ensemble": "RF + XGB soft vote",
    "mlp": "Neural net (PyTorch MLP)",
}

FAMILY = {
    "logistic": "linear",
    "knn": "distance",
    "svm": "kernel",
    "random_forest": "tree ensemble",
    "xgboost": "gradient boosting",
    "lightgbm": "gradient boosting",
    "ensemble": "tree ensemble",
    "mlp": "neural net",
}


def _estimator(name: str, cfg: Config, class_weight):
    seed = cfg.seed
    n_est = cfg.benchmark.n_estimators or 400
    epochs = cfg.benchmark.mlp_epochs or 60
    # scale_pos_weight for the boosters when class_weight mode is on (≈ neg/pos ≈ 2.77)
    spw = 2.77 if class_weight else 1.0
    if name == "logistic":
        return LogisticRegression(C=1.0, max_iter=2000, class_weight=class_weight)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=25, weights="distance")
    if name == "svm":
        # Platt scaling on 3 inner folds gives SVC a probability output (SVC(probability=True) is deprecated)
        return CalibratedClassifierCV(
            SVC(C=1.0, gamma="scale", class_weight=class_weight, random_state=seed),
            method="sigmoid",
            cv=3,
            ensemble=False,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=n_est,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=-1,
            class_weight=class_weight,
            random_state=seed,
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=n_est,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_lambda=1.0,
            scale_pos_weight=spw,
            n_jobs=4,
            random_state=seed,
            eval_metric="logloss",
        )
    if name == "lightgbm":
        return LGBMClassifier(
            n_estimators=n_est,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=30,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            scale_pos_weight=spw,
            n_jobs=4,
            random_state=seed,
            verbose=-1,
        )
    if name == "ensemble":
        return VotingClassifier(
            [
                ("rf", _estimator("random_forest", cfg, class_weight)),
                ("xgb", _estimator("xgboost", cfg, class_weight)),
            ],
            voting="soft",
        )
    if name == "mlp":
        return TorchMLPClassifier(epochs=epochs, random_state=seed)
    raise KeyError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


def make_model(
    name: str,
    cfg: Config,
    feature_cols: list[str] | None = None,
    resampling: str | None = None,
) -> Pipeline:
    """Build ``prep -> [smote] -> clf`` for one model.

    ``resampling``: ``"smote"`` (default from config), ``"class_weight"`` or ``"none"``.
    """
    mode = resampling or cfg.benchmark.resampling
    class_weight = "balanced" if mode == "class_weight" else None
    steps = [("prep", make_preprocessor(cfg, feature_cols, scale=name not in ("random_forest",)))]
    if mode == "smote":
        steps.append(
            ("smote", SMOTE(k_neighbors=cfg.benchmark.smote_k_neighbors, random_state=cfg.seed))
        )
    steps.append(("clf", _estimator(name, cfg, class_weight)))
    return Pipeline(steps)
