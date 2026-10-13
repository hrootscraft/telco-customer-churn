"""Every figure in ``reports/figures``. One palette, thin marks, recessive axes.

Colours follow a fixed categorical order (never cycled), one hue per model
family, and a single blue ramp for magnitude. Text is always ink, never the
series colour.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from churnprop.propensity.models import PRETTY  # noqa: E402

# Validated categorical palette (light surface), fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING_POS, DIVERGING_NEG = (
    "#e34948",
    "#2a78d6",
)  # diverging pair: red raises risk, blue lowers it
MODEL_ORDER = ["logistic", "knn", "svm", "random_forest", "xgboost", "lightgbm", "ensemble", "mlp"]
MODEL_COLOR = dict(zip(MODEL_ORDER, SERIES, strict=True))

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titleweight": "normal",
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.size": 10,
        "text.color": INK,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    }
)


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_churn_by_contract(table: pd.DataFrame, out_dir: Path) -> Path:
    t = table.loc[["Month-to-month", "One year", "Two year"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    y = np.arange(len(t))
    ax.barh(y, t["churn_rate"] * 100, color=BLUE_RAMP[4], height=0.55)
    for i, (r, n) in enumerate(zip(t["churn_rate"], t["customers"], strict=True)):
        ax.text(r * 100 + 0.8, i, f"{r * 100:.1f}%  (n={n:,})", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, t.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 55)
    ax.set_xlabel("Churn rate, %")
    ax.set_title("Month-to-month contracts churn 15× as often as two-year contracts")
    ax.grid(axis="y", visible=False)
    return _save(fig, out_dir, "01_churn_by_contract.png")


def plot_benchmark(
    test: pd.DataFrame, cv_summary: pd.DataFrame, regime: str, out_dir: Path
) -> Path:
    """Hold-out ROC-AUC and PR-AUC per model, with the CV sd as a whisker."""
    t = test.set_index("model")
    cv = cv_summary[cv_summary["regime"] == regime].set_index("model")
    order = [m for m in t.sort_values("roc_auc", ascending=False).index]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, metric, label in zip(axes, ["roc_auc", "pr_auc"], ["ROC-AUC", "PR-AUC"], strict=True):
        y = np.arange(len(order))
        vals = t.loc[order, metric]
        err = cv.loc[order, f"{metric}_std"] if f"{metric}_std" in cv else None
        ax.barh(
            y,
            vals,
            color=[MODEL_COLOR[m] for m in order],
            height=0.6,
            xerr=err,
            error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2},
        )
        pad = (err.fillna(0).to_numpy() if err is not None else np.zeros(len(order))) + 0.006
        for i, v in enumerate(vals):
            ax.text(v + pad[i], i, f"{v:.3f}", va="center", fontsize=8.5, color=INK)
        ax.set_yticks(y, [PRETTY[m] for m in order])
        ax.invert_yaxis()
        lo = max(0.0, float(vals.min()) - 0.08)
        ax.set_xlim(lo, min(1.0, float(vals.max()) + 0.07))
        ax.set_title(f"Hold-out {label}")
        ax.grid(axis="y", visible=False)
    fig.suptitle(
        f"Eight models on the 20% hold-out ({regime} resampling inside CV; whisker = CV sd)",
        y=1.03,
        fontsize=11,
    )
    return _save(fig, out_dir, "02_benchmark.png")


def plot_regimes(cv_summary: pd.DataFrame, out_dir: Path) -> Path:
    """What SMOTE and class weights buy, per model, on CV PR-AUC and Brier."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    regimes = ["none", "smote", "class_weight"]
    colors = dict(zip(regimes, [SERIES[0], SERIES[1], SERIES[2]], strict=True))
    for ax, metric, label in zip(
        axes,
        ["pr_auc_mean", "brier_mean"],
        ["CV PR-AUC (higher is better)", "CV Brier (lower is better)"],
        strict=True,
    ):
        models = [m for m in MODEL_ORDER if m in set(cv_summary["model"])]
        x = np.arange(len(models))
        w = 0.26
        for j, r in enumerate(regimes):
            sub = cv_summary[cv_summary["regime"] == r].set_index("model")
            vals = [sub.loc[m, metric] if m in sub.index else np.nan for m in models]
            ax.bar(
                x + (j - 1) * w, vals, width=w - 0.03, color=colors[r], label=r.replace("_", " ")
            )
        ax.set_xticks(x, [PRETTY[m].split(" (")[0] for m in models], rotation=30, ha="right")
        ax.set_title(label)
        ax.grid(axis="x", visible=False)
        finite = cv_summary[metric].dropna()
        ax.set_ylim(max(0, finite.min() - 0.05), finite.max() + 0.03)
    axes[0].legend(
        title="Resampling inside each CV fold",
        loc="upper center",
        bbox_to_anchor=(1.1, -0.42),
        ncol=3,
    )
    return _save(fig, out_dir, "03_resampling_regimes.png")


def plot_curves(scores: pd.DataFrame, models: list[str], out_dir: Path) -> Path:
    y = scores["y"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4))
    for m in models:
        p = scores[m].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        axes[0].plot(fpr, tpr, color=MODEL_COLOR[m], lw=1.8, label=PRETTY[m])
        axes[1].plot(rec, prec, color=MODEL_COLOR[m], lw=1.8, label=PRETTY[m])
    axes[0].plot([0, 1], [0, 1], color=GRID, lw=1, ls="--")
    axes[1].axhline(y.mean(), color=GRID, lw=1, ls="--")
    axes[1].text(0.01, y.mean() + 0.01, f"base rate {y.mean():.2f}", color=INK2, fontsize=8)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall")
    axes[1].legend(loc="upper right")
    return _save(fig, out_dir, "04_roc_pr_curves.png")


def plot_calibration(cal: pd.DataFrame, models: list[str], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls="--")
    for m in models:
        c = cal[cal["model"] == m]
        ax.plot(
            c["mean_predicted"],
            c["observed_rate"],
            marker="o",
            ms=4,
            lw=1.6,
            color=MODEL_COLOR[m],
            label=PRETTY[m],
        )
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed churn rate",
        title="Reliability on the hold-out",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(loc="upper left")
    return _save(fig, out_dir, "05_calibration.png")


def plot_shap_global(global_: pd.DataFrame, out_dir: Path, top: int = 12) -> Path:
    g = global_.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.barh(g["feature"], g["mean_abs_shap"], color=BLUE_RAMP[4], height=0.6)
    for i, (v, s) in enumerate(zip(g["mean_abs_shap"], g["share"], strict=True)):
        ax.text(v + 0.005, i, f"{s * 100:.0f}%", va="center", fontsize=8.5, color=INK)
    ax.set_xlabel("Mean |SHAP| (log-odds), share of total in label")
    ax.set_title("What drives the churn score, whole book")
    ax.grid(axis="y", visible=False)
    return _save(fig, out_dir, "06_shap_global.png")


def plot_shap_segments(by_segment: pd.DataFrame, seg_col: str, out_dir: Path, name: str) -> Path:
    sub = by_segment[by_segment["segment_col"] == seg_col]
    levels = list(dict.fromkeys(sub["segment"]))
    fig, axes = plt.subplots(1, len(levels), figsize=(3.1 * len(levels), 3.4), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, lv in zip(axes, levels, strict=True):
        s = sub[sub["segment"] == lv].sort_values("rank", ascending=False)
        colors = [DIVERGING_POS if v > 0 else DIVERGING_NEG for v in s["mean_shap"]]
        ax.barh(s["feature"], s["mean_shap"], color=colors, height=0.6)
        ax.axvline(0, color=INK2, lw=0.8)
        n, cr = int(s["n"].iloc[0]), float(s["churn_rate"].iloc[0])
        ax.set_title(f"{lv}\nn={n:,}, churn {cr * 100:.0f}%", fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle(
        f"Mean SHAP by {seg_col}: red pushes risk up, blue pulls it down", y=1.04, fontsize=11
    )
    return _save(fig, out_dir, name)


def plot_feature_selection(table: pd.DataFrame, out_dir: Path) -> Path:
    t = table.sort_values("permutation_importance_auc", ascending=True)
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    colors = [BLUE_RAMP[4] if k else SERIES[1] for k in t["stepwise_kept"]]
    ax.barh(t["feature"], t["permutation_importance_auc"], color=colors, height=0.6)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Permutation importance (ΔROC-AUC when shuffled), LightGBM")
    kept = int(t["stepwise_kept"].sum())
    ax.set_title(f"Backward stepwise keeps {kept} of {len(t)} features (orange = dropped)")
    ax.grid(axis="y", visible=False)
    return _save(fig, out_dir, "07_feature_selection.png")
