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


def plot_shap_global(
    global_: pd.DataFrame,
    out_dir: Path,
    top: int = 12,
    name: str = "06_shap_global.png",
    title: str = "What drives the churn score, whole book",
) -> Path:
    g = global_.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.barh(g["feature"], g["mean_abs_shap"], color=BLUE_RAMP[4], height=0.6)
    for i, (v, s) in enumerate(zip(g["mean_abs_shap"], g["share"], strict=True)):
        ax.text(v + 0.005, i, f"{s * 100:.0f}%", va="center", fontsize=8.5, color=INK)
    ax.set_xlabel("Mean |SHAP| (log-odds), share of total in label")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    return _save(fig, out_dir, name)


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


def plot_event_feature_lift(comp: pd.DataFrame, out_dir: Path) -> Path:
    """Hold-out PR-AUC per model, snapshot features vs snapshot + event features."""
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = np.arange(len(comp))
    h = 0.36
    ax.barh(
        y - h / 2,
        comp["pr_auc_snapshot"],
        height=h - 0.03,
        color=BLUE_RAMP[2],
        label="snapshot features",
    )
    ax.barh(
        y + h / 2,
        comp["pr_auc_events"],
        height=h - 0.03,
        color=BLUE_RAMP[5],
        label="snapshot + event features",
    )
    for i, (a, b) in enumerate(zip(comp["pr_auc_snapshot"], comp["pr_auc_events"], strict=True)):
        ax.text(a + 0.004, i - h / 2, f"{a:.3f}", va="center", fontsize=8.5, color=INK)
        ax.text(
            b + 0.004, i + h / 2, f"{b:.3f}  (+{b - a:.3f})", va="center", fontsize=8.5, color=INK
        )
    ax.set_yticks(y, [PRETTY[m] for m in comp["model"]])
    ax.invert_yaxis()
    lo = float(min(comp["pr_auc_snapshot"].min(), comp["pr_auc_events"].min())) - 0.05
    hi = float(comp["pr_auc_events"].max()) + 0.08
    ax.set_xlim(max(0, lo), min(1, hi))
    ax.set_xlabel("Hold-out PR-AUC")
    ax.set_title("What point-in-time event features add (planted signal, see D9)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    return _save(fig, out_dir, "10_event_feature_lift.png")


# ----------------------------------------------------------------------------- phase 3

POLICY_LABEL = {
    "no_offers": "No offers",
    "random_outreach": "Random outreach",
    "risk_top": "Highest churn risk",
    "risk_value": "Risk × value, fixed save rate",
    "uplift": "Learned uplift + budget",
    "oracle": "Oracle (planted truth)",
}
POLICY_COLOR = {
    "no_offers": "#9a9890",
    "random_outreach": SERIES[1],
    "risk_top": SERIES[3],
    "risk_value": SERIES[4],
    "uplift": SERIES[0],
    "oracle": "#0d366b",
}
ARM_COLOR = {"bill_credit": SERIES[0], "tech_bundle": SERIES[2], "contract_upgrade": SERIES[6]}


def plot_policy_comparison(comp: pd.DataFrame, n_weeks: int, out_dir: Path) -> Path:
    """Expected net value per week (planted truth) per policy, with the realised replays."""
    c = comp[comp["policy"] != "no_offers"].copy()
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    y = np.arange(len(c))
    ax.barh(
        y,
        c["truth_reward_per_week"] / 1000,
        color=[POLICY_COLOR[p] for p in c["policy"]],
        height=0.58,
        label="expected value under the planted responses",
    )
    if "realised_net_mean" in c and c["realised_net_mean"].notna().any():
        ok = c["realised_net_mean"].notna()
        ax.errorbar(
            c.loc[ok, "realised_net_mean"] / n_weeks / 1000,
            y[ok.to_numpy()],
            xerr=c.loc[ok, "realised_net_sd"] / n_weeks / 1000,
            fmt="o",
            color=INK,
            ms=5,
            capsize=3,
            lw=1.2,
            label="realised, mean ± sd over replays of the coin flips",
        )
    for i, v in enumerate(c["truth_reward_per_week"] / 1000):
        ax.text(
            v + (0.6 if v >= 0 else -0.6),
            i - 0.33,
            f"${v:,.1f}k",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=8.5,
            color=INK,
        )
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, [POLICY_LABEL[p] for p in c["policy"]])
    ax.invert_yaxis()
    ax.set_xlabel("Net value of the week's offers vs doing nothing, $ thousands per week")
    ax.set_title("Same budget, same customers, same draws: what each targeting rule earns")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right", fontsize=8.5)
    lo = min(0, float(c["truth_reward_per_week"].min() / 1000)) - 6
    ax.set_xlim(lo, float(c["truth_reward_per_week"].max() / 1000) * 1.3)
    return _save(fig, out_dir, "12_policy_comparison.png")


def plot_ope_power(comp: pd.DataFrame, out_dir: Path) -> Path:
    """What the exploration holdout alone can say: DR estimate ± 95% CI against the truth."""
    c = comp[comp["policy"].isin(["risk_value", "uplift", "oracle"])].copy()
    c = c[c["dr_per_week"].notna()] if "dr_per_week" in c else c.iloc[0:0]
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    if len(c) == 0:
        ax.text(0.5, 0.5, "not enough exploration rows", ha="center", transform=ax.transAxes)
        return _save(fig, out_dir, "16_ope_power.png")
    y = np.arange(len(c))
    ax.errorbar(
        c["dr_per_week"] / 1000,
        y,
        xerr=1.96 * c["dr_se_per_week"] / 1000,
        fmt="o",
        color=INK,
        ms=6,
        capsize=4,
        lw=1.4,
        label="doubly robust estimate from the holdout, 95% CI",
    )
    ax.scatter(
        c["truth_per_week_same_weeks"] / 1000,
        y,
        marker="D",
        s=48,
        color=[POLICY_COLOR[p] for p in c["policy"]],
        zorder=3,
        label="planted truth, same weeks",
    )
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, [POLICY_LABEL[p] for p in c["policy"]])
    ax.invert_yaxis()
    ax.set_xlabel("Net value vs doing nothing, $ thousands per week")
    up = c[c["policy"] == "uplift"]
    if len(up):
        r = up.iloc[0]
        ax.set_title(
            "The holdout alone cannot see it yet: detectable effect ≈ "
            f"\\${r['mde_per_week_80pct_power'] / 1000:,.0f}k/week against a true "
            f"\\${r['truth_per_week_same_weeks'] / 1000:,.0f}k; "
            f"~{r['weeks_to_detect_truth']:,.0f} weeks of logging to detect it",
            fontsize=10,
        )
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False, fontsize=8.5)
    return _save(fig, out_dir, "16_ope_power.png")


def plot_learning_curve(weekly: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    for pol in ("risk_top", "risk_value", "uplift", "oracle"):
        w = weekly[weekly["policy"] == pol]
        if len(w) == 0:
            continue
        ax.plot(
            w["week"],
            w["truth_reward"] / 1000,
            color=POLICY_COLOR[pol],
            lw=2,
            marker="o",
            ms=4,
            label=POLICY_LABEL[pol],
        )
    up = weekly[(weekly["policy"] == "uplift") & (weekly["policy_used"] == "uplift")]
    if len(up):
        first = int(up["week"].min())
        ax.axvline(first - 0.5, color=GRID, lw=1.2, ls="--")
        ax.text(
            first - 0.4,
            ax.get_ylim()[1] * 0.62,
            "uplift model takes over\n(first matured outcomes)",
            fontsize=8.5,
            color=INK2,
            va="top",
        )
    ax.set_xlabel("Week")
    ax.set_ylabel("Expected net value, \\$k per week")
    ax.set_title(
        "Week by week: the learned policy starts as risk × value and improves as outcomes mature"
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False)
    ax.set_xticks(sorted(weekly["week"].unique()))
    return _save(fig, out_dir, "13_learning_curve.png")


def plot_offer_mix(weekly_uplift: pd.DataFrame, arms: tuple[str, ...], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    bottom = np.zeros(len(weekly_uplift))
    for k in arms[1:]:
        v = weekly_uplift[f"n_{k}"].to_numpy()
        ax.bar(
            weekly_uplift["week"],
            v,
            bottom=bottom,
            color=ARM_COLOR[k],
            width=0.7,
            label=k.replace("_", " "),
            edgecolor=SURFACE,
            linewidth=1.5,
        )
        bottom += v
    ax.set_xlabel("Week")
    ax.set_ylabel("Offers sent (policy + exploration)")
    ax.set_title("Offer mix under the learned policy")
    ax.legend(loc="upper right", ncol=3)
    ax.set_xticks(weekly_uplift["week"])
    ax.grid(axis="x", visible=False)
    return _save(fig, out_dir, "14_offer_mix.png")


def plot_budget_frontier(frontier: pd.DataFrame, budget: float, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax = axes[0]
    ax.plot(
        frontier["spend"] / 1000,
        frontier["truth_reward"] / 1000,
        color=BLUE_RAMP[5],
        lw=2,
        marker="o",
        ms=4,
    )
    ax.axvline(budget / 1000, color=GRID, lw=1.2, ls="--")
    ax.text(
        budget / 1000 * 1.02,
        ax.get_ylim()[0] + 1,
        "configured weekly budget",
        fontsize=8.5,
        color=INK2,
        rotation=90,
        va="bottom",
    )
    ax.set_xlabel("Weekly spend, $k")
    ax.set_ylabel("Expected net value, $k per week")
    ax.set_title("Diminishing returns to budget (oracle allocation, week 1)")
    ax = axes[1]
    ax.plot(frontier["spend"] / 1000, frontier["lambda"], color=SERIES[1], lw=2, marker="o", ms=4)
    ax.set_xlabel("Weekly spend, $k")
    ax.set_ylabel("Lagrange multiplier λ (\\$ of value required per \\$ of cost)")
    ax.set_title("The knob: λ falls as the budget loosens")
    return _save(fig, out_dir, "15_budget_frontier.png")
