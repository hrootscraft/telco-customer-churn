"""Drift monitoring for the weekly batches: features, scores, outcomes, uplift (D22).

Everything compares a scoring batch to **reference statistics captured at training
time** (``reference_stats``: per-feature bin proportions and quantiles on the training
frame, the training score distribution, the training churn rate). Nothing here needs
the training data at monitoring time, which is what lets the GitHub Actions job run
from the committed export alone.

Four questions, in the order a team should ask them:

1. **Did the inputs move?** PSI and a KS statistic per feature (numeric: decile bins
   fixed on the training frame; categorical: category shares).
2. **Did the scores move?** PSI of the score distribution and the shift in mean score.
   Inputs can move without the score moving (the model ignores that feature) and the
   score can move without any single input moving much.
3. **Is the model still right?** Once a batch's labels mature: observed vs predicted
   churn rate, ECE, PR-AUC against the model card. This is the only check that catches
   concept drift, and it arrives ``outcome_weeks`` late.
4. **Are the offers still doing what the uplift model says?** Realised minus predicted
   uplift on the loop's randomised exploration rows, per arm.

Thresholds live in ``configs/default.yaml: monitoring``; each check returns
``ok | warn | alert`` and the week's verdict is whether to retrain.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from churnprop.config import Config
from churnprop.propensity.metrics import expected_calibration_error, score_all

log = logging.getLogger(__name__)
EPS = 1e-4


@dataclass(frozen=True)
class MonitorConfig:
    n_bins: int = 10
    psi_warn: float = 0.10
    psi_alert: float = 0.25
    ks_alert: float = 0.15
    score_psi_alert: float = 0.10
    ece_alert: float = 0.05
    churn_rate_rel_alert: float = 0.30
    pr_auc_drop_alert: float = 0.05
    uplift_gap_alert: float = 0.04
    retrain_when: str = "alert"

    @classmethod
    def from_config(cls, cfg: Config) -> MonitorConfig:
        return cls(
            **{
                k: v
                for k, v in cfg.extra.get("monitoring", {}).items()
                if k in cls.__dataclass_fields__
            }
        )


# ----------------------------------------------------------------------------- reference


def _numeric_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    qs = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(qs)
    if len(edges) < 2:
        edges = np.array([edges[0] - 0.5, edges[0] + 0.5])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _props_numeric(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts = np.histogram(x, bins=edges)[0].astype(float)
    return counts / max(counts.sum(), 1)


def reference_stats(X: pd.DataFrame, scores: np.ndarray, y: np.ndarray, cfg: Config) -> dict:
    """Capture what monitoring needs about the training frame, as JSON-able numbers."""
    mcfg = MonitorConfig.from_config(cfg)
    cat = set(cfg.data.all_categorical)
    feats: dict[str, dict] = {}
    qgrid = np.linspace(0, 1, 21)
    for c in X.columns:
        s = X[c]
        if c in cat or s.dtype == object:
            vc = s.astype(str).value_counts(normalize=True)
            feats[c] = {
                "kind": "categorical",
                "levels": list(vc.index),
                "props": [float(v) for v in vc.values],
            }
        else:
            x = s.to_numpy(float)
            edges = _numeric_bins(x, mcfg.n_bins)
            feats[c] = {
                "kind": "numeric",
                "edges": [float(e) for e in edges],
                "props": [float(p) for p in _props_numeric(x, edges)],
                "quantiles": [float(q) for q in np.quantile(x, qgrid)],
                # the reference ECDF at those points (differs from the grid under ties)
                "quantile_cdf": [float(np.mean(x <= q)) for q in np.quantile(x, qgrid)],
                "mean": float(x.mean()),
                "std": float(x.std()),
            }
    score_edges = np.linspace(0, 1, mcfg.n_bins + 1)
    score_edges[0], score_edges[-1] = -np.inf, np.inf
    return {
        "n": int(len(X)),
        "features": feats,
        "score": {
            "edges": [float(e) for e in score_edges],
            "props": [float(p) for p in _props_numeric(np.asarray(scores, float), score_edges)],
            "mean": float(np.mean(scores)),
        },
        "churn_rate": float(np.mean(y)),
    }


# ----------------------------------------------------------------------------- statistics


def psi(ref_props, cur_props) -> float:
    """Population stability index: sum over bins of (cur − ref) · ln(cur / ref)."""
    r = np.clip(np.asarray(ref_props, float), EPS, None)
    c = np.clip(np.asarray(cur_props, float), EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def ks_from_quantiles(ref_quantiles, x: np.ndarray, ref_cdf=None) -> float:
    """KS statistic between a reference distribution and a sample, evaluated on a grid.

    The grid is the reference's 21 quantiles; ``ref_cdf`` is the reference ECDF at those
    points (needed for discrete features, where many quantiles coincide and the grid
    level is not the ECDF). Without it the grid levels are used.
    """
    q = np.asarray(ref_quantiles, float)
    levels = np.linspace(0, 1, len(q)) if ref_cdf is None else np.asarray(ref_cdf, float)
    x = np.sort(np.asarray(x, float))
    ecdf = np.searchsorted(x, q, side="right") / max(len(x), 1)
    return float(np.max(np.abs(ecdf - levels)))


def feature_drift(X: pd.DataFrame, ref: dict, mcfg: MonitorConfig) -> pd.DataFrame:
    rows = []
    for c, r in ref["features"].items():
        if c not in X.columns:
            continue
        if r["kind"] == "categorical":
            vc = X[c].astype(str).value_counts(normalize=True)
            cur = np.array([float(vc.get(level, 0.0)) for level in r["levels"]])
            new_share = float(1 - cur.sum())
            v = psi(r["props"], cur)
            rows.append(
                {
                    "feature": c,
                    "kind": "categorical",
                    "psi": v,
                    "ks": np.nan,
                    "new_levels_share": new_share,
                }
            )
        else:
            x = X[c].to_numpy(float)
            v = psi(r["props"], _props_numeric(x, np.array(r["edges"])))
            k = ks_from_quantiles(r["quantiles"], x, r.get("quantile_cdf"))
            rows.append(
                {
                    "feature": c,
                    "kind": "numeric",
                    "psi": v,
                    "ks": k,
                    "mean_shift_sd": float((x.mean() - r["mean"]) / (r["std"] or 1)),
                }
            )
    out = pd.DataFrame(rows)
    out["status"] = np.select(
        [
            out["psi"] >= mcfg.psi_alert,
            (out["ks"].fillna(0) >= mcfg.ks_alert),
            out["psi"] >= mcfg.psi_warn,
        ],
        ["alert", "alert", "warn"],
        default="ok",
    )
    return out.sort_values("psi", ascending=False).reset_index(drop=True)


def score_drift(scores: np.ndarray, ref: dict, mcfg: MonitorConfig) -> dict:
    s = np.asarray(scores, float)
    v = psi(ref["score"]["props"], _props_numeric(s, np.array(ref["score"]["edges"])))
    return {
        "score_psi": v,
        "mean_score": float(s.mean()),
        "ref_mean_score": ref["score"]["mean"],
        "mean_shift": float(s.mean() - ref["score"]["mean"]),
        "status": "alert"
        if v >= mcfg.score_psi_alert
        else ("warn" if v >= mcfg.score_psi_alert / 2 else "ok"),
    }


def outcome_drift(y: np.ndarray, scores: np.ndarray, card: dict, mcfg: MonitorConfig) -> dict:
    """Once labels are in: calibration and ranking against the model card's hold-out numbers."""
    y = np.asarray(y).astype(int)
    p = np.asarray(scores, float)
    m = score_all(y, p)
    ece = expected_calibration_error(y, p, mcfg.n_bins)
    pred, obs = float(p.mean()), float(y.mean())
    rel = (obs - pred) / max(pred, EPS)
    # a gap is an alert only if it is both material (relative threshold) and not noise
    # (z-test against the binomial standard error of the observed rate)
    se = float(np.sqrt(max(pred * (1 - pred), EPS) / max(len(y), 1)))
    z = (obs - pred) / se
    drop = card["holdout"]["pr_auc"] - m["pr_auc"]
    alerts = []
    if ece >= mcfg.ece_alert:
        alerts.append(f"ECE {ece:.3f}")
    if abs(rel) >= mcfg.churn_rate_rel_alert and abs(z) >= 3:
        alerts.append(f"observed churn {obs:.3f} vs predicted {pred:.3f} ({rel:+.0%}, z={z:.1f})")
    if drop >= mcfg.pr_auc_drop_alert:
        alerts.append(f"PR-AUC {m['pr_auc']:.3f}, {drop:.3f} below the card")
    return {
        "n_labelled": int(len(y)),
        "observed_churn": obs,
        "predicted_churn": pred,
        "churn_rel_gap": float(rel),
        "churn_gap_z": float(z),
        "ece": float(ece),
        "brier": float(m["brier"]),
        "pr_auc": float(m["pr_auc"]),
        "roc_auc": float(m["roc_auc"]),
        "card_pr_auc": float(card["holdout"]["pr_auc"]),
        "status": "alert" if alerts else "ok",
        "alerts": alerts,
    }


def uplift_drift(log_df: pd.DataFrame, week: int, arms: list[str], mcfg: MonitorConfig) -> dict:
    """Realised vs predicted uplift per arm on matured exploration rows up to ``week``."""
    ex = log_df[log_df["explore"] & log_df["matured"] & (log_df["week"] <= week)]
    ex = ex[ex["pred_uplift"].notna()] if "pred_uplift" in ex else ex.iloc[0:0]
    if len(ex) < 100:
        return {
            "status": "ok",
            "n_rows": int(len(ex)),
            "arms": {},
            "note": "not enough matured exploration rows with a prediction",
        }
    ctrl = ex[ex["arm"] == 0]["churned_window"].mean()
    out, alerts = {}, []
    for k, name in enumerate(arms):
        if k == 0:
            continue
        g = ex[ex["arm"] == k]
        if len(g) < 30:
            continue
        realised = float(ctrl - g["churned_window"].mean())
        predicted = float(g["pred_uplift"].mean())
        se = float(
            np.sqrt(
                ctrl * (1 - ctrl) / max((ex["arm"] == 0).sum(), 1)
                + g["churned_window"].var() / len(g)
            )
        )
        gap = realised - predicted
        out[name] = {
            "n": int(len(g)),
            "realised": realised,
            "predicted": predicted,
            "gap": gap,
            "se": se,
        }
        if abs(gap) >= mcfg.uplift_gap_alert and abs(gap) > 2 * se:
            alerts.append(f"{name}: realised {realised:+.3f} vs predicted {predicted:+.3f}")
    return {
        "status": "alert" if alerts else "ok",
        "n_rows": int(len(ex)),
        "arms": out,
        "alerts": alerts,
    }


# ----------------------------------------------------------------------------- one week


def monitor_week(
    week: int,
    X: pd.DataFrame,
    scores: np.ndarray,
    model_card: dict,
    ref: dict,
    cfg: Config,
    *,
    labelled: list[tuple[int, np.ndarray, np.ndarray]] | None = None,
    uplift_log: pd.DataFrame | None = None,
    arms: list[str] | None = None,
) -> dict:
    """Run every check for one batch; ``labelled`` are (batch_week, y, scores) for matured batches."""
    mcfg = MonitorConfig.from_config(cfg)
    fd = feature_drift(X, ref, mcfg)
    sd = score_drift(scores, ref, mcfg)
    outcomes = [
        {"batch_week": w, **outcome_drift(y, s, model_card, mcfg)} for w, y, s in (labelled or [])
    ]
    up = uplift_drift(uplift_log, week, arms or [], mcfg) if uplift_log is not None else None

    statuses = {
        "features": "alert"
        if (fd["status"] == "alert").any()
        else ("warn" if (fd["status"] == "warn").any() else "ok"),
        "scores": sd["status"],
    }
    if outcomes:
        statuses["outcomes"] = "alert" if any(o["status"] == "alert" for o in outcomes) else "ok"
    if up is not None:
        statuses["uplift"] = up["status"]
    worst = (
        "alert"
        if "alert" in statuses.values()
        else ("warn" if "warn" in statuses.values() else "ok")
    )
    reasons = [
        f"{r.feature}: PSI {r.psi:.2f}" + (f", KS {r.ks:.2f}" if r.kind == "numeric" else "")
        for r in fd[fd["status"] == "alert"].itertuples()
    ]
    if sd["status"] == "alert":
        reasons.append(
            f"score PSI {sd['score_psi']:.2f}, mean score {sd['ref_mean_score']:.3f} -> {sd['mean_score']:.3f}"
        )
    for o in outcomes:
        reasons += [f"batch {o['batch_week']}: {a}" for a in o["alerts"]]
    if up is not None:
        reasons += [f"uplift {a}" for a in up.get("alerts", [])]
    retrain = worst == "alert" if mcfg.retrain_when == "alert" else worst in ("warn", "alert")
    return {
        "week": int(week),
        "n_scored": int(len(X)),
        "model_version": int(model_card.get("version", 0)),
        "status": worst,
        "statuses": statuses,
        "retrain": bool(retrain),
        "reasons": reasons,
        "features": fd.to_dict(orient="records"),
        "top_drifted": fd.head(5)[["feature", "psi", "ks", "status"]].to_dict(orient="records"),
        "scores": sd,
        "outcomes": outcomes,
        "uplift": up,
    }


def write_report(report: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    js = out_dir / f"week_{report['week']:02d}.json"
    with open(js, "w") as f:
        json.dump(report, f, indent=2, default=_jsonable)
    md = out_dir / f"week_{report['week']:02d}.md"
    md.write_text(report_markdown(report))
    return js, md


def report_markdown(r: dict) -> str:
    lines = [
        f"# Monitoring, week {r['week']} (model v{r['model_version']}): **{r['status'].upper()}**",
        "",
        f"Scored {r['n_scored']:,} customers. Retrain recommended: **{'yes' if r['retrain'] else 'no'}**.",
        "",
    ]
    if r["reasons"]:
        lines += ["Alerts:", *[f"- {x}" for x in r["reasons"]], ""]
    lines += [
        "| Check | Status |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in r["statuses"].items()],
        "",
    ]
    lines += ["Most drifted features:", "", "| Feature | PSI | KS | Status |", "|---|---|---|---|"]
    for t in r["top_drifted"]:
        ks = (
            ""
            if t["ks"] is None or (isinstance(t["ks"], float) and np.isnan(t["ks"]))
            else f"{t['ks']:.3f}"
        )
        lines.append(f"| {t['feature']} | {t['psi']:.3f} | {ks} | {t['status']} |")
    s = r["scores"]
    lines += [
        "",
        f"Score PSI {s['score_psi']:.3f}; mean score {s['ref_mean_score']:.3f} at training, {s['mean_score']:.3f} now.",
    ]
    for o in r["outcomes"]:
        lines.append(
            f"Batch {o['batch_week']} matured: observed churn {o['observed_churn']:.3f} vs predicted "
            f"{o['predicted_churn']:.3f}, ECE {o['ece']:.3f}, PR-AUC {o['pr_auc']:.3f} (card {o['card_pr_auc']:.3f})."
        )
    if r.get("uplift") and r["uplift"].get("arms"):
        for name, a in r["uplift"]["arms"].items():
            lines.append(
                f"Uplift {name}: realised {a['realised']:+.3f} vs predicted {a['predicted']:+.3f} (n={a['n']})."
            )
    return "\n".join(lines) + "\n"


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and np.isnan(o):
        return None
    return str(o)
