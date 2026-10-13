"""Typed view of ``configs/offers.yaml``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from churnprop.config import REPO_ROOT

DEFAULT_OFFERS = REPO_ROOT / "configs" / "offers.yaml"
CONTROL = "control"


@dataclass(frozen=True)
class Rule:
    when: str
    add: float
    why: str = ""


@dataclass(frozen=True)
class Offer:
    name: str
    label: str
    cost: float
    base: float
    rules: tuple[Rule, ...]


@dataclass(frozen=True)
class OffersConfig:
    n_weeks: int
    outcome_weeks: int
    cooldown_weeks: int
    baseline_4w_churn_mean: float
    weekly_budget: float
    exploration_share: float
    value_column: str
    value_margin: float
    offers: tuple[Offer, ...]
    heterogeneity_sd: float
    heterogeneity_clip: tuple[float, float]
    policies: tuple[str, ...]
    learner: str
    train_on: str
    warm_start_policy: str
    min_training_rows: int
    assumed_save_rate: float
    n_estimators: int
    raw: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def arms(self) -> tuple[str, ...]:
        """Arm 0 is control, then the offers in menu order."""
        return (CONTROL, *[o.name for o in self.offers])

    @property
    def costs(self) -> tuple[float, ...]:
        return (0.0, *[o.cost for o in self.offers])

    def offer(self, name: str) -> Offer:
        return next(o for o in self.offers if o.name == name)


def _deep_update(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = (
            _deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        )
    return out


def load_offers(path: str | Path | None = None, **overrides: Any) -> OffersConfig:
    with open(path or DEFAULT_OFFERS) as f:
        raw = yaml.safe_load(f)
    raw = _deep_update(raw, overrides)
    offers = tuple(
        Offer(
            name=name,
            label=o["label"],
            cost=float(o["cost"]),
            base=float(o["base"]),
            rules=tuple(Rule(**r) for r in o.get("rules", [])),
        )
        for name, o in raw["offers"].items()
    )
    h, u, b, v = raw["heterogeneity"], raw["uplift"], raw["budget"], raw["value"]
    hz = raw["horizon"]
    return OffersConfig(
        n_weeks=int(hz["n_weeks"]),
        outcome_weeks=int(hz["outcome_weeks"]),
        cooldown_weeks=int(hz["cooldown_weeks"]),
        baseline_4w_churn_mean=float(hz["baseline_4w_churn_mean"]),
        weekly_budget=float(b["weekly"]),
        exploration_share=float(b["exploration_share"]),
        value_column=v["column"],
        value_margin=float(v["margin"]),
        offers=offers,
        heterogeneity_sd=float(h["sd"]),
        heterogeneity_clip=(float(h["clip"][0]), float(h["clip"][1])),
        policies=tuple(raw["policies"]),
        learner=u["learner"],
        train_on=u.get("train_on", "exploration"),
        warm_start_policy=u["warm_start_policy"],
        min_training_rows=int(u["min_training_rows"]),
        assumed_save_rate=float(u["assumed_save_rate"]),
        n_estimators=int(u["n_estimators"]),
        raw=raw,
    )
