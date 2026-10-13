"""Typed configuration loaded from ``configs/default.yaml``.

Every tunable number lives in YAML so a reviewer can see the choices in one place.
This module turns that YAML into frozen dataclasses so a typo fails loudly
(``cfg.benchmark.n_fold`` raises) instead of silently returning ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class DataConfig:
    raw_path: str
    id_col: str
    target: str
    leakage_cols: tuple[str, ...]
    drop_cols: tuple[str, ...]
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    engineered_numeric: tuple[str, ...] = ()
    engineered_categorical: tuple[str, ...] = ()

    @property
    def all_numeric(self) -> tuple[str, ...]:
        return self.numeric + self.engineered_numeric

    @property
    def all_categorical(self) -> tuple[str, ...]:
        return self.categorical + self.engineered_categorical

    @property
    def feature_cols(self) -> tuple[str, ...]:
        return self.all_numeric + self.all_categorical


@dataclass(frozen=True)
class BenchmarkConfig:
    test_size: float
    n_folds: int
    n_repeats: int
    resampling: str
    smote_k_neighbors: int
    models: tuple[str, ...]
    top_decile: float
    precision_floor: float
    calibration_bins: int
    fast: dict[str, Any] = field(default_factory=dict)
    # Set by ``load_config(fast=True)``; read by the model zoo.
    n_estimators: int | None = None
    mlp_epochs: int | None = None


@dataclass(frozen=True)
class StepwiseConfig:
    alpha: float
    l1_C: float
    n_permutations: int


@dataclass(frozen=True)
class ExplainConfig:
    model: str
    segments: tuple[str, ...]
    n_customers_waterfall: int


@dataclass(frozen=True)
class Config:
    seed: int
    data: DataConfig
    benchmark: BenchmarkConfig
    stepwise: StepwiseConfig
    explain: ExplainConfig
    fast: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def raw_path(self) -> Path:
        p = Path(self.data.raw_path)
        return p if p.is_absolute() else REPO_ROOT / p


def _tuple(x: Any) -> tuple:
    return tuple(x) if x is not None else ()


def _deep_update(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, *, fast: bool = False, **overrides: Any) -> Config:
    """Load the YAML config, apply ``overrides`` (nested dicts merge), build the dataclasses.

    ``fast=True`` applies the ``benchmark.fast`` block (fewer folds, trees, epochs) for CI.
    """
    with open(path or DEFAULT_CONFIG) as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    raw = _deep_update(raw, overrides)

    d = raw["data"]
    data = DataConfig(
        raw_path=d["raw_path"],
        id_col=d["id_col"],
        target=d["target"],
        leakage_cols=_tuple(d.get("leakage_cols")),
        drop_cols=_tuple(d.get("drop_cols")),
        numeric=_tuple(d["numeric"]),
        categorical=_tuple(d["categorical"]),
        engineered_numeric=_tuple(d.get("engineered_numeric")),
        engineered_categorical=_tuple(d.get("engineered_categorical")),
    )
    b = dict(raw["benchmark"])
    fast_block = dict(b.pop("fast", {}) or {})
    bench = BenchmarkConfig(models=_tuple(b.pop("models")), fast=fast_block, **b)
    if fast:
        bench = replace(
            bench,
            n_repeats=fast_block.get("n_repeats", bench.n_repeats),
            n_folds=fast_block.get("n_folds", bench.n_folds),
            n_estimators=fast_block.get("n_estimators"),
            mlp_epochs=fast_block.get("mlp_epochs"),
        )
    s = raw["stepwise"]
    e = raw["explain"]
    known = {"seed", "data", "benchmark", "stepwise", "explain"}
    return Config(
        seed=int(raw["seed"]),
        data=data,
        benchmark=bench,
        stepwise=StepwiseConfig(**s),
        explain=ExplainConfig(
            model=e["model"],
            segments=_tuple(e["segments"]),
            n_customers_waterfall=int(e["n_customers_waterfall"]),
        ),
        fast=fast,
        extra={k: v for k, v in raw.items() if k not in known},
    )
