"""Generate an event log that is consistent with each customer's real snapshot row.

**The events are synthetic.** The IBM file is one row per customer; nothing in it
says when bills were paid or how many tickets were raised. This module invents a
plausible history *behind* each real row so the repository can show what the
target roles ask for: features built from event-level data, computed as of a date,
with a point-in-time leakage test. The README says so in the first paragraph.

Consistency rules (checked in ``tests/test_events.py``):

* A customer with tenure *t* months has *t* monthly bills ending at the snapshot
  date; the bills sum to ``Total Charges`` within rounding.
* Payment behaviour depends on the real payment method (electronic check is late
  more often) and on the real services (fibre without tech support raises tickets).
* A modest churn signal is **planted**: in the last three months before the
  snapshot, customers who churned have more late payments, more tickets and a
  declining usage trend. The size is set in the config (``events.signal``). The
  PR-AUC lift from event features therefore measures that the SQL pipeline
  *recovers* a planted signal, not that real telco events would help by that much.
* Events can extend ``months_after`` past the snapshot for customers who did not
  churn (used by the weekly loop in phase 3), so point-in-time correctness is
  testable: a feature as of the snapshot must not change when later events are
  edited.

Grain: billing and usage are monthly; tickets and plan changes are dated events.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from churnprop.config import REPO_ROOT, Config

log = logging.getLogger(__name__)

SNAPSHOT_DATE = pd.Timestamp("2024-06-30")
TICKET_CATEGORIES = np.array(["billing", "network", "device", "service_change", "other"])
CHANGE_TYPES = np.array(["add_service", "remove_service", "downgrade", "upgrade"])


@dataclass
class EventsConfig:
    months_after: int = 3
    signal: float = 1.0  # multiplies every planted churn effect; 0 removes the signal
    price_rise_month: int = -4  # months relative to snapshot; bills rise 5% for fibre
    seed: int = 42

    @classmethod
    def from_config(cls, cfg: Config) -> EventsConfig:
        e = dict(cfg.extra.get("events", {}) or {})
        return cls(
            months_after=int(e.get("months_after", 3)),
            signal=float(e.get("signal", 1.0)),
            price_rise_month=int(e.get("price_rise_month", -4)),
            seed=int(e.get("seed", cfg.seed)),
        )


@dataclass
class EventLog:
    billing: pd.DataFrame
    usage: pd.DataFrame
    tickets: pd.DataFrame
    plan_changes: pd.DataFrame
    snapshot_date: pd.Timestamp

    def save(self, root: Path | str = REPO_ROOT / "data" / "events") -> None:
        root = Path(root)
        for name, df, datecol in [
            ("billing", self.billing, "bill_date"),
            ("usage", self.usage, "month_end"),
            ("tickets", self.tickets, "ticket_date"),
            ("plan_changes", self.plan_changes, "change_date"),
        ]:
            out = root / name
            if out.exists():
                shutil.rmtree(out)
            out.mkdir(parents=True, exist_ok=True)
            d = df.copy()
            d["month"] = d[datecol].dt.strftime("%Y-%m")
            # combine_chunks: the frames come from a per-customer concat and would otherwise be
            # written as one row group per customer (thousands of 1-row groups in every file)
            table = pa.Table.from_pandas(d, preserve_index=False).combine_chunks()
            pq.write_to_dataset(
                table,
                root_path=str(out),
                partition_cols=["month"],
                compression="zstd",
                row_group_size=200_000,
                existing_data_behavior="overwrite_or_ignore",
            )
        (root / "SNAPSHOT_DATE").write_text(self.snapshot_date.strftime("%Y-%m-%d"))
        log.info(
            "events saved to %s: %d bills, %d usage rows, %d tickets, %d plan changes",
            root,
            len(self.billing),
            len(self.usage),
            len(self.tickets),
            len(self.plan_changes),
        )

    @classmethod
    def load(cls, root: Path | str = REPO_ROOT / "data" / "events") -> EventLog:
        root = Path(root)
        snap = pd.Timestamp((root / "SNAPSHOT_DATE").read_text().strip())
        frames = {}
        for name in ("billing", "usage", "tickets", "plan_changes"):
            df = pd.read_parquet(root / name)
            frames[name] = df.drop(columns=[c for c in ("month",) if c in df.columns])
        return cls(snapshot_date=snap, **frames)

    @property
    def n_rows(self) -> int:
        return len(self.billing) + len(self.usage) + len(self.tickets) + len(self.plan_changes)


def _month_ends(snapshot: pd.Timestamp, offsets: np.ndarray) -> pd.DatetimeIndex:
    """Month-end dates at integer month offsets from the snapshot (0 = snapshot month)."""
    return pd.DatetimeIndex(
        [snapshot + pd.offsets.MonthEnd(int(k)) if k != 0 else snapshot for k in offsets]
    )


def simulate_events(df: pd.DataFrame, cfg: Config, ecfg: EventsConfig | None = None) -> EventLog:
    """Build the four event tables from the cleaned snapshot ``df`` (one row per customer)."""
    ecfg = ecfg or EventsConfig.from_config(cfg)
    rng = np.random.default_rng(ecfg.seed)
    snap = SNAPSHOT_DATE
    sig = ecfg.signal
    idc = cfg.data.id_col

    churn = df[cfg.data.target].to_numpy().astype(bool)
    tenure = df["Tenure Months"].to_numpy().astype(int)
    monthly = df["Monthly Charges"].to_numpy(float)
    total = df["Total Charges"].to_numpy(float)
    echeck = (df["Payment Method"] == "Electronic check").to_numpy()
    auto = df["Payment Method"].str.contains("automatic").to_numpy()
    fibre = (df["Internet Service"] == "Fiber optic").to_numpy()
    has_internet = (df["Internet Service"] != "No").to_numpy()
    no_support = (df["Tech Support"] != "Yes").to_numpy()
    streaming = ((df["Streaming TV"] == "Yes") | (df["Streaming Movies"] == "Yes")).to_numpy()
    senior = (df["Senior Citizen"] == "Yes").to_numpy()
    ids = df[idc].to_numpy()

    # Months relative to the snapshot: history is -(t-1) .. 0, future 1 .. months_after (non-churners only)
    rows_b, rows_u, rows_t, rows_p = [], [], [], []
    for i in range(len(df)):
        t = tenure[i]
        if t == 0:
            continue
        hist = np.arange(-(t - 1), 1)
        fut = np.arange(1, ecfg.months_after + 1) if not churn[i] else np.array([], dtype=int)
        offs = np.concatenate([hist, fut])
        n_hist = len(hist)
        dates = _month_ends(snap, offs)
        last3 = offs >= -2  # the three months ending at the snapshot
        last3 &= offs <= 0

        # ---- billing: amounts sum to Total Charges over the history, with a price rise for fibre
        base = np.full(len(offs), monthly[i])
        if fibre[i]:
            base[offs >= ecfg.price_rise_month] *= 1.05
        noise = rng.normal(0, 0.02, len(offs))
        amt = base * (1 + noise)
        # rescale history to hit Total Charges exactly (the snapshot's own number)
        hs = amt[:n_hist].sum()
        if hs > 0:
            amt[:n_hist] *= total[i] / hs
        # late payments: electronic check base 18%, mailed 10%, auto 2%; churners +25pp in last 3 months
        p_late = np.where(echeck[i], 0.18, np.where(auto[i], 0.02, 0.10))
        p_late = np.full(len(offs), p_late, dtype=float)
        p_late[last3] += 0.25 * sig * churn[i]
        late = rng.random(len(offs)) < np.clip(p_late, 0, 0.95)
        late_days = np.where(late, rng.integers(3, 30, len(offs)), 0)
        rows_b.append(
            pd.DataFrame(
                {
                    "customer_id": ids[i],
                    "bill_date": dates,
                    "amount": np.round(amt, 2),
                    "paid_date": dates + pd.to_timedelta(late_days, unit="D"),
                    "late_days": late_days,
                    "is_late": late,
                }
            )
        )

        # ---- usage (internet customers only): level by plan, churners trend down in last 3 months
        if has_internet[i]:
            level = (
                (180 if fibre[i] else 70)
                * (1.6 if streaming[i] else 1.0)
                * (0.8 if senior[i] else 1.0)
            )
            level *= rng.lognormal(0, 0.25)
            trend = 1 + 0.01 * np.arange(len(offs))  # gentle growth
            drop = np.where(last3 & churn[i], 1 - 0.15 * sig * (offs + 3), 1.0)  # -15%/-30%/-45%
            gb = level * trend * drop * rng.lognormal(0, 0.12, len(offs))
            rows_u.append(
                pd.DataFrame(
                    {
                        "customer_id": ids[i],
                        "month_end": dates,
                        "data_gb": np.round(gb, 1),
                        "streaming_hours": np.round(
                            gb
                            * (0.25 if streaming[i] else 0.05)
                            * rng.lognormal(0, 0.2, len(offs)),
                            1,
                        ),
                    }
                )
            )

        # ---- tickets: Poisson per month; fibre without support 2x; churners 2.5x in last 3 months
        lam = 0.06 * (2.0 if (fibre[i] and no_support[i]) else 1.0) * (1.3 if echeck[i] else 1.0)
        lam = np.full(len(offs), lam)
        lam[last3] *= 1 + 1.5 * sig * churn[i]
        n_t = rng.poisson(lam)
        for k in np.nonzero(n_t)[0]:
            for _ in range(int(n_t[k])):
                d = dates[k] - pd.Timedelta(days=int(rng.integers(0, 28)))
                cat = rng.choice(
                    TICKET_CATEGORIES,
                    p=[0.3, 0.3 if fibre[i] else 0.15, 0.15, 0.15, 0.1 if fibre[i] else 0.25],
                )
                csat = int(np.clip(rng.normal(3.6 - 1.2 * sig * churn[i] * last3[k], 1.0), 1, 5))
                rows_t.append((ids[i], d, cat, int(rng.integers(0, 10)), csat))

        # ---- plan changes: rare; churners downgrade more in last 6 months
        lam_p = np.full(len(offs), 0.02)
        lam_p[offs >= -5] += 0.06 * sig * churn[i]
        n_p = rng.poisson(lam_p)
        for k in np.nonzero(n_p)[0]:
            d = dates[k] - pd.Timedelta(days=int(rng.integers(0, 28)))
            p_down = 0.6 if churn[i] else 0.2
            ct = rng.choice(
                CHANGE_TYPES, p=[(1 - p_down) * 0.5, p_down * 0.5, p_down * 0.5, (1 - p_down) * 0.5]
            )
            rows_p.append((ids[i], d, ct))

    billing = pd.concat(rows_b, ignore_index=True)
    usage = (
        pd.concat(rows_u, ignore_index=True)
        if rows_u
        else pd.DataFrame(columns=["customer_id", "month_end", "data_gb", "streaming_hours"])
    )
    tickets = pd.DataFrame(
        rows_t, columns=["customer_id", "ticket_date", "category", "resolution_days", "csat"]
    )
    tickets["ticket_date"] = pd.to_datetime(tickets["ticket_date"])
    plan_changes = pd.DataFrame(rows_p, columns=["customer_id", "change_date", "change_type"])
    plan_changes["change_date"] = pd.to_datetime(plan_changes["change_date"])
    log.info(
        "simulated %d bills, %d usage rows, %d tickets, %d plan changes",
        len(billing),
        len(usage),
        len(tickets),
        len(plan_changes),
    )
    return EventLog(
        billing=billing, usage=usage, tickets=tickets, plan_changes=plan_changes, snapshot_date=snap
    )
