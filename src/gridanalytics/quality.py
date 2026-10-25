"""Data quality checks on the AESO tables. Each check returns plain numbers so it can be reported and tested."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

PRICE_CAP = 999.99   # Alberta pool price cap before the restructured market
PRICE_FLOOR = 0.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hourly_continuity(idx: pd.DatetimeIndex) -> dict:
    d = pd.Series(idx).diff().dropna()
    return {"start": idx.min(), "end": idx.max(), "hours": int(len(idx)),
            "duplicates": int(pd.Index(idx).duplicated().sum()),
            "gaps": int((d != pd.Timedelta(hours=1)).sum()),
            "expected_hours": int((idx.max() - idx.min()) / pd.Timedelta(hours=1)) + 1}


def price_checks(price: pd.Series) -> dict:
    return {"missing": int(price.isna().sum()),
            "below_floor": int((price < PRICE_FLOOR).sum()),
            "above_cap": int((price > 1000.0).sum()),
            "at_cap_hours": int((price >= PRICE_CAP).sum()),
            "zero_price_hours": int((price <= 0).sum()),
            "max": float(price.max()), "min": float(price.min())}


def supply_balance(sysd: pd.DataFrame, k: float = 6.0) -> dict:
    """Metered generation over 5 MW plus net imports, as a share of Alberta Internal Load.

    The share sits below 1 because AIL also includes load served by on-site (behind the fence) generation
    and small generators. A sudden departure from the month's usual share points to a metering or data problem.
    """
    gen = sysd[["wind_mwh", "solar_mwh", "hydro_mwh", "storage_mwh", "thermal_mwh", "other_mwh"]].sum(axis=1)
    ratio = (gen + sysd["net_import_mw"].fillna(0.0)) / sysd["ail_mw"]
    month = ratio.index.tz_convert("America/Edmonton").strftime("%Y-%m")
    med = ratio.groupby(month).transform("median")
    mad = (ratio - med).abs().groupby(month).transform("median") * 1.4826
    flag = (ratio - med).abs() > k * mad
    by_year = ratio.groupby(ratio.index.year).mean()
    return {"share_mean": float(ratio.mean()), "share_by_year": {int(y): round(float(v), 3) for y, v in by_year.items()},
            "flagged_hours": int(flag.sum()), "flagged_share_pct": float(100 * flag.mean()),
            "flagged_examples": [t.isoformat() for t in ratio.index[flag][:5]]}


def stuck_values(s: pd.Series, run: int = 24) -> int:
    """Number of runs of at least `run` identical non-zero hourly values (a frozen meter or feed)."""
    v = s.to_numpy()
    nz = np.isfinite(v) & (v != 0)
    change = np.r_[True, v[1:] != v[:-1]]
    grp = np.cumsum(change)
    lengths = pd.Series(nz).groupby(grp).agg(["sum", "size"])
    return int(((lengths["sum"] == lengths["size"]) & (lengths["size"] >= run)).sum())


def asset_checks(assets_hourly: pd.DataFrame, asset_table: pd.DataFrame) -> dict:
    neg = (assets_hourly < -0.01).sum()
    stuck = {a: stuck_values(assets_hourly[a]) for a in assets_hourly.columns}
    cap = asset_table.set_index("asset_id")["maximum_capability_mw"]
    over = {}
    for a in assets_hourly.columns:
        c = cap.get(a)
        if c is not None and np.isfinite(c) and c > 0:
            n = int((assets_hourly[a] > 1.10 * c).sum())
            if n:
                over[a] = n
    return {"assets": int(assets_hourly.shape[1]),
            "negative_value_hours": int(neg.sum()),
            "assets_with_negative_values": [a for a, n in neg.items() if n > 0],
            "stuck_runs_24h": {a: n for a, n in stuck.items() if n > 0},
            "hours_above_110pct_capability": over,
            "no_output_assets": asset_table.asset_id[asset_table.first_output_utc.isna()].tolist()}


def constrained_checks(c: pd.DataFrame) -> dict:
    dup = c.duplicated(["time_utc", "planning_area", "fuel"]).sum()
    return {"rows": int(len(c)), "start": c.time_utc.min(), "end": c.time_utc.max(),
            "duplicates": int(dup), "negative_mw": int((c.constrained_mw < 0).sum()),
            "minutes_out_of_range": int(((c.minutes < 0) | (c.minutes > 60)).sum()),
            "planning_areas": int(c.planning_area.nunique()),
            "fuels": sorted(c.fuel.unique().tolist())}


def wind_forecast_checks(wf: pd.DataFrame) -> dict:
    bad = (wf.wpf_min > wf.wpf_ml + 1e-6) | (wf.wpf_ml > wf.wpf_max + 1e-6)
    return {"rows": int(len(wf)), "order_violations": int(bad.sum()),
            "duplicate_hours": int(wf.time_utc.duplicated().sum())}
