"""Congestion screening: constrained-down generation by planning area, regional balance, and intertie limits.

This is a screening study from public data, not a network model. It shows where and when transmission limits
already hold generation back, which is where locational prices would most likely separate from the system price.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TZ = "America/Edmonton"
FUEL_GROUP = {"WIND": "Wind", "SOLAR": "Solar", "THERMAL": "Thermal", "HYDRO": "Hydro", "ENERGY STORAGE": "Storage"}


def constrained_energy(c: pd.DataFrame) -> pd.DataFrame:
    """Constrained energy (MWh) per row: average constrained MW over the hour times one hour."""
    c = c.copy()
    c["mwh"] = c.constrained_mw * 1.0
    c["group"] = c.fuel.map(FUEL_GROUP)
    loc = c.time_utc.dt.tz_convert(TZ)
    c["year"] = loc.dt.year
    c["month"] = loc.dt.strftime("%Y-%m")
    c["hour"] = loc.dt.hour
    return c


def annual_by_fuel(c: pd.DataFrame) -> pd.DataFrame:
    t = c.pivot_table(index="year", columns="group", values="mwh", aggfunc="sum", fill_value=0) / 1e3
    t["Total"] = t.sum(axis=1)
    hrs = c.groupby("year").time_utc.nunique()
    t["hours_with_constraint"] = hrs
    return t.reset_index()


def curtailment_rate(c: pd.DataFrame, sysd: pd.DataFrame, years) -> pd.DataFrame:
    """Constrained energy as a share of what wind and solar could have produced (metered plus constrained)."""
    loc_year = sysd.index.tz_convert(TZ).year
    end = sysd.index.max()
    rows = []
    for y in years:
        m = (loc_year == y)
        cm = (c.year == y) & (c.time_utc <= end)
        for g in ("Wind", "Solar"):
            gen = sysd.loc[m, f"{g.lower()}_mwh"].sum()
            con = c.loc[cm & (c.group == g), "mwh"].sum()
            rows.append({"year": int(y), "group": g, "generation_gwh": gen / 1e3, "constrained_gwh": con / 1e3,
                         "curtailment_pct": 100 * con / (gen + con) if gen + con > 0 else np.nan})
    return pd.DataFrame(rows)


def by_planning_area(c: pd.DataFrame, years) -> pd.DataFrame:
    d = c[c.year.isin(years)]
    t = d.pivot_table(index="planning_area", columns="group", values="mwh", aggfunc="sum", fill_value=0) / 1e3
    t["Total"] = t.sum(axis=1)
    t["hours"] = d.groupby("planning_area").time_utc.nunique()
    return t.sort_values("Total", ascending=False).reset_index()


def constraint_vs_wind(c: pd.DataFrame, sysd: pd.DataFrame, years) -> pd.DataFrame:
    """Share of hours with any wind or solar constraint, by decile of wind output relative to capacity."""
    d = sysd[sysd.index.tz_convert(TZ).year.isin(years)]
    ren = c[c.group.isin(["Wind", "Solar"]) & c.year.isin(years)]
    hours = set(ren.time_utc)
    mwh = ren.groupby("time_utc").mwh.sum()
    cf = d.wind_mwh / d.wind_cap_mw
    q = pd.qcut(cf, 10, labels=False, duplicates="drop") + 1
    flag = pd.Series(d.index.isin(list(hours)), index=d.index)
    con = mwh.reindex(d.index).fillna(0)
    return pd.DataFrame({"wind_cf_pct": (100 * cf).groupby(q).mean(),
                         "hours_constrained_pct": (100 * flag).groupby(q).mean(),
                         "avg_constrained_mw": con.groupby(q).mean(),
                         "avg_price": d.pool_price.groupby(q).mean()}).reset_index(names="decile")


def price_when_constrained(c: pd.DataFrame, sysd: pd.DataFrame, years) -> dict:
    d = sysd[sysd.index.tz_convert(TZ).year.isin(years)]
    ren = c[c.group.isin(["Wind", "Solar"]) & c.year.isin(years)]
    mwh = ren.groupby("time_utc").mwh.sum().reindex(d.index).fillna(0)
    flag = mwh > 0
    return {"avg_price_all": float(d.pool_price.mean()),
            "avg_price_constrained_hours": float(d.pool_price[flag].mean()),
            "avg_price_other_hours": float(d.pool_price[~flag].mean()),
            "value_at_pool_price_musd": float((mwh * d.pool_price).sum() / 1e6),
            "constrained_hours_pct": float(100 * flag.mean())}


def regional_balance(pa: pd.DataFrame) -> pd.DataFrame:
    """Average hourly generation, load and net injection (generation minus load) by planning area and year."""
    pa = pa.copy()
    pa["year"] = pa.time_utc.dt.tz_convert(TZ).dt.year
    pa["net_mw"] = pa.csd_gen.fillna(0) - pa.act_load.fillna(0)
    t = pa.groupby(["year", "region", "planning_area"]).agg(gen_mw=("csd_gen", "mean"), load_mw=("act_load", "mean"),
                                                             net_mw=("net_mw", "mean"),
                                                             net_p95_mw=("net_mw", lambda s: s.quantile(0.95)))
    return t.reset_index()


def intertie_limits(it: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """BC import capability against its physical rating, and how often scheduled imports used all of it."""
    d = it[it.path == "BC_MATL_IMPORT"].copy()
    if d.empty or d.atc_mw.isna().all():
        d = it[it.path == "BC_IMPORT"].copy()
    d["year"] = d.time_local.dt.year
    d = d[d.ttc_mw > 0]
    d["atc_share_of_ttc"] = d.atc_mw / d.ttc_mw
    d["binding"] = (d.atc_mw > 0) & (d.sched_mw >= 0.98 * d.atc_mw)
    t = d.groupby("year").agg(hours=("atc_mw", "size"), avg_atc_mw=("atc_mw", "mean"), avg_ttc_mw=("ttc_mw", "mean"),
                              atc_share_pct=("atc_share_of_ttc", lambda s: 100 * s.mean()),
                              import_at_limit_pct=("binding", lambda s: 100 * s.mean()),
                              avg_sched_import_mw=("sched_mw", "mean")).reset_index()
    reasons = (d.assign(reason=d.limit_reason.fillna("No reason given").str.strip())
               .groupby("reason").size().sort_values(ascending=False).head(8)
               .rename("hours").reset_index())
    return t, reasons
