"""Wind and solar value: capture prices, value factors, ramps, and AESO's day-ahead wind forecast."""
from __future__ import annotations

import numpy as np
import pandas as pd

TZ = "America/Edmonton"


def capture_price(gen: pd.Series, price: pd.Series) -> float:
    """Generation-weighted average pool price ($/MWh): what a plant earns per MWh at the pool price."""
    g = gen.clip(lower=0)
    return float((g * price).sum() / g.sum()) if g.sum() > 0 else np.nan


def annual_summary(sysd: pd.DataFrame) -> pd.DataFrame:
    yr = sysd.index.tz_convert(TZ).year
    rows = []
    for y, d in sysd.groupby(yr):
        avg = float(d.pool_price.mean())
        r = {"year": int(y), "hours": int(len(d)), "avg_price": avg, "ail_twh": d.ail_mw.sum() / 1e6,
             "peak_ail_mw": float(d.ail_mw.max()), "zero_price_hours": int((d.pool_price <= 0.01).sum()),
             "cap_price_hours": int((d.pool_price >= 999.99).sum())}
        for g in ("wind", "solar", "storage"):
            gen = d[f"{g}_mwh"]
            cap = d[f"{g}_cap_mw"]
            r[f"{g}_gwh"] = gen.sum() / 1e3
            r[f"{g}_share_pct"] = 100 * gen.sum() / d.ail_mw.sum()
            r[f"{g}_cap_mw_end"] = float(cap.iloc[-1])
            r[f"{g}_cf_pct"] = 100 * gen.sum() / cap.sum() if cap.sum() > 0 else np.nan
            cp = capture_price(gen, d.pool_price)
            r[f"{g}_capture"] = cp
            r[f"{g}_value_factor"] = cp / avg if gen.sum() > 0 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def asset_capture(assets_hourly: pd.DataFrame, price: pd.Series, assets: pd.DataFrame, year: int,
                  groups=("Wind", "Solar", "Storage")) -> pd.DataFrame:
    """Capture price, value factor and capacity factor per asset for one (full) year."""
    loc_year = assets_hourly.index.tz_convert(TZ).year
    m = loc_year == year
    p = price[m]
    avg = float(p.mean())
    rows = []
    for _, a in assets[assets.group.isin(groups)].iterrows():
        g = assets_hourly.loc[m, a.asset_id].fillna(0).clip(lower=0)
        if g.sum() < 1000 or (g > 0.1).mean() < 0.05:   # skip assets barely running that year
            continue
        cp = capture_price(g, p)
        cap = a.capacity_mw
        full = (g > 0.1).idxmax() <= p.index[0] + pd.Timedelta(days=31)
        rows.append({"asset_id": a.asset_id, "asset_name": a.asset_name, "group": a.group,
                     "capacity_mw": cap, "gwh": g.sum() / 1e3, "capture": cp, "value_factor": cp / avg,
                     "cf_pct": 100 * g.mean() / cap if cap and np.isfinite(cap) else np.nan,
                     "full_year": bool(full)})
    return pd.DataFrame(rows).sort_values(["group", "value_factor"], ascending=[True, False])


def price_by_wind_decile(sysd: pd.DataFrame, year: int) -> pd.DataFrame:
    d = sysd[sysd.index.tz_convert(TZ).year == year]
    share = d.wind_mwh / d.ail_mw
    q = pd.qcut(share, 10, labels=False, duplicates="drop")
    out = d.groupby(q).agg(wind_share_pct=("wind_mwh", lambda s: np.nan), avg_price=("pool_price", "mean"),
                           median_price=("pool_price", "median"), zero_price_pct=("pool_price", lambda s: 100 * (s <= 0.01).mean()))
    out["wind_share_pct"] = (100 * share).groupby(q).mean()
    out.index = out.index + 1
    out.index.name = "decile"
    return out.reset_index()


def hourly_profile(sysd: pd.DataFrame, year: int) -> pd.DataFrame:
    d = sysd[sysd.index.tz_convert(TZ).year == year]
    h = d.index.tz_convert(TZ).hour
    net = d.ail_mw - d.wind_mwh - d.solar_mwh
    return pd.DataFrame({"hour": range(24),
                         "price": d.pool_price.groupby(h).mean().to_numpy(),
                         "ail": d.ail_mw.groupby(h).mean().to_numpy(),
                         "wind": d.wind_mwh.groupby(h).mean().to_numpy(),
                         "solar": d.solar_mwh.groupby(h).mean().to_numpy(),
                         "net_load": net.groupby(h).mean().to_numpy()})


def ramps(sysd: pd.DataFrame) -> pd.DataFrame:
    """Hourly wind ramps and three-hour net load ramps by year (the flexibility the thermal fleet must supply)."""
    yr = sysd.index.tz_convert(TZ).year
    wind_d = sysd.wind_mwh.diff()
    net = sysd.ail_mw - sysd.wind_mwh - sysd.solar_mwh
    net3 = net.diff(3)
    rows = []
    for y in sorted(set(yr)):
        m = yr == y
        cap = sysd.wind_cap_mw[m].mean()
        rows.append({"year": int(y), "wind_cap_mw": float(cap),
                     "wind_ramp_down_p1_mw": float(wind_d[m].quantile(0.01)),
                     "wind_ramp_up_p99_mw": float(wind_d[m].quantile(0.99)),
                     "wind_ramp_down_p1_pct_cap": float(100 * wind_d[m].quantile(0.01) / cap),
                     "net_load_3h_up_p99_mw": float(net3[m].quantile(0.99)),
                     "net_load_3h_up_max_mw": float(net3[m].max()),
                     "net_load_3h_down_p1_mw": float(net3[m].quantile(0.01))})
    return pd.DataFrame(rows)


def wind_forecast_eval(wf: pd.DataFrame, sysd: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """AESO day-ahead wind power forecast against metered wind output, by year (2012 to 2016 full years)."""
    d = wf.set_index("time_utc").join(sysd[["wind_mwh", "wind_cap_mw"]], how="inner").dropna()
    yr = d.index.tz_convert(TZ).year
    err = d.wpf_ml - d.wind_mwh
    rows = []
    for y in sorted(set(yr)):
        m = yr == y
        if m.sum() < 8000:
            continue
        cap = d.wind_cap_mw[m].mean()
        inside = (d.wind_mwh[m] >= d.wpf_min[m]) & (d.wind_mwh[m] <= d.wpf_max[m])
        rows.append({"year": int(y), "hours": int(m.sum()), "wind_cap_mw": float(cap),
                     "mae_mw": float(err[m].abs().mean()), "mae_pct_cap": float(100 * err[m].abs().mean() / cap),
                     "bias_mw": float(err[m].mean()), "rmse_pct_cap": float(100 * np.sqrt((err[m] ** 2).mean()) / cap),
                     "band_coverage_pct": float(100 * inside.mean()),
                     "below_min_pct": float(100 * (d.wind_mwh[m] < d.wpf_min[m]).mean()),
                     "above_max_pct": float(100 * (d.wind_mwh[m] > d.wpf_max[m]).mean()),
                     "band_width_pct_cap": float(100 * (d.wpf_max[m] - d.wpf_min[m]).mean() / cap)})
    tab = pd.DataFrame(rows)
    # Error by forecast level: where in the range is the forecast least reliable?
    lvl = pd.cut(d.wpf_ml / d.wind_cap_mw, [0, .1, .2, .3, .4, .5, .6, .8, 1.2])
    by_level = d.assign(abs_err=err.abs() / d.wind_cap_mw * 100).groupby(lvl, observed=True)["abs_err"].mean()
    return tab, {"by_forecast_level_pct_cap": {str(k): float(v) for k, v in by_level.items()},
                 "overall_mae_pct_cap": float(100 * (err.abs() / d.wind_cap_mw).mean()),
                 "overall_band_coverage_pct": float(100 * ((d.wind_mwh >= d.wpf_min) & (d.wind_mwh <= d.wpf_max)).mean())}
