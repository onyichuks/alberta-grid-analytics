"""Wind and solar value and congestion screening on AESO public data.

Usage:  python run_market.py
Writes outputs/market_results.json, outputs/market_dashboard.json and outputs/tables/r*.csv, c*.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from gridanalytics import data as D, renewables as R, congestion as C  # noqa: E402
from gridanalytics.outputs import to_json, rnd  # noqa: E402

TZ = "America/Edmonton"
T = ROOT / "outputs" / "tables"


def records(df: pd.DataFrame, digits: int = 2) -> list[dict]:
    out = []
    for r in df.to_dict("records"):
        out.append({k: (round(v, digits) if isinstance(v, float) and np.isfinite(v) else (None if isinstance(v, float) else v))
                    for k, v in r.items()})
    return out


def area_names() -> dict:
    gj = json.loads((D.STATIC / "planning_areas.geojson").read_text())
    return {int(f["properties"]["Area_ID"]): (f["properties"]["NAME"].replace("  ", " "), f["properties"]["REGION"])
            for f in gj["features"]}


def main() -> None:
    T.mkdir(parents=True, exist_ok=True)
    sysd = D.load_processed("system")
    ah = D.load_processed("assets_hourly")
    at = pd.read_csv(D.STATIC / "assets.csv")
    names = area_names()
    last_local = sysd.index.max().tz_convert(TZ)

    # ------------------------------------------------------------- wind and solar value
    annual = R.annual_summary(sysd)
    annual["partial_year"] = annual.year == last_local.year
    annual.to_csv(T / "r01_annual_summary.csv", index=False)
    cap24 = R.asset_capture(ah, sysd.pool_price, at, 2024)
    cap24.to_csv(T / "r02_asset_capture_2024.csv", index=False)
    dec = {y: R.price_by_wind_decile(sysd, y) for y in (2018, 2024)}
    pd.concat([d.assign(year=y) for y, d in dec.items()]).to_csv(T / "r03_price_by_wind_decile.csv", index=False)
    prof = {y: R.hourly_profile(sysd, y) for y in (2018, 2021, 2024)}
    pd.concat([d.assign(year=y) for y, d in prof.items()]).to_csv(T / "r04_hourly_profile.csv", index=False)
    rmp = R.ramps(sysd)
    rmp.to_csv(T / "r05_ramps.csv", index=False)
    wf_tab, wf_extra = R.wind_forecast_eval(D.load_processed("wind_forecast"), sysd)
    wf_tab.to_csv(T / "r06_aeso_wind_forecast.csv", index=False)

    a = annual.set_index("year")
    pre = a.loc[2019:2021]
    post = a.loc[2023:2025]
    vf = annual[(annual.year >= 2010) & annual.wind_value_factor.notna()]
    slope = float(np.polyfit(vf.wind_share_pct, vf.wind_value_factor, 1)[0])
    ren = {
        "wind_share_2019_pct": a.loc[2019, "wind_share_pct"], "wind_share_2024_pct": a.loc[2024, "wind_share_pct"],
        "solar_share_2024_pct": a.loc[2024, "solar_share_pct"],
        "wind_cap_2019_mw": a.loc[2019, "wind_cap_mw_end"], "wind_cap_2025_mw": a.loc[2025, "wind_cap_mw_end"],
        "solar_cap_2025_mw": a.loc[2025, "solar_cap_mw_end"],
        "wind_vf_2019_2021": float(pre.wind_value_factor.mean()), "wind_vf_2023_2025": float(post.wind_value_factor.mean()),
        "solar_vf_2019_2021": float(pre.solar_value_factor.mean()), "solar_vf_2023_2025": float(post.solar_value_factor.mean()),
        "wind_vf_2024": a.loc[2024, "wind_value_factor"], "solar_vf_2024": a.loc[2024, "solar_value_factor"],
        "storage_vf_2024": a.loc[2024, "storage_value_factor"],
        "wind_capture_2024": a.loc[2024, "wind_capture"], "avg_price_2024": a.loc[2024, "avg_price"],
        "zero_price_hours_2024": int(a.loc[2024, "zero_price_hours"]), "zero_price_hours_2019": int(a.loc[2019, "zero_price_hours"]),
        "vf_slope_per_pct_share": slope,
        "price_low_wind_2024": float(dec[2024].avg_price.iloc[0]), "price_high_wind_2024": float(dec[2024].avg_price.iloc[-1]),
        "wind_vf_spread_2024": [float(cap24[cap24.group == "Wind"].value_factor.min()), float(cap24[cap24.group == "Wind"].value_factor.max())],
        "solar_vf_spread_2024": [float(cap24[cap24.group == "Solar"].value_factor.min()), float(cap24[cap24.group == "Solar"].value_factor.max())],
        "net_ramp_p99_2022_mw": float(rmp.set_index("year").loc[2022, "net_load_3h_up_p99_mw"]),
        "net_ramp_p99_2025_mw": float(rmp.set_index("year").loc[2025, "net_load_3h_up_p99_mw"]),
        "wind_ramp_down_p1_2025_mw": float(rmp.set_index("year").loc[2025, "wind_ramp_down_p1_mw"]),
        "aeso_wind_fc": wf_extra,
    }

    # ------------------------------------------------------------- congestion screening
    c = C.constrained_energy(D.load_processed("constrained"))
    byfuel = C.annual_by_fuel(c)
    byfuel.to_csv(T / "c01_constrained_by_fuel.csv", index=False)
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    curt = C.curtailment_rate(c, sysd, years)
    curt.to_csv(T / "c02_curtailment_rate.csv", index=False)
    pa_tabs = {}
    for y in (2023, 2024, 2025):
        t = C.by_planning_area(c, [y])
        t["name"] = t.planning_area.map(lambda v: names.get(int(v), (str(v), ""))[0])
        t["region"] = t.planning_area.map(lambda v: names.get(int(v), ("", ""))[1])
        pa_tabs[y] = t
    pd.concat([t.assign(year=y) for y, t in pa_tabs.items()]).to_csv(T / "c03_constrained_by_planning_area.csv", index=False)
    cvw = {y: C.constraint_vs_wind(c, sysd, [y]) for y in (2023, 2024)}
    pd.concat([d.assign(year=y) for y, d in cvw.items()]).to_csv(T / "c04_constraints_by_wind_decile.csv", index=False)
    pwc = {y: C.price_when_constrained(c[c.time_utc <= sysd.index.max()], sysd, [y]) for y in (2023, 2024, 2025)}
    monthly = (c.pivot_table(index="month", columns="group", values="mwh", aggfunc="sum", fill_value=0) / 1e3)
    monthly.to_csv(T / "c05_constrained_monthly.csv")
    rb = C.regional_balance(D.load_processed("planning_area_hourly"))
    rb["name"] = rb.planning_area.map(lambda v: names.get(int(v), (str(v), ""))[0])
    rb.to_csv(T / "c06_regional_balance.csv", index=False)
    itl, reasons = C.intertie_limits(D.load_processed("interties"))
    itl.to_csv(T / "c07_intertie_limits.csv", index=False)

    # Screening table: where would locational prices most likely fall below the system price?
    scr = rb[rb.year.isin([2024, 2025])].groupby(["planning_area", "name", "region"]).agg(
        net_mw=("net_mw", "mean"), net_p95_mw=("net_p95_mw", "mean"),
        gen_mw=("gen_mw", "mean"), load_mw=("load_mw", "mean")).reset_index()
    con2 = C.by_planning_area(c, [2024, 2025])[["planning_area", "Total", "Wind", "Solar", "Thermal", "hours"]]
    scr = scr.merge(con2, on="planning_area", how="left").fillna({"Total": 0, "Wind": 0, "Solar": 0, "Thermal": 0, "hours": 0})
    scr["constrained_hours_pct"] = 100 * scr["hours"] / (2 * 8760)
    scr["screen"] = np.where((scr.net_mw > 0) & (scr.constrained_hours_pct >= 5), "Exporting and often constrained",
                    np.where(scr.net_mw > 0, "Exporting", "Importing"))
    scr = scr.sort_values(["Total"], ascending=False)
    scr.to_csv(T / "c08_lmp_screen.csv", index=False)

    b = byfuel.set_index("year")
    cr = curt.set_index(["year", "group"]).curtailment_pct
    top = pa_tabs[2025].head(3)
    cong = {
        "constrained_gwh": {int(y): float(v) for y, v in b.Total.items()},
        "wind_constrained_gwh_2024": float(b.loc[2024, "Wind"]), "wind_constrained_gwh_2025": float(b.loc[2025, "Wind"]),
        "wind_curtail_pct": {int(y): float(cr[(y, "Wind")]) for y in years},
        "solar_curtail_pct": {int(y): float(cr[(y, "Solar")]) for y in years},
        "top_areas_2025": records(top[["planning_area", "name", "Total", "Wind", "Solar", "hours"]], 1),
        "top_area_share_2025_pct": float(100 * pa_tabs[2025].Total.iloc[0] / pa_tabs[2025].Total.sum()),
        "south_share_2025_pct": float(100 * pa_tabs[2025][pa_tabs[2025].region == "South"].Total.sum() / pa_tabs[2025].Total.sum()),
        "price_when_constrained": pwc,
        "constrained_hours_top_decile_2024_pct": float(cvw[2024].hours_constrained_pct.iloc[-1]),
        "constrained_hours_bottom_decile_2024_pct": float(cvw[2024].hours_constrained_pct.iloc[0]),
        "intertie_atc_share_2016_2020_pct": float(itl[itl.year >= 2016].atc_share_pct.mean()),
        "intertie_at_limit_2020_pct": float(itl.set_index("year").loc[2020, "import_at_limit_pct"]),
        "exporting_constrained_areas": scr[scr.screen == "Exporting and often constrained"].name.tolist(),
        "constrained_data_end": str(c.time_utc.max().tz_convert(TZ).date()),
    }

    res = {"renewables": ren, "congestion": cong,
           "annual": records(annual, 3), "ramps": records(rmp, 1), "aeso_wind_forecast": records(wf_tab, 2),
           "constrained_by_fuel": records(byfuel, 2), "curtailment": records(curt, 3),
           "intertie": records(itl, 2), "intertie_reasons": records(reasons), "lmp_screen": records(scr, 1)}
    (ROOT / "outputs" / "market_results.json").write_text(to_json(res, indent=1))

    gj = json.loads((D.STATIC / "planning_areas.geojson").read_text())
    dash = {
        "annual": records(annual, 3), "assets2024": records(cap24, 3),
        "decile": {str(y): records(d, 2) for y, d in dec.items()},
        "profile": {str(y): records(d, 1) for y, d in prof.items()},
        "ramps": records(rmp, 1), "wind_fc": records(wf_tab, 2), "wind_fc_level": wf_extra,
        "byfuel": records(byfuel, 2), "curtail": records(curt, 3),
        "monthly": {"month": monthly.index.tolist(), **{g: rnd(monthly[g], 2) for g in monthly.columns}},
        "pa": {str(y): records(t, 2) for y, t in pa_tabs.items()},
        "cvw": {str(y): records(d, 2) for y, d in cvw.items()},
        "balance": records(rb, 1), "screen": records(scr, 1), "intertie": records(itl, 2),
        "reasons": records(reasons), "geo": gj, "summary": {"renewables": ren, "congestion": cong},
    }
    (ROOT / "outputs" / "market_dashboard.json").write_text(to_json(dash, compact=True))
    print("renewables:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in ren.items() if k != "aeso_wind_fc"})
    print("congestion:", {k: v for k, v in cong.items() if k not in ("price_when_constrained", "top_areas_2025")})
    print("top areas 2025:", cong["top_areas_2025"])
    print("price when constrained:", pwc)


if __name__ == "__main__":
    main()
