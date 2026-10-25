"""Grid questions an AI agent can answer from the analysis outputs.

Each function reads the published result files (outputs/*.json), returns plain data with the source table
it came from, and never invents a number. mcp_server.py exposes these functions as tools over the Model
Context Protocol, so an AI assistant can answer questions like "how much wind was constrained
near Medicine Hat in 2025?" or "what did the forecast say for 6 pm on 13 January 2024?".
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

OUT = Path(__file__).resolve().parents[2] / "outputs"
TZ = ZoneInfo("America/Edmonton")


@lru_cache(maxsize=None)
def _load(name: str) -> dict:
    return json.loads((OUT / name).read_text())


def _area_name(area: int) -> str:
    for f in _load("market_dashboard.json")["geo"]["features"]:
        if int(f["properties"]["Area_ID"]) == int(area):
            return " ".join(f["properties"]["NAME"].split())
    return str(area)


def _find_area(planning_area: str | int) -> int:
    """Accept an AESO planning area number or a name such as 'Medicine Hat'."""
    try:
        return int(planning_area)
    except (TypeError, ValueError):
        q = str(planning_area).strip().lower()
        for f in _load("market_dashboard.json")["geo"]["features"]:
            if q in " ".join(f["properties"]["NAME"].split()).lower():
                return int(f["properties"]["Area_ID"])
    raise ValueError(f"Unknown planning area: {planning_area}")


def grid_summary() -> dict:
    """Headline findings: forecast skill, wind and solar value, and congestion."""
    fr, mk = _load("forecast_results.json"), _load("market_results.json")
    return {"load_forecast": fr["load"]["headline"], "price_forecast": fr["price"]["headline"],
            "spike_warning": {k: v for k, v in fr["price"]["spike"].items() if k != "reliability"},
            "renewables": {k: v for k, v in mk["renewables"].items() if k != "aeso_wind_fc"},
            "congestion": {k: v for k, v in mk["congestion"].items() if k != "price_when_constrained"},
            "source": "outputs/forecast_results.json and outputs/market_results.json"}


def forecast_scores(target: str = "load", period: str = "all test") -> dict:
    """Scores of every forecasting method for 'load' or 'price', for 'all test' or a single year."""
    if target not in ("load", "price"):
        raise ValueError("target must be 'load' or 'price'")
    rows = [r for r in _load("forecast_results.json")[target]["scores"] if r["period"] == str(period)]
    if not rows:
        raise ValueError(f"No scores for period {period}")
    return {"target": target, "period": period, "scores": rows, "source": f"outputs/tables/f0{1 if target == 'load' else 2}_{target}_scores.csv"}


def forecast_at(local_time: str) -> dict:
    """Actual and forecast load and price for one hour, given as Alberta local time 'YYYY-MM-DD HH:00'.

    Load covers 2024 to July 2025 and price 2024 to July 2025 (the dashboard test period).
    """
    dt = datetime.strptime(local_time.strip()[:13], "%Y-%m-%d %H").replace(tzinfo=TZ)
    t_utc = dt.astimezone(timezone.utc)
    fc = _load("forecast_dashboard.json")
    out = {"local_time": dt.strftime("%Y-%m-%d %H:00 %Z")}
    for key in ("load", "price"):
        s = fc[key]
        t0 = datetime.strptime(s["t0_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        i = int((t_utc - t0) / timedelta(hours=1))
        if s.get("offsets"):
            i = s["offsets"].index(i) if i in s["offsets"] else -1
        if not (0 <= i < len(s["actual"])):
            out[key] = None
            continue
        rec = {"actual": s["actual"][i], **{q: s[q][i] for q in ("q10", "q25", "q50", "q75", "q90")}}
        if key == "price":
            rec["aeso_hour_ahead"] = s["aeso"][i]
            rec["spike_probability"] = s["p_spike"][i]
        out[key] = rec
    out["units"] = {"load": "MW", "price": "$/MWh"}
    return out


def constrained_energy(year: int, planning_area: str | int | None = None) -> dict:
    """Generation held back by transmission limits (GWh) by fuel, for one planning area or the top ten."""
    pa = _load("market_dashboard.json")["pa"].get(str(year))
    if pa is None:
        raise ValueError("year must be 2023, 2024 or 2025")
    if planning_area is None:
        rows = [{"planning_area": r["planning_area"], "name": _area_name(r["planning_area"]), "total_gwh": r["Total"],
                 "wind_gwh": r.get("Wind", 0), "solar_gwh": r.get("Solar", 0), "hours": r["hours"]} for r in pa[:10]]
        return {"year": year, "top_areas": rows, "total_gwh": sum(r["Total"] for r in pa),
                "source": "outputs/tables/c03_constrained_by_planning_area.csv"}
    a = _find_area(planning_area)
    r = next((x for x in pa if int(x["planning_area"]) == a), None)
    return {"year": year, "planning_area": a, "name": _area_name(a),
            "total_gwh": r["Total"] if r else 0.0, "wind_gwh": r.get("Wind", 0) if r else 0.0,
            "solar_gwh": r.get("Solar", 0) if r else 0.0, "thermal_gwh": r.get("Thermal", 0) if r else 0.0,
            "hours_with_constraint": r["hours"] if r else 0,
            "source": "outputs/tables/c03_constrained_by_planning_area.csv"}


def area_balance(planning_area: str | int, year: int = 2024) -> dict:
    """Average generation, load and net injection (MW) for a planning area, plus its locational pricing screen."""
    a = _find_area(planning_area)
    mk = _load("market_dashboard.json")
    bal = next((r for r in mk["balance"] if int(r["planning_area"]) == a and int(r["year"]) == int(year)), None)
    scr = next((r for r in mk["screen"] if int(r["planning_area"]) == a), None)
    return {"planning_area": a, "name": _area_name(a), "year": year, "balance": bal, "screen_2024_2025": scr,
            "source": "outputs/tables/c06_regional_balance.csv and c08_lmp_screen.csv"}


def asset_value(asset: str, year: int = 2024) -> dict:
    """Capture price, value factor and capacity factor for a wind, solar or storage asset (2024)."""
    if int(year) != 2024:
        raise ValueError("Per-asset results are published for 2024")
    q = asset.strip().lower()
    for r in _load("market_dashboard.json")["assets2024"]:
        if q == r["asset_id"].lower() or q in r["asset_name"].lower():
            return {**r, "year": 2024, "source": "outputs/tables/r02_asset_capture_2024.csv"}
    raise ValueError(f"No 2024 result for asset {asset}")


def annual(year: int) -> dict:
    """Annual system summary: average price, load, wind and solar share, capture prices and value factors."""
    for r in _load("market_results.json")["annual"]:
        if int(r["year"]) == int(year):
            return {**r, "source": "outputs/tables/r01_annual_summary.csv"}
    raise ValueError("year must be between 2010 and 2025")


TOOLS = [grid_summary, forecast_scores, forecast_at, constrained_energy, area_balance, asset_value, annual]
