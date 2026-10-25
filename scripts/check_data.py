"""Data quality report: checksums, continuity, price limits, supply balance, asset meters, constraints.

Usage:  python scripts/check_data.py      (writes outputs/data_quality.json)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridanalytics import data as D, quality as Q  # noqa: E402
from gridanalytics.outputs import to_json  # noqa: E402


def main() -> None:
    files = {}
    for key, (name, url, expected) in D.FILES.items():
        p = D.RAW / name
        got = Q.sha256(p) if p.exists() else None
        files[name] = {"url": url, "sha256_expected": expected, "sha256": got, "match": got == expected}
    sysd = D.load_processed("system")
    ah = D.load_processed("assets_hourly")
    at = pd.read_csv(D.STATIC / "assets.csv", parse_dates=["first_output_utc", "last_output_utc"])
    c = D.load_processed("constrained")
    wf = D.load_processed("wind_forecast")
    prov = sysd.index >= pd.Timestamp("2025-04-01", tz="America/Edmonton")
    rep = {
        "files": files,
        "continuity": Q.hourly_continuity(sysd.index),
        "price": Q.price_checks(sysd.pool_price),
        "hour_ahead_forecast_missing": int(sysd.aeso_ha_price_fc.isna().sum()),
        "ail": {"min": float(sysd.ail_mw.min()), "max": float(sysd.ail_mw.max()),
                "largest_hourly_change_mw": float(sysd.ail_mw.diff().abs().max()),
                "stuck_runs_24h": Q.stuck_values(sysd.ail_mw)},
        "supply_balance": Q.supply_balance(sysd),
        "assets": Q.asset_checks(ah, at),
        "weather_missing": int(sysd[["calgary_temp_c", "edmonton_temp_c"]].isna().sum().sum()),
        "provisional_hours_apr_jul_2025": int(prov.sum()),
        "constrained": Q.constrained_checks(c),
        "wind_forecast": Q.wind_forecast_checks(wf),
    }
    out = ROOT / "outputs" / "data_quality.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(to_json(rep, indent=1))
    bad = [n for n, f in files.items() if not f["match"]]
    print("checksums:", "all match" if not bad else f"MISMATCH {bad}")
    print("continuity:", rep["continuity"])
    print("price:", rep["price"])
    print("supply balance flagged hours:", rep["supply_balance"]["flagged_hours"])
    print("assets:", {k: v for k, v in rep["assets"].items() if k != "no_output_assets"})
    print("constrained:", rep["constrained"])
    print("wind forecast:", rep["wind_forecast"])


if __name__ == "__main__":
    main()
