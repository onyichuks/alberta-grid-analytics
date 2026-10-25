"""Turn the raw AESO files into processed tables (data/processed) and small static files (static_data).

Usage:  python scripts/prepare_data.py
Needs the raw files in data/raw (see scripts/get_data.py and docs/DATASET.md).
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile  # pyshp
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridanalytics import data as D  # noqa: E402
from gridanalytics.assets import asset_table  # noqa: E402


def capacity_series(assets: pd.DataFrame, idx: pd.DatetimeIndex, group: str) -> pd.Series:
    """Installed capacity of a group over time: each asset counts from its first to its last metered output."""
    cap = np.zeros(len(idx))
    still_running = idx.max() - pd.Timedelta(days=14)
    for _, r in assets[assets.group == group].iterrows():
        if pd.isna(r.first_output_utc) or not np.isfinite(r.capacity_mw):
            continue
        last = idx.max() if r.last_output_utc >= still_running else r.last_output_utc
        on = (idx >= r.first_output_utc) & (idx <= last)
        cap[on] += r.capacity_mw
    return pd.Series(cap, index=idx)


def planning_area_geojson() -> dict:
    """AESO planning area polygons, reprojected to longitude and latitude, simplified for the web."""
    with zipfile.ZipFile(D.raw_path("planning_areas")) as z:
        prj = z.read("AESO_Planning_Areas.prj").decode()
        shp = z.open("AESO_Planning_Areas.shp")
        shx = z.open("AESO_Planning_Areas.shx")
        dbf = z.open("AESO_Planning_Areas.dbf")
        r = shapefile.Reader(shp=shp, shx=shx, dbf=dbf)
        fields = [f[0] for f in r.fields[1:]]
        tr = Transformer.from_crs(prj, "EPSG:4326", always_xy=True)
        feats = []
        for sr in r.iterShapeRecords():
            props = dict(zip(fields, sr.record))
            pts = sr.shape.points
            parts = list(sr.shape.parts) + [len(pts)]
            rings = []
            for a, b in zip(parts[:-1], parts[1:]):
                xs, ys = tr.transform([p[0] for p in pts[a:b]], [p[1] for p in pts[a:b]])
                rings.append([[round(x, 4), round(y, 4)] for x, y in zip(xs, ys)])
            feats.append({"type": "Feature", "properties": props,
                          "geometry": {"type": "Polygon" if len(rings) == 1 else "MultiPolygon",
                                       "coordinates": rings if len(rings) == 1 else [[ring] for ring in rings]}})
    return {"type": "FeatureCollection", "features": feats}


def main() -> None:
    D.PROCESSED.mkdir(parents=True, exist_ok=True)
    D.STATIC.mkdir(exist_ok=True)

    met = D.read_metered()
    asset_ids = [c for c in met.columns if c not in D.SYSTEM_COLS]
    assets = asset_table(asset_ids, met)
    assets.to_csv(D.STATIC / "assets.csv", index=False)
    print("assets:", assets.group.value_counts().to_dict())

    sysd = met[D.SYSTEM_COLS].rename(columns={
        "ACTUAL_POOL_PRICE": "pool_price", "ACTUAL_AIL": "ail_mw",
        "HOUR_AHEAD_POOL_PRICE_FORECAST": "aeso_ha_price_fc",
        "EXPORT_BC": "export_bc", "EXPORT_MT": "export_mt", "EXPORT_SK": "export_sk",
        "IMPORT_BC": "import_bc", "IMPORT_MT": "import_mt", "IMPORT_SK": "import_sk"}).copy()
    for g in ["Wind", "Solar", "Hydro", "Storage", "Thermal", "Other"]:
        ids = assets.asset_id[assets.group == g].tolist()
        sysd[g.lower() + "_mwh"] = met[ids].sum(axis=1, min_count=1).fillna(0.0)
    for g in ["Wind", "Solar", "Storage"]:
        sysd[g.lower() + "_cap_mw"] = capacity_series(assets, met.index, g)
    sysd["net_import_mw"] = (sysd[["import_bc", "import_mt", "import_sk"]].sum(axis=1)
                             - sysd[["export_bc", "export_mt", "export_sk"]].sum(axis=1))
    w = D.read_weather()
    sysd = sysd.join(w, how="left")
    sysd.to_parquet(D.PROCESSED / "system.parquet")
    met[asset_ids].astype("float32").to_parquet(D.PROCESSED / "assets_hourly.parquet")
    print("system:", sysd.shape, sysd.index.min(), sysd.index.max())

    D.read_constrained().to_parquet(D.PROCESSED / "constrained.parquet", index=False)
    pa = D.read_planning_area_load()
    pa.to_parquet(D.PROCESSED / "planning_area_hourly.parquet", index=False)
    D.read_interties().to_parquet(D.PROCESSED / "interties.parquet", index=False)
    D.read_wind_forecast().to_parquet(D.PROCESSED / "wind_forecast.parquet", index=False)
    o = D.read_outages()
    o.astype({c: "string" for c in o.columns if o[c].dtype == object}).to_parquet(
        D.PROCESSED / "outages.parquet", index=False)

    gj = planning_area_geojson()
    (D.STATIC / "planning_areas.geojson").write_text(json.dumps(gj, separators=(",", ":")))
    print("planning areas:", len(gj["features"]), "fields:", list(gj["features"][0]["properties"]))


if __name__ == "__main__":
    main()
