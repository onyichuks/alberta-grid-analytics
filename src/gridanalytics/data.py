"""Reading the public AESO files into tidy tables.

All timestamps are kept in UTC (the AESO files carry a GMT column), so daylight saving
changes never create duplicate or missing hours. Local time is derived only for calendar
features, using America/Edmonton.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
STATIC = ROOT / "static_data"
TZ = "America/Edmonton"

AESO = "https://aeso.ca/assets/Uploads/"
FILES = {
    "metered_2010_2019": ("Hourly_Metered_Volumes_and_Pool_Price_and_AIL_2010-2019.csv",
                          AESO + "data-requests/Hourly_Metered_Volumes_and_Pool_Price_and_AIL_2010-2019.csv",
                          "52fbff7fc325aad40043ce997fe79dc58f08af8a3b398a2564129b4a07dd0f50"),
    "metered_2020_2025": ("Hourly_Metered_Volumes_and_Pool_Price_and_AIL_2020-Jul2025.csv",
                          AESO + "data-requests/Hourly_Metered_Volumes_and_Pool_Price_and_AIL_2020-Jul2025.csv",
                          "6fbd272eb2db3638e86b22fa7965474cbcb91a946257abef232529db0c5f62da"),
    "constrained": ("Constrained-MW-by-Planning-Area-and-Fuel-Type.csv",
                    AESO + "data-requests/Constrained-MW-by-Planning-Area-and-Fuel-Type.csv",
                    "dad7cc4b31fdcd1d6420b5fe1dbbbb778027256325873fb197f9de0b96621351"),
    "csd_assets_old": ("CSD-Assets.xlsx", AESO + "CSD-Assets.xlsx",
                       "757768faa20277a8b75a91fb6ab1d129826c5230e1843052a868b03976fa779d"),
    "planning_areas": ("AESO-Planning-Areas-2020-06-23.zip", AESO + "AESO-Planning-Areas-2020-06-23.zip",
                       "613b7aafdaf4c67468429148c8c0e563ddfe21b4be3c47dc47dd30e1e5aa5ca7"),
    "interties": ("BC-MATL-Flows-Scheduled-Flows.xlsx", AESO + "BC-MATL-Flows-Scheduled-Flows.xlsx",
                  "c418e84b48ff4aceb573c770733c6c3572c7220373cd72812b9d20d8451da96f"),
    "outages": ("Transmission-Outages.xlsx", AESO + "data-requests/Transmission-Outages.xlsx",
                "ee4eb22f564f587d80f851b82df4460af19a3bbd971864018aa66d34f731f3f0"),
    "wind_forecast": ("WPF-Day-Ahead-Forecast.xlsx", AESO + "WPF-Day-Ahead-Forecast.xlsx",
                      "382800940a2c6aec75d64cbfcacceef37fd468c3a99c086dacb4696c74115562"),
    "pa_load_2023": ("Planning-Area-Load-Hourly-2023.zip",
                     AESO + "data-requests/Planning-Area-Load/Planning-Area-Load-Hourly-2023.zip",
                     "8da4826725294ac9a67702f4da307d1d117339a21e12852e03a3f799a9918e77"),
    "pa_load_2024": ("Planning-Area-Load-Hourly-2024.zip",
                     AESO + "data-requests/Planning-Area-Load/Planning-Area-Load-Hourly-2024.zip",
                     "8a9f72e7ffe3955bee08d90ee579a77a2a7a0bd8aac66a8812033899b4829225"),
    "pa_load_2025": ("Planning-Area-Load-Hourly-2025.zip",
                     AESO + "data-requests/Planning-Area-Load/Planning-Area-Load-Hourly-2025.zip",
                     "f650d7449a8c93214545925af4c81cc1561df7503322fe40ec61307332b56df8"),
}
# Built from public pages rather than a single download (see static_data/README.md).
WEATHER_FILE = "open-meteo-calgary-edmonton-utc.csv"
CSD_CURRENT_FILE = "csd_current_assets_2026-10-25.csv"

SYSTEM_COLS = ["ACTUAL_POOL_PRICE", "ACTUAL_AIL", "HOUR_AHEAD_POOL_PRICE_FORECAST",
               "EXPORT_BC", "EXPORT_MT", "EXPORT_SK", "IMPORT_BC", "IMPORT_MT", "IMPORT_SK"]


def raw_path(key: str) -> Path:
    return RAW / FILES[key][0]


def _parse_gmt(s: pd.Series) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(s, format="mixed")).tz_localize("UTC")


def read_metered() -> pd.DataFrame:
    """Hourly metered volumes (MWh) by asset plus pool price, AIL and intertie flows, 2010 to July 2025.

    Index: hour beginning, UTC. Columns: one per asset, then SYSTEM_COLS.
    """
    parts = []
    for key in ("metered_2010_2019", "metered_2020_2025"):
        d = pd.read_csv(raw_path(key), low_memory=False)
        idx = _parse_gmt(d.pop("Date_Begin_GMT"))
        d = d.drop(columns=["Date_Begin_Local"])
        d.index = idx
        parts.append(d.apply(pd.to_numeric, errors="coerce"))
    df = pd.concat(parts, axis=0, sort=False)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = "time_utc"
    assets = [c for c in df.columns if c not in SYSTEM_COLS]
    return df[assets + SYSTEM_COLS]


def read_weather() -> pd.DataFrame:
    w = pd.read_csv(RAW / WEATHER_FILE)
    w.index = pd.DatetimeIndex(pd.to_datetime(w.pop("time_utc"))).tz_localize("UTC")
    w.index.name = "time_utc"
    return w


def read_constrained() -> pd.DataFrame:
    """Constrained-down generation (average MW over the hour) by planning area and fuel type."""
    c = pd.read_csv(raw_path("constrained"))
    c["time_utc"] = pd.to_datetime(c["Date (GMT)"]).dt.tz_localize("UTC")
    c = c.rename(columns={"Fuel Type": "fuel", "Planning Area": "planning_area",
                          "Constrained_MW": "constrained_mw", "Minutes": "minutes"})
    return c[["time_utc", "fuel", "planning_area", "constrained_mw", "minutes"]]


def read_planning_area_load(years=(2023, 2024, 2025)) -> pd.DataFrame:
    """Hourly load and generation by planning area. DT_MST is Mountain Standard Time (no daylight saving)."""
    frames = []
    for y in years:
        p = raw_path(f"pa_load_{y}")
        with zipfile.ZipFile(p) as z:
            with z.open(z.namelist()[0]) as f:
                d = pd.read_csv(f, usecols=["DT_MST", "REGION", "PLANNING_AREA", "Load", "SysGen",
                                            "CSDGen", "BTFGen", "ActLoad"])
        d["time_utc"] = (pd.to_datetime(d.pop("DT_MST")) + pd.Timedelta(hours=7)).dt.tz_localize("UTC")
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"REGION": "region", "PLANNING_AREA": "planning_area", "Load": "load",
                              "SysGen": "sys_gen", "CSDGen": "csd_gen", "BTFGen": "btf_gen", "ActLoad": "act_load"})
    return out


def read_interties() -> pd.DataFrame:
    """BC and Montana (MATL) scheduled flows, available transfer capability (ATC) and total transfer capability (TTC)."""
    d = pd.read_excel(raw_path("interties"))
    # FLOW_DT is hour beginning in Mountain prevailing time; keep it as local and derive UTC with DST handling.
    local = pd.to_datetime(d["FLOW_DT"])
    d["time_local"] = local
    d = d.rename(columns={"TRANSFER_TYPE": "path", "SCHED_FLOW_T": "sched_mw", "ATC_T": "atc_mw",
                          "TTC_T": "ttc_mw", "COMMENTS_T": "limit_reason"})
    return d[["time_local", "path", "sched_mw", "atc_mw", "ttc_mw", "limit_reason"]]


def read_outages() -> pd.DataFrame:
    d = pd.read_excel(raw_path("outages"))
    d.columns = [c.strip().lower().replace(" ", "_") for c in d.columns]
    return d


def read_wind_forecast() -> pd.DataFrame:
    """AESO day-ahead wind power forecast (minimum, most likely, maximum), Nov 2011 to Sep 2017."""
    d = pd.read_excel(raw_path("wind_forecast"))
    d.columns = ["issued_utc", "issued_local", "time_utc", "time_local", "wpf_min", "wpf_ml", "wpf_max"]
    d["time_utc"] = pd.to_datetime(d["time_utc"]).dt.tz_localize("UTC")
    d["issued_utc"] = pd.to_datetime(d["issued_utc"]).dt.tz_localize("UTC")
    return d[["issued_utc", "time_utc", "wpf_min", "wpf_ml", "wpf_max"]]


def local_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.tz_convert(TZ)


def load_processed(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"{name}.parquet")


def numeric_hours(idx: pd.DatetimeIndex) -> np.ndarray:
    return ((idx - idx[0]) / pd.Timedelta(hours=1)).to_numpy()
