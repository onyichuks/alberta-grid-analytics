"""Asset fuel types and capabilities.

Sources, in order of preference:
1. AESO Current Supply and Demand report, captured 25 October 2026 (fuel type, sub type, maximum capability).
2. AESO's older Current Supply/Demand asset list file (CSD-Assets.xlsx).
3. Assets retired before both lists: classified from public knowledge of Alberta's coal fleet, or from
   their output profile (solar output is near zero at night; wind is variable day and night).
"""
from __future__ import annotations

import pandas as pd

from .data import RAW, raw_path, CSD_CURRENT_FILE

# Assets missing from both AESO lists. Basis is recorded for each one.
RETIRED = {
    "BR3": ("COAL", "Battle River 3, coal unit"),
    "KH1": ("COAL", "Keephills 1, coal unit"),
    "SD1": ("COAL", "Sundance 1, coal unit"),
    "SD2": ("COAL", "Sundance 2, coal unit"),
    "SD3": ("COAL", "Sundance 3, coal unit"),
    "SD4": ("COAL", "Sundance 4, coal unit"),
    "SD5": ("COAL", "Sundance 5, coal unit"),
    "WB4": ("COAL", "Wabamun 4, coal unit"),
    "CRW1": ("WIND", "output profile: variable day and night, capacity factor 15 percent"),
    "TAY2": ("WIND", "output profile: variable day and night, 3 MW"),
    "CHP1": ("SOLAR", "output profile: near zero at night"),
    "MCH1": ("SOLAR", "output profile: near zero at night"),
    "NPC1": ("GAS", "same site as the NPC2 and NPC3 gas units; daytime peaking profile"),
    "RB2": ("GAS", "Rainbow gas unit"),
    "DV1": ("BIOMASS", "steady baseload profile, about 11 MW"),
    "HSM1": ("BIOMASS", "steady profile, about 6 MW"),
    "PW01": ("BIOMASS", "steady profile, about 6 MW"),
    "MFG1": ("BIOMASS", "about 14 MW, 2015 to 2016 only"),
}

# Grouping used across the analysis.
GROUP = {"WIND": "Wind", "SOLAR": "Solar", "HYDRO": "Hydro", "ENERGY STORAGE": "Storage",
         "GAS": "Thermal", "COAL": "Thermal", "DUAL FUEL": "Thermal",
         "BIOMASS": "Other", "OTHER": "Other"}


def asset_table(asset_ids: list[str], metered: pd.DataFrame | None = None) -> pd.DataFrame:
    cur = pd.read_csv(RAW / CSD_CURRENT_FILE)
    old = pd.read_excel(raw_path("csd_assets_old"))
    rows = []
    for a in asset_ids:
        c = cur[cur.asset_id == a]
        o = old[old.ASSET_SHORT_NAME == a]
        if len(c):
            r = c.iloc[0]
            rows.append((a, r.asset_name, r.fuel_type, r.sub_fuel_type, float(r.maximum_capability_mw),
                         "AESO Current Supply and Demand report, 25 October 2026"))
        elif len(o):
            r = o.iloc[0]
            fuel = "BIOMASS" if r.FUEL_TYPE == "OTHER" else r.FUEL_TYPE
            rows.append((a, r.ASSET_NAME, fuel, r.SUB_FUEL_TYPE, float(r.MAXIMUM_CAPABILITY),
                         "AESO CSD-Assets.xlsx"))
        elif a in RETIRED:
            fuel, basis = RETIRED[a]
            rows.append((a, a, fuel, fuel, None, "Inferred: " + basis))
        else:
            rows.append((a, a, "UNKNOWN", "UNKNOWN", None, "No output and no listing"))
    t = pd.DataFrame(rows, columns=["asset_id", "asset_name", "fuel_type", "sub_fuel_type",
                                    "maximum_capability_mw", "source"])
    t["group"] = t.fuel_type.map(GROUP).fillna("Other")
    if metered is not None:
        m = metered[asset_ids]
        t["max_hourly_mwh"] = m.max().reindex(t.asset_id).to_numpy()
        prod = m.gt(0.1)
        t["first_output_utc"] = [m.index[prod[a]].min() if prod[a].any() else pd.NaT for a in t.asset_id]
        t["last_output_utc"] = [m.index[prod[a]].max() if prod[a].any() else pd.NaT for a in t.asset_id]
        # Capacity used for capacity factors: listed capability, or the largest hourly output for retired assets.
        t["capacity_mw"] = t.maximum_capability_mw.fillna(t.max_hourly_mwh)
    return t
