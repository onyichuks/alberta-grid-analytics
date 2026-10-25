# Static data

Small files kept in the repository so the analysis can be rebuilt without the steps that are not a single download.

| File | What it is | How it was made |
|---|---|---|
| `csd_current_assets_2026-10-25.csv` | Asset ID, name, maximum capability (MW), fuel type and sub type for the 222 assets on AESO's Current Supply and Demand report | Read from the public report at http://ets.aeso.ca/ets_web/ip/Market/Reports/CSDReportServlet on 25 October 2026 (report time stamp Oct 25, 2026 15:07) |
| `open-meteo-calgary-edmonton-utc.csv` | Hourly 2 m air temperature (°C) for Calgary (51.05 N, 114.07 W) and Edmonton (53.55 N, 113.49 W), 31 December 2009 to 1 August 2025, UTC | Open-Meteo historical weather API, ERA5 model (`models=era5`, `timezone=GMT`); 136,632 hours, no missing values |
| `assets.csv` | Fuel type, group, capacity and first and last metered output for every asset in the hourly file | Written by `scripts/prepare_data.py` from the two AESO asset lists; 26 retired assets missing from both lists are classified in `src/gridanalytics/assets.py`, with the basis for each |
| `planning_areas.geojson` | AESO planning area boundaries (42 areas), reprojected to longitude and latitude | Written by `scripts/prepare_data.py` from `AESO-Planning-Areas-2020-06-23.zip` |

To use the first two files, copy them into `data/raw/`.

Sources: Alberta Electric System Operator (AESO); ERA5 reanalysis, Copernicus Climate Change Service, through Open-Meteo (CC BY 4.0). AESO material is used for non-commercial, educational purposes under the AESO website terms; the raw AESO files are not redistributed here and should be downloaded from aeso.ca.
