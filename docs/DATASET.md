# About the data

All inputs are public. The raw AESO files are not stored in this repository; `scripts/get_data.py` downloads them from aeso.ca and checks each against the SHA-256 checksum recorded in `src/gridanalytics/data.py`.

## Alberta Electric System Operator (AESO) data requests

Listed at https://aeso.ca/market/market-and-system-reporting/data-requests/

| File | Content | Coverage | Used for |
|---|---|---|---|
| Hourly Metered Volumes and Pool Price and AIL (2010 to 2019, 2020 to July 2025) | Hourly metered energy (MWh) for every generating asset over 5 MW, pool price, Alberta Internal Load (AIL), AESO's hour-ahead pool price forecast, imports and exports by intertie. Time stamps in GMT and local time | 1 January 2010 to 31 July 2025, 136,583 hours, 242 assets | Load and price forecasts, wind and solar value, ramps |
| Constrained Volume by Planning Area and Fuel Type | Generation constrained down below its offer (average MW and minutes in each hour) by planning area and fuel | 20 February 2020 to 31 December 2025 | Congestion screen |
| Planning Area Load, Hourly (2023, 2024, 2025) | Load, metered generation and behind-the-fence generation by planning area | 2023 to 2025 | Regional balance (net injection) |
| BC and MATL Intertie Flows | Scheduled flows, available transfer capability (ATC), total transfer capability (TTC) and the reason for any limit | 2013 to October 2020 | Intertie limits |
| Historical long-term wind power forecast | AESO's day-ahead wind forecast: minimum, most likely, maximum | November 2011 to September 2017 | Wind forecast evaluation |
| Historical Transmission Outages | Outage records with equipment, timing and condition | 2013 to 2026 | Kept for context |
| Current Supply/Demand Asset List | Asset name, fuel type and capability (older list) | | Fuel types for assets no longer on the current report |
| Planning Area Boundary Data | Shapefile of the 42 planning areas | 2020 | Map |

AESO notes that metered volumes for April to July 2025 were not yet final when the file was published.

## Other sources

* **AESO Current Supply and Demand report**, read on 25 October 2026 for current fuel types and capabilities (see `static_data/README.md`).
* **ERA5 reanalysis** (Copernicus Climate Change Service) through the Open-Meteo historical weather API: hourly temperature for Calgary and Edmonton, licensed CC BY 4.0.

## Checks

`scripts/check_data.py` writes `outputs/data_quality.json`. On the published run:

* every downloaded file matches its recorded checksum;
* the hourly series has no gaps or duplicate hours (136,583 hours from 2010-01-01 07:00 UTC);
* the pool price stays within its $0 to $1,000 limits (160 hours at the cap, 897 at zero);
* no asset meter reads negative; one gas peaker shows four runs of a constant output over 24 hours, which is plausible for that unit;
* metered generation plus net imports is a steady share of AIL in every month (about 76 percent; the rest is load served by on-site generation and small plants), with no hours flagged as outliers;
* constraint records have no duplicates, negative values or minutes outside 0 to 60;
* a few assets exceed their current listed capability in earlier years because they have since been re-rated; this affects capacity factors only.

## Use

AESO material is used for non-commercial, educational purposes under the AESO website terms. This is an independent project and is not affiliated with or endorsed by the AESO.
