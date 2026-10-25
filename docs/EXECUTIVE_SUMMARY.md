# Executive summary

**Alberta grid analytics: probabilistic load and price forecasts, the value of wind and solar, and a congestion screen ahead of locational pricing**

Onyi Chuks, Calgary, Alberta. October 2026. [Interactive dashboard](https://onyichuks.github.io/alberta-grid-analytics/) · [Full report](REPORT.md) · [Machine learning note](ML_TECHNICAL_NOTE.md) · [Code](https://github.com/onyichuks/alberta-grid-analytics)

## The question

Alberta's grid is changing fast. Wind and solar have grown from 5 to 20 percent of load since 2019, and the market is moving to locational pricing. Operators need forecasts that state their uncertainty honestly, an understanding of what renewable growth does to prices and flexibility, and an early view of where transmission limits bind. This project answers those questions from 15 years of hourly public AESO data (2010 to July 2025), with every result tested on years the models never saw.

## What was found

* **Day-ahead load can be forecast to within 1.5 percent.** Across 2023 to July 2025 the median forecast was off by 1.48 percent on average (150 MW), half the error of the same-hour-last-week benchmark. The first model drifted 148 MW low as Alberta's load grew past its training years. Forecasting load relative to its recent level and retraining each year removed the drift.
* **The ranges are calibrated.** Raw quantile models put only 63 percent of hours inside their 80 percent range. Calibrating each year on the year before (conformal prediction) brought that to **81 percent**.
* **A probabilistic layer on AESO's hour-ahead price forecast cuts its error by 18 percent.** The mean error was $13.11 against $15.96/MWh, and the quantile loss was 34 percent lower in 2024 and 2025.
* **The spike warning has 69 percent Brier skill.** It warns of prices at or above $200/MWh, catching 76 percent of spikes, and 20 percent of its warnings were false alarms. A simple flag on AESO's forecast had 31 percent false alarms.
* **Wind is cannibalizing its own price.** Wind's value factor fell from 0.71 (2019 to 2021) to **0.55** (2023 to 2025) as its share of load rose from 4.8 to 14.4 percent. In 2024 the pool price averaged $192 in the least windy hours and $12 in the windiest. Zero-price hours went from none in 2019 to 386 in 2024.
* **Solar's premium is gone, and batteries earn the most per MWh.** Solar's value factor fell from 1.27 to 0.83. Batteries earned 5.5 times the average price in 2024 by discharging into the scarcest hours.
* **Generation held back by transmission limits rose tenfold.** It went from 116 GWh in 2020 to **1,332 GWh** in 2025, mostly wind. Medicine Hat alone was 63 percent of the 2025 total and the South 77 percent. Wind or solar was held back in 83 percent of the windiest hours of 2024.
* **Flexibility needs are growing.** The 99th percentile three-hour rise in net load (load less wind and solar) grew from 1,099 MW in 2022 to 1,548 MW in 2025.

## What it means

1. **Run forecasts the way they will be used.** Retrain on a schedule, forecast relative to a moving level, and recalibrate ranges every year. A single test year hid a drift that reached 235 MW.
2. **Build on AESO's own forecasts.** The gain comes from a calibrated range and spike probability on top of AESO's point forecast, not from replacing it.
3. **Expect locational prices to separate first in Medicine Hat, Hanna and Sheerness.** These areas export far more than they use and are often constrained; generation sited there would earn below the system price, and new load there would pay less.
4. **Value flexibility.** Storage, demand response and fast-ramping supply are where the next unit of value sits as wind and solar keep growing.

## How it was done

The forecasts use gradient-boosted quantile models (scikit-learn) with yearly retraining and conformalized quantile regression. They are scored with the pinball loss, interval coverage and the Brier score.

Wind and solar value uses capture prices and value factors for every asset over 5 MW. The congestion screen combines AESO's constrained volumes by planning area with each area's net injection.

The work is a tested Python package (24 unit tests; 81 automated dashboard checks) with a research report and an interactive dashboard. A Model Context Protocol server lets an AI assistant answer grid questions from the published tables, citing its source for every number.

## Limits

* Temperature inputs are reanalysis values standing in for weather forecasts, so load scores are somewhat optimistic.
* The price model depends on AESO's hour-ahead forecast.
* The congestion screen ranks exposure but does not estimate nodal prices, which needs a network model.

Data: Alberta Electric System Operator public data; ERA5 through Open-Meteo. Independent project; not affiliated with or endorsed by the AESO.
