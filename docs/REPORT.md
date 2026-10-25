# Alberta grid analytics: probabilistic forecasting, the value of wind and solar, and a congestion screen ahead of locational pricing

Onyi Chuks, Calgary, Alberta. October 2026. [Interactive dashboard](https://onyichuks.github.io/alberta-grid-analytics/) · [Executive summary](EXECUTIVE_SUMMARY.md) · [Machine learning note](ML_TECHNICAL_NOTE.md) · [Data](DATASET.md)

## Abstract

Fifteen years of hourly public data from the Alberta Electric System Operator (AESO) are used for three grid analytics problems.

**Load forecasting.** A day-ahead load forecast built from gradient-boosted quantile models has a mean absolute percentage error of **1.48 percent** across 2023 to July 2025, half that of a same-hour-last-week benchmark (2.99 percent). Two design choices mattered:
* forecasting load relative to its recent level, and retraining each year, removed a drift of 148 MW that appeared as Alberta's load grew beyond the training years;
* conformal calibration on the previous year raised the share of hours inside the 80 percent range from 63 to **81 percent**.

**Price forecasting.** An hour-ahead pool price model that uses AESO's own hour-ahead forecast as an input cuts its mean absolute error by **18 percent** ($13.11 against $15.96/MWh) and its quantile loss by 34 percent in 2024 and 2025. A companion classifier warns of prices at or above $200/MWh with 69 percent Brier skill.

**Wind and solar value.** Wind's value factor (capture price over average price) fell from 0.71 in 2019 to 2021 to **0.55** in 2023 to 2025 as wind's share of load rose from 4.8 to 14.4 percent.

**Congestion.** Generation held back by transmission limits rose from 116 GWh in 2020 to **1,332 GWh** in 2025, 77 percent of it in the South and 63 percent in the Medicine Hat planning area alone. Areas that export and are often constrained are flagged as where locational prices would most likely fall below the system price.

**AI agent access.** The results are also exposed through a Model Context Protocol server, so an AI assistant can answer grid questions from the published tables.

## 1. Objectives

1. Build operational forecasts of load and price that state their uncertainty honestly, and test them on years the models never saw.
2. Measure how the value of wind and solar changes as their share of the grid grows, and what that means for flexibility.
3. Screen where transmission limits already hold generation back, as a first view of where locational marginal pricing would separate prices.
4. Make the results usable by people and by AI agents, with every number traceable to code and public data.

## 2. Data

All inputs are public; see [DATASET.md](DATASET.md) for files, coverage and checksums. The core table is AESO's hourly metered volumes for every asset over 5 MW, with pool price, Alberta Internal Load (AIL), AESO's hour-ahead price forecast and intertie flows. It runs from 1 January 2010 to 31 July 2025: 136,583 hours and 242 assets.

The supporting files are:
* constrained-down generation by planning area and fuel (2020 to 2025);
* hourly load and generation by planning area (2023 to 2025);
* BC and Montana intertie capability (2013 to 2020);
* AESO's day-ahead wind forecast (2011 to 2017);
* the 42 planning area boundaries.

Fuel types come from AESO's Current Supply and Demand report, captured on 25 October 2026. The 26 retired assets missing from both AESO lists are classified from public knowledge of the coal fleet or from their output profile, with the basis recorded. Hourly temperature for Calgary and Edmonton comes from the ERA5 reanalysis (Hersbach et al., 2020) through Open-Meteo.

## 3. Methods

### 3.1 Data checks

Each file is checked against a recorded SHA-256 checksum. The hourly series is checked for:
* gaps and duplicates, with all times kept in UTC so daylight saving creates neither;
* prices outside the $0 to $1,000 limits;
* negative meter readings;
* runs of identical values;
* output above listed capability.

A supply balance check tracks metered generation plus net imports as a share of AIL each month and flags hours that leave the month's usual share by more than six robust standard deviations.

### 3.2 Day-ahead load forecast

The forecast is treated as issued on the morning of the day before, so every load input is at least 48 hours old. The inputs are:
* hour, day of week, month, day of year and Alberta statutory holidays;
* Calgary and Edmonton temperature for the hour, its daily mean, and its value 24 hours earlier;
* load 48, 168 and 336 hours earlier and the daily mean 48 hours earlier.

Five gradient-boosted models (scikit-learn's histogram gradient boosting with the pinball loss; Pedregosa et al., 2011) give the 10th, 25th, 50th, 75th and 90th percentiles. Crossing quantiles are sorted into order (Chernozhukov et al., 2010).

**Relative target.** The first version predicted load in MW from absolute lags. Trees cannot predict beyond the values they were trained on, so as Alberta's load grew the forecasts fell progressively low. The final version predicts load divided by its average over the week ending 48 hours earlier, with lags expressed the same way, and multiplies back.

**Retraining.** Each test year Y is forecast by models trained on the seven years ending Y-2, as an operator retraining every January would do.

**Calibration.** Quantile models are usually over-confident on new data. Conformalized quantile regression (Romano et al., 2019) widens each interval by the smallest amount that would have made it hold on year Y-1.

**Benchmark.** The same hour one week earlier, with that method's own error quantiles by hour from 2015 to 2021.

**Scores.** Mean absolute percentage error of the median, mean pinball loss over the five quantiles (a proper score for quantile forecasts; Gneiting and Raftery, 2007), and the share of hours inside the 10 to 90 percent range.

### 3.3 Hour-ahead price forecast and spike warning

The same quantile method is used one hour ahead. The inputs are:
* price lags of 1, 2, 3, 24 and 168 hours, and the 24-hour maximum and spread;
* load and its last change;
* wind, solar and thermal output and the wind share;
* net imports;
* temperature;
* AESO's published hour-ahead forecast for the hour.

Models are trained on five years and calibrated on the year before each test year (2024 and 2025). Quantiles are clipped to the market's $0 and $1,000 limits.

AESO's forecast is turned into a probabilistic benchmark by adding its own 2018 to 2022 errors by hour, so every method is scored the same way. A gradient-boosted classifier trained on the six years before each test year gives the probability that the price reaches $200/MWh or more. It is scored with the Brier score (Brier, 1950) against the training base rate, and checked with a reliability table.

### 3.4 Wind and solar value

Capture price is the generation-weighted average pool price; value factor is capture price divided by the time-weighted average price (Hirth, 2013). Installed capacity counts each asset at its listed capability from its first to its last metered output.

Ramps are measured two ways: the 1st percentile of hourly wind change, and the 99th percentile of the three-hour change in net load (AIL less wind and solar). AESO's day-ahead wind forecast is compared with metered wind output for 2012 to 2016.

### 3.5 Congestion screen

Constrained energy is the hourly constrained-down MW by planning area and fuel. Curtailment rate is constrained energy as a share of metered plus constrained output. Constraints are compared with:
* wind output, by decile of fleet capacity factor;
* pool price;
* each planning area's net injection (metered generation minus load, 2024 and 2025 average).

Areas with positive net injection and constraints in at least 5 percent of hours are flagged as exporting and often constrained. For interties, the import capability offered (ATC) is compared with the path rating (TTC), along with the share of hours when scheduled imports used at least 98 percent of ATC.

### 3.6 AI agent access

`src/gridanalytics/agent_tools.py` defines seven read-only functions:
* headline summary;
* forecast scores;
* forecast for a given hour;
* constrained energy by year and area;
* area balance;
* asset value;
* annual summary.

`mcp_server.py` exposes them through the Model Context Protocol. Every answer names the table it came from, and the server cannot change data or results.

## 4. Results

### 4.1 Data quality

* All 11 AESO files match their recorded checksums.
* The hourly series has no gaps or duplicate hours.
* The pool price stays within its limits: 160 hours at the cap and 897 at zero.
* No asset meter reads negative.
* Metered generation plus net imports is a steady 76 percent of AIL, with no hours flagged. The rest is load served by on-site generation and small plants.
* One gas peaker shows four 24-hour runs of constant output, which is plausible for that unit.
* Several thermal assets exceed their current listed capability in earlier years because they were re-rated; this affects capacity factors only.
* AESO marks April to July 2025 metered volumes as provisional.

### 4.2 Load forecast

| Method (2023 to July 2025) | MAPE | Bias (MW) | Pinball loss | Inside 80% range | 80% range width (MW) |
|---|---|---|---|---|---|
| Same hour last week | 2.99% | +5 | 109.3 | 72.9% | 810 |
| First attempt: absolute load, trained once | 1.95% | -148 | 71.0 | 68.0% | 496 |
| Relative load, retrained yearly, raw quantiles | 1.48% | -3 | 54.6 | 62.7% | 336 |
| **Relative load, retrained yearly, conformal intervals** | **1.48%** | **-3** | **53.2** | **81.2%** | **496** |

**Drift in the first attempt.** Its bias grew from -76 MW in 2023 to -235 MW in 2025. The final method's bias is +3, +5 and -28 MW in the three test years, and its MAPE is 1.41, 1.58 and 1.45 percent.

**Calibration.** The conformal adjustment widened the 80 percent range by 99, 57 and 87 MW for 2023, 2024 and 2025. Coverage was 86.6, 74.5 and 83.3 percent. 2024 was harder to forecast than the year it was calibrated on, which is the known weakness of calibrating on a single past year.

**Quantile loss.** Against the benchmark it is 51 percent lower.

**Inputs.** Ranked by permutation importance:
1. the 48-hour load lag;
2. the 168-hour and 336-hour lags;
3. Edmonton temperature;
4. hour of day;
5. the weekend and holiday flag.

**Largest errors.** Over-forecasts of about 1,100 MW on the afternoon of 25 July 2024, when load fell well below its usual pattern.

### 4.3 Price forecast

| Method (2024 to July 2025) | Mean error ($/MWh) | Pinball loss | Inside 80% range |
|---|---|---|---|
| Persistence (last hour's price) | 19.91 | 9.07 | 74.7% |
| AESO hour-ahead forecast with its error spread | 15.96 | 7.46 | 73.7% |
| Gradient boosting, market data only | 18.10 | 6.82 | 83.7% |
| **Gradient boosting with AESO's forecast as an input** | **13.11** | **4.96** | **84.0%** |

**Market data alone.** The model has a higher mean error than AESO's forecast. AESO sees offers and dispatch that public data does not.

**With AESO's forecast as an input.** The model's median is closer than AESO's own forecast, and its quantile loss is a third lower. It learns when AESO's forecast tends to run high or low, given recent volatility, wind, load and imports.

**Coverage.** The range was slightly wide: 86.0 percent in 2024 and 80.5 percent in 2025, against 80 percent nominal. The calibration year 2023 was more volatile than the test years.

### 4.4 Spike warning

Prices reached $200/MWh or more in 604 hours of 2024 and 2025 (4.4 percent).

* **Brier score:** 0.014, against 0.044 for the training base rate, which is 69 percent skill.
* **AESO forecast as a warning:** a flag whenever AESO's hour-ahead forecast was $200 or more scores 0.025.
* **At a 50 percent threshold:** the warning catches 76 percent of spikes, and 20 percent of warnings are false alarms. The AESO flag catches 80 percent, with 31 percent false alarms.
* **Reliability:** probabilities below 5 percent and above 70 percent match outcomes closely. In between the warning runs high: forecasts near 60 percent came true 48 percent of the time.

### 4.5 Wind and solar value

| | 2019 | 2021 | 2022 | 2023 | 2024 | 2025 (Jan to Jul) |
|---|---|---|---|---|---|---|
| Wind capacity, year end (MW) | 1,696 | 2,289 | 3,223 | 4,522 | 5,582 | 5,582 |
| Wind share of load | 4.8% | 7.2% | 8.5% | 11.9% | 14.4% | 15.6% |
| Average pool price ($/MWh) | 54.88 | 101.93 | 162.46 | 133.63 | 62.78 | 38.83 |
| Wind value factor | 0.72 | 0.68 | 0.64 | 0.56 | 0.53 | 0.56 |
| Solar value factor | 1.29 | 1.32 | 1.25 | 1.08 | 0.82 | 0.83 |
| Battery value factor | | 2.04 | 2.79 | 1.59 | 5.52 | 5.31 |
| Hours at a zero price | 0 | 0 | 2 | 49 | 386 | 334 |

**Value factor and wind share.** Across 2010 to 2025, each extra percentage point of wind share is associated with a value factor about 0.015 lower.

**Prices by wind output, 2024.** The pool price averaged $192 in the least windy tenth of hours and $12 in the windiest, when 22 percent of hours cleared at zero. In 2018 the same contrast was $81 against $29.

**Solar.** Solar lost its afternoon premium as solar capacity reached 1,800 MW and the evening, when solar fades, became the most expensive time of day.

**Spread between plants, 2024.** Wind farms' value factors ranged from 0.29 to 0.64 and solar from 0.58 to 0.95.

**Batteries.** They discharged into the scarcest hours and earned 5.5 times the average price in 2024.

### 4.6 Flexibility and AESO's day-ahead wind forecast

The 99th percentile three-hour rise in net load fell slightly from 2010 to 2022 (1,181 to 1,099 MW), then grew to 1,548 MW in 2025. The fleet's 1st percentile one-hour wind drop grew from 302 MW in 2022 to 571 MW in 2025, about 10 to 12 percent of installed capacity.

AESO's public day-ahead wind forecast for 2012 to 2016:
* **Error:** mean absolute error of 10.3 to 12.9 percent of installed capacity, largest when the forecast was 30 to 60 percent of capacity.
* **Band:** its minimum to maximum band held 67 to 80 percent of outcomes.
* **Direction of misses:** actual output fell below the minimum (15 to 29 percent of hours) far more often than above the maximum (3 to 11 percent), consistent with the forecast's positive bias in 2013 to 2015.

The calibration method in 3.2 would centre and size such a band.

### 4.7 Congestion

| | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|
| Constrained energy (GWh) | 116 | 55 | 92 | 303 | 458 | 1,332 |
| of which wind (GWh) | 0 | 0 | 55 | 251 | 335 | 1,246 |
| Wind curtailment rate | 0.0% | 0.0% | 0.75% | 2.38% | 2.55% | 4.79% (Jan to Jul) |
| Solar curtailment rate | 0.0% | 0.21% | 0.45% | 0.76% | 1.14% | 1.30% (Jan to Jul) |

**Growth and shift.** Constraints were small and mostly thermal until 2022, then grew with each wave of new wind.

**Where, 2025.**
* Medicine Hat: 846 GWh, all wind and 63 percent of the total.
* Hanna: 260 GWh.
* Empress: 45 GWh.
* The South as a whole: 77 percent of constrained energy.

**When.**
* **Share of hours, 2024:** wind or solar was held back in 42 percent of hours, rising from 15 percent of the calmest tenth of hours to 83 percent of the windiest.
* **Prices, 2024:** the pool price averaged $34 in hours with a wind or solar constraint and $84 in other hours.
* **Value:** the constrained energy, valued at the pool price, was worth $5.6 million in 2024. That figure is small because constraints happen when prices are low.

**Net exporters (2024 to 2025 average).**
* Wabamun, with its large thermal units: +1,596 MW in 2024.
* Medicine Hat: +378 MW, reaching +843 MW in the top 5 percent of hours.
* Hinton / Edson: +467 MW.
* Hanna: +274 MW.
* Stavely: +238 MW.
* Sheerness: +176 MW.

Edmonton (-1,261 MW in 2024) is by far the largest importer.

**Screen result.** Six areas export and are constrained in at least 5 percent of hours: Medicine Hat, Hanna, Sheerness, Hinton / Edson, Stavely and Strathmore / Blackie.

### 4.8 Intertie limits (2013 to 2020)

From 2016 to 2020 the combined BC and Montana import path offered on average 57 percent of its rating (49 to 62 percent by year), against 77 to 81 percent in 2013 to 2015. The most common recorded reasons were:
* the load shed service rule for imports (ISO rules Section 303.1), alone or with insufficient contingency reserve: about 45,000 hours together;
* specific line outages.

Imports ran at the offered limit in 33 percent of hours in 2020.

## 5. Discussion

**Forecasts that hold up out of sample need operational design as much as a strong model.** The first load model scored well on 2023 and then drifted 235 MW low by 2025, a failure that a single test year would have hidden. Forecasting relative to the recent level and retraining each year fixed it. Calibrating every year on the most recent year turned an over-confident 63 percent range into a usable 81 percent one, which matters for reserve and risk decisions that depend on the range rather than the median.

**AESO's forecasts are the right starting point.** The price model only beats AESO when AESO's forecast is one of its inputs. The value is a probabilistic layer on top of it: a calibrated range and a spike probability with fewer false alarms than a simple threshold on AESO's point forecast.

**Wind is cannibalizing its own price.** Wind earns about 55 cents per dollar of average price, and plants that produce with the rest of the fleet earn least. Batteries earn five times the average because they deliver exactly what wind does not. This points the next tranche of value toward storage, demand response and the evening ramp, which grew by about 450 MW between 2022 and 2025.

**Constraints are concentrated, which suits a locational price.** Most constrained energy sits in a few southern and east-central areas that export far more than they use, and constraints coincide with low pool prices and high wind. Under the locational pricing in Alberta's restructured market design, these are the areas where nodal prices would most likely fall below the system price. Generation sited there would earn less, and new load there (data centres, for example) would see lower prices than at the system level. A network model is needed to say by how much.

## 6. Conclusions

1. Day-ahead load can be forecast to within about 1.5 percent with calibrated 80 percent ranges, provided the model follows load growth and is recalibrated each year.
2. A probabilistic layer on AESO's hour-ahead price forecast reduces its error by 18 percent, and provides a spike probability that is reliable at the low and high ends.
3. Wind's value factor has fallen to about 0.55 and solar's to about 0.83; zero-price hours went from none in 2019 to 386 in 2024.
4. Generation held back by transmission limits rose more than tenfold from 2020 to 2025 and is concentrated in Medicine Hat and Hanna: a first screen for where locational prices would separate.
5. Every result is reproducible from public data and can be queried by an AI agent through read-only tools that cite their source tables.

## 7. Limitations

* **Temperature inputs.** They are reanalysis values for the hour, standing in for a day-ahead weather forecast, so load scores are better than a live forecast would achieve. A live system would use forecast temperature and score that.
* **Calibration.** Conformal calibration on one past year assumes the next year behaves similarly; 2024 coverage (74.5 percent) shows the limit.
* **The price model and AESO's forecast.** The model depends on AESO's hour-ahead forecast and does not replace AESO's dispatch and pricing tools.
* **Constraint volumes.** They show generation held back, not which transmission element bound or what nodal prices would be. The screen ranks exposure; it does not estimate price separation.
* **Capacity.** Capacity uses current listed capabilities; re-rated assets make early-year capacity factors approximate.
* **Provisional data.** 2025 metered data for April to July is provisional, and 2025 generation figures cover January to July.

## 8. Reproducing

```
pip install -r requirements.txt
python scripts/get_data.py        # downloads and checks the AESO files
python scripts/prepare_data.py    # processed tables, asset list, planning area map
python scripts/check_data.py      # data quality report
python run_forecast.py            # load and price forecasts (about 3 minutes)
python run_market.py              # wind and solar value, congestion screen
python run_ml_analysis.py         # machine learning note (about 40 minutes)
python scripts/build_dashboard.py # docs/index.html
python scripts/make_figures.py    # figures/
python scripts/make_ml_figures.py # figures for the machine learning note
pytest                            # 24 tests
```

## References

* Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1 to 3.
* Chernozhukov, V., Fernández-Val, I. and Galichon, A. (2010). Quantile and probability curves without crossing. *Econometrica*, 78(3), 1093 to 1125.
* Gneiting, T. and Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359 to 378.
* Hersbach, H. et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146(730), 1999 to 2049.
* Hirth, L. (2013). The market value of variable renewables: the effect of solar wind power variability on their relative price. *Energy Economics*, 38, 218 to 236.
* Hong, T., Pinson, P., Fan, S., Zareipour, H., Troccoli, A. and Hyndman, R. J. (2016). Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond. *International Journal of Forecasting*, 32(3), 896 to 913.
* Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. *Journal of Machine Learning Research*, 12, 2825 to 2830.
* Romano, Y., Patterson, E. and Candès, E. (2019). Conformalized quantile regression. *Advances in Neural Information Processing Systems*, 32.

Data: Alberta Electric System Operator; ERA5, Copernicus Climate Change Service, through Open-Meteo. Independent project; not affiliated with or endorsed by the AESO.
