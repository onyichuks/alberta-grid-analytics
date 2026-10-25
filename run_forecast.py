"""Probabilistic forecasts: day-ahead Alberta Internal Load and hour-ahead pool price, tested on unseen years.

Usage:  python run_forecast.py
Writes outputs/forecast_results.json, outputs/forecast_dashboard.json and outputs/tables/f*.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from gridanalytics import data as D, forecast as F  # noqa: E402
from gridanalytics.outputs import to_json, rnd  # noqa: E402

TZ = "America/Edmonton"
SPIKE = 200.0   # $/MWh threshold for a price spike
PRICE_FLOOR, PRICE_CAP = 0.0, 1000.0   # pool price limits before the restructured market


def local_year(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.tz_convert(TZ).year.to_numpy()


def period(idx, start, end):
    loc = idx.tz_convert(TZ)
    return (loc >= pd.Timestamp(start, tz=TZ)) & (loc < pd.Timestamp(end, tz=TZ))


def time_axis(idx: pd.DatetimeIndex) -> dict:
    """Compact time axis for the dashboard: first hour in UTC plus hourly offsets (empty when contiguous)."""
    off = ((idx - idx[0]) / pd.Timedelta(hours=1)).astype(int)
    contiguous = bool((np.diff(off) == 1).all())
    return {"t0_utc": idx[0].strftime("%Y-%m-%dT%H:00:00Z"), "offsets": [] if contiguous else off.tolist()}


def score_table(name: str, y: pd.Series, P: pd.DataFrame, years, mape: bool = True) -> list[dict]:
    rows = [{"model": name, "period": "all test", **F.scores(y, P, mape)}]
    yr = local_year(y.index)
    for yv in years:
        m = yr == yv
        rows.append({"model": name, "period": str(yv), **F.scores(y[m], P.loc[y.index[m]], mape)})
    return rows


def run_load(sysd: pd.DataFrame) -> tuple[dict, dict]:
    y = sysd["ail_mw"]
    X = F.load_features(sysd)
    level = F.load_level(y)
    ly = local_year(sysd.index)
    valid = X.notna().all(axis=1) & y.notna() & level.notna()
    te = valid & (ly >= 2023)
    years = [2023, 2024, 2025]

    # 1. Benchmark: same hour last week, with that method's own error spread from 2015 to 2021
    naive = F.seasonal_naive(y, 168, valid & (ly >= 2015) & (ly <= 2021))
    # 2. First attempt kept for comparison: absolute load lags, one model trained on 2015 to 2021
    Xabs = X.drop(columns=[c for c in X.columns if c.endswith("_rel")]).copy()
    for c in [c for c in X.columns if c.endswith("_rel")]:
        Xabs[c.replace("_rel", "")] = X[c] * level
    tr0 = valid & (ly >= 2015) & (ly <= 2021)
    ca0 = valid & (ly == 2022)
    m0 = F.QuantileForecaster().fit(Xabs[tr0], y[tr0])
    m0.conformalize(Xabs[ca0], y[ca0])
    fixed = m0.predict(Xabs[te])
    # 3. Final method: load relative to its recent level, retrained every year, conformal intervals
    raw, cal, log = F.rolling_origin(X, y, years, 7, ly, scale=level)
    model = F.rolling_origin.last_model

    rows = (score_table("Seasonal naive (same hour last week)", y[te], naive[te], years)
            + score_table("First attempt: absolute load, trained once", y[te], fixed, years)
            + score_table("Relative load, retrained yearly, raw quantiles", y[te], raw, years)
            + score_table("Relative load, retrained yearly, conformal intervals", y[te], cal, years))
    tab = pd.DataFrame(rows)
    tab.to_csv(ROOT / "outputs" / "tables" / "f01_load_scores.csv", index=False)

    err = (cal[0.5] - y[te])
    month = err.index.tz_convert(TZ).strftime("%Y-%m")
    inside = ((y[te] >= cal[0.1]) & (y[te] <= cal[0.9])) * 100
    monthly = pd.DataFrame({"mape_pct": (err.abs() / y[te] * 100).groupby(month).mean(),
                            "bias_mw": err.groupby(month).mean(),
                            "coverage_80_pct": inside.groupby(month).mean()})
    by_hour = (err.abs() / y[te] * 100).groupby(err.index.tz_convert(TZ).hour).mean()
    worst = err.abs().sort_values(ascending=False).head(8)
    te25 = valid & (ly == 2025)
    smp = X[te25].sample(4000, random_state=0)
    imp = permutation_importance(model.models_[0.5], smp, (y / level)[smp.index])

    allte = tab[tab.period == "all test"].set_index("model")
    fin = "Relative load, retrained yearly, conformal intervals"
    res = {
        "setup": {"test": "2023 to July 2025", "retraining": log,
                  "issue_time": "morning of the day before (all load inputs at least 48 hours old)",
                  "weather_input": "ERA5 temperature for the hour, standing in for a day-ahead weather forecast",
                  "test_hours": int(te.sum()), "features": list(X.columns)},
        "scores": rows,
        "headline": {
            "mape_naive_pct": allte.loc["Seasonal naive (same hour last week)", "mape_pct"],
            "mape_fixed_pct": allte.loc["First attempt: absolute load, trained once", "mape_pct"],
            "bias_fixed_mw": allte.loc["First attempt: absolute load, trained once", "bias"],
            "mape_gbm_pct": allte.loc[fin, "mape_pct"],
            "mae_gbm_mw": allte.loc[fin, "mae"],
            "bias_gbm_mw": allte.loc[fin, "bias"],
            "coverage_raw_pct": allte.loc["Relative load, retrained yearly, raw quantiles", "coverage_10_90_pct"],
            "coverage_cal_pct": allte.loc[fin, "coverage_10_90_pct"],
            "coverage_fixed_pct": allte.loc["First attempt: absolute load, trained once", "coverage_10_90_pct"],
            "width_cal_mw": allte.loc[fin, "width_10_90"],
            "pinball_skill_pct": 100 * (1 - allte.loc[fin, "pinball_mean"]
                                        / allte.loc["Seasonal naive (same hour last week)", "pinball_mean"]),
        },
        "monthly": monthly.reset_index(names="month").to_dict("records"),
        "mape_by_hour": {int(k): float(v) for k, v in by_hour.items()},
        "worst_hours": [{"time_local": t.tz_convert(TZ).strftime("%Y-%m-%d %H:00"), "error_mw": float(err[t]),
                         "actual_mw": float(y[t])} for t in worst.index],
        "importance": imp,
    }
    show = te & (ly >= 2024)
    idx = sysd.index[show]
    dash = {**time_axis(idx),
            "actual": rnd(y[idx], 0),
            **{f"q{int(q*100)}": rnd(cal.loc[idx, q], 0) for q in F.QUANTILES},
            "naive50": rnd(naive.loc[idx, 0.5], 0),
            "temp": rnd(sysd.loc[idx, "edmonton_temp_c"], 1)}
    return res, dash


def permutation_importance(model, X: pd.DataFrame, y: pd.Series, repeats: int = 3) -> list[dict]:
    rng = np.random.default_rng(0)
    base = np.mean(np.abs(model.predict(X) - y))
    out = []
    for c in X.columns:
        inc = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[c] = rng.permutation(Xp[c].to_numpy())
            inc.append(np.mean(np.abs(model.predict(Xp) - y)) - base)
        out.append({"feature": c, "mae_increase": float(np.mean(inc))})
    return sorted(out, key=lambda r: -r["mae_increase"])


def run_price(sysd: pd.DataFrame) -> tuple[dict, dict]:
    y = sysd["pool_price"]
    Xa = F.price_features(sysd, with_aeso=True)
    Xn = F.price_features(sysd, with_aeso=False)
    ly = local_year(sysd.index)
    valid = Xa.notna().all(axis=1) & y.notna()
    te = valid & (ly >= 2024)
    years = [2024, 2025]
    base_tr = valid & (ly >= 2018) & (ly <= 2022)

    persist = F.seasonal_naive(y, 1, base_tr)
    # AESO's hour-ahead forecast as the median, with its own 2018 to 2022 errors (by hour) as the spread
    aeso_point = sysd["aeso_ha_price_fc"]
    err = (y - aeso_point)[base_tr]
    hour = sysd.index.tz_convert(TZ).hour
    qtab = err.groupby(hour[base_tr]).quantile(list(F.QUANTILES)).unstack()
    aeso = F.sort_quantiles(pd.DataFrame({q: aeso_point.to_numpy() + qtab[q].reindex(hour).to_numpy()
                                          for q in F.QUANTILES}, index=sysd.index))
    _, pn, _ = F.rolling_origin(Xn, y, years, 5, ly)
    raw, pa, log = F.rolling_origin(Xa, y, years, 5, ly)
    # The pool price is bounded by the market floor and cap, so no quantile should go beyond them.
    persist, aeso, pn, raw, pa = (P.clip(PRICE_FLOOR, PRICE_CAP) for P in (persist, aeso, pn, raw, pa))

    rows = (score_table("Persistence (last hour's price)", y[te], persist[te], years, mape=False)
            + score_table("AESO hour-ahead forecast with its error spread", y[te], aeso[te], years, mape=False)
            + score_table("Gradient boosting, market data only", y[te], pn, years, mape=False)
            + score_table("Gradient boosting with the AESO forecast as an input, raw quantiles", y[te], raw, years, mape=False)
            + score_table("Gradient boosting with the AESO forecast as an input", y[te], pa, years, mape=False))
    tab = pd.DataFrame(rows)
    tab.to_csv(ROOT / "outputs" / "tables" / "f02_price_scores.csv", index=False)

    # Probability of a spike (price at or above SPIKE), retrained yearly on the six years before
    p_te = pd.Series(np.nan, index=sysd.index)
    base = {}
    for Y in years:
        trm = valid & (ly >= Y - 6) & (ly <= Y - 1)
        tem = valid & (ly == Y)
        clf = F.spike_classifier().fit(Xa[trm], (y[trm] >= SPIKE).astype(int))
        p_te[tem] = clf.predict_proba(Xa[tem])[:, 1]
        base[Y] = float((y[trm] >= SPIKE).mean())
    p = p_te[te].to_numpy()
    s_te = (y[te] >= SPIKE).astype(int).to_numpy()
    clim = np.array([base[v] for v in ly[te]])
    b_model, b_clim = F.brier(s_te, p), F.brier(s_te, clim)
    aeso_flag = (aeso_point[te] >= SPIKE).astype(int).to_numpy()
    spike = {"threshold": SPIKE, "base_rate_train_pct": 100 * float(np.mean(list(base.values()))),
             "rate_test_pct": float(100 * s_te.mean()), "spike_hours_test": int(s_te.sum()),
             "brier_model": b_model, "brier_climatology": b_clim, "brier_skill_pct": 100 * (1 - b_model / b_clim),
             "brier_aeso_flag": F.brier(s_te, aeso_flag),
             "reliability": F.reliability(s_te, p),
             "hit_rate_at_50pct": float(((p >= 0.5) & (s_te == 1)).sum() / max(1, s_te.sum()) * 100),
             "false_alarm_at_50pct": float(((p >= 0.5) & (s_te == 0)).sum() / max(1, (p >= 0.5).sum()) * 100),
             "aeso_hit_rate": float(((aeso_flag == 1) & (s_te == 1)).sum() / max(1, s_te.sum()) * 100),
             "aeso_false_alarm": float(((aeso_flag == 1) & (s_te == 0)).sum() / max(1, aeso_flag.sum()) * 100)}

    allte = tab[tab.period == "all test"].set_index("model")
    fin = "Gradient boosting with the AESO forecast as an input"
    ae = "AESO hour-ahead forecast with its error spread"
    res = {
        "setup": {"test": "2024 to July 2025", "retraining": log,
                  "issue_time": "one hour before delivery (AESO's hour-ahead forecast is published before the hour)",
                  "test_hours": int(te.sum()), "features_with_aeso": list(Xa.columns)},
        "scores": rows,
        "headline": {
            "mae_aeso": allte.loc[ae, "mae"], "mae_gbm_aeso": allte.loc[fin, "mae"],
            "mae_gbm_market": allte.loc["Gradient boosting, market data only", "mae"],
            "mae_persist": allte.loc["Persistence (last hour's price)", "mae"],
            "mae_cut_vs_aeso_pct": 100 * (1 - allte.loc[fin, "mae"] / allte.loc[ae, "mae"]),
            "pinball_aeso": allte.loc[ae, "pinball_mean"], "pinball_gbm_aeso": allte.loc[fin, "pinball_mean"],
            "pinball_cut_vs_aeso_pct": 100 * (1 - allte.loc[fin, "pinball_mean"] / allte.loc[ae, "pinball_mean"]),
            "coverage_cal_pct": allte.loc[fin, "coverage_10_90_pct"],
            "coverage_raw_pct": allte.loc["Gradient boosting with the AESO forecast as an input, raw quantiles", "coverage_10_90_pct"],
            "coverage_aeso_pct": allte.loc[ae, "coverage_10_90_pct"],
        },
        "spike": spike,
    }
    idx = sysd.index[te]
    dash = {**time_axis(idx),
            "actual": rnd(y[idx], 1), "aeso": rnd(aeso_point[idx], 1),
            **{f"q{int(q*100)}": rnd(pa.loc[idx, q], 0) for q in F.QUANTILES},
            "p_spike": rnd(p_te[idx], 3)}
    return res, dash


def main() -> None:
    (ROOT / "outputs" / "tables").mkdir(parents=True, exist_ok=True)
    sysd = D.load_processed("system")
    load_res, load_dash = run_load(sysd)
    print("load:", {k: round(v, 2) for k, v in load_res["headline"].items()})
    price_res, price_dash = run_price(sysd)
    print("price:", {k: round(v, 2) for k, v in price_res["headline"].items()})
    print("spike:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in price_res["spike"].items() if k != "reliability"})
    (ROOT / "outputs" / "forecast_results.json").write_text(to_json({"load": load_res, "price": price_res}, indent=1))
    (ROOT / "outputs" / "forecast_dashboard.json").write_text(to_json({"load": load_dash, "price": price_dash}, compact=True))


if __name__ == "__main__":
    main()
