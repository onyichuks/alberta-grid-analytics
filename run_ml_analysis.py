"""Machine learning checks behind the forecasts, for docs/ML_TECHNICAL_NOTE.md.

* training curves on a later, time-ordered validation year (not a random split);
* how much history the load model needs (learning curve by years of training data);
* a small hyperparameter check that uses only years before the test period;
* linear quantile regression and logistic regression baselines with the same inputs and calibration;
* where the errors are: by hour, month, day type, temperature and price level;
* which input groups matter (grouped permutation importance);
* the spike warning against a logistic baseline (Brier score, ROC AUC, average precision, reliability).

Usage:  python run_ml_analysis.py      (about 40 minutes on two cores)
Writes outputs/ml_results.json and outputs/tables/m*.csv.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from gridanalytics import baselines as B, data as D, forecast as F  # noqa: E402
from gridanalytics.outputs import to_json  # noqa: E402

TZ = "America/Edmonton"
SPIKE = 200.0
PRICE_FLOOR, PRICE_CAP = 0.0, 1000.0
TABLES = ROOT / "outputs" / "tables"
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def local_year(idx) -> np.ndarray:
    return idx.tz_convert(TZ).year.to_numpy()


def score_rows(name: str, y: pd.Series, P: pd.DataFrame, years, mape=True) -> list[dict]:
    rows = [{"model": name, "period": "all test", **F.scores(y, P.loc[y.index], mape)}]
    yr = local_year(y.index)
    for v in years:
        m = yr == v
        rows.append({"model": name, "period": str(v), **F.scores(y[m], P.loc[y.index[m]], mape)})
    return rows


def hgb(q: float, **kw) -> HistGradientBoostingRegressor:
    """The production quantile model with selected settings changed."""
    m = F.make_quantile_model(q)
    return m.set_params(**kw)


def training_curve(X, t, scale, tr, va, quantiles=(0.1, 0.5, 0.9), iters=1500, every=5, rng_seed=0) -> dict:
    """Pinball loss by boosting iteration on the training years and on the following (validation) year,
    in original units (MW or $/MWh). Early stopping is switched off so the whole curve is visible."""
    rng = np.random.default_rng(rng_seed)
    tr_idx = X.index[tr]
    tr_idx = tr_idx[rng.choice(len(tr_idx), size=min(20000, len(tr_idx)), replace=False)]
    va_idx = X.index[va]
    out = {"iterations": list(range(every, iters + 1, every))}
    y_true = t * (1 if scale is None else scale)
    for q in quantiles:
        m = hgb(q, max_iter=iters, early_stopping=False).fit(X[tr], t[tr])
        curves = {}
        for name, idx in (("train", tr_idx), ("validation", va_idx)):
            s = 1 if scale is None else scale[idx].to_numpy()
            yt = y_true[idx].to_numpy()
            vals = []
            for i, p in enumerate(m.staged_predict(X.loc[idx]), start=1):
                if i % every == 0:
                    vals.append(F.pinball(yt, p * s, q))
            curves[name] = vals
        v = np.array(curves["validation"])
        best = int(np.argmin(v))
        out[f"q{int(q*100)}"] = {**curves, "best_iteration": out["iterations"][best], "best_loss": float(v[best]),
                                 "loss_at_400": float(v[out["iterations"].index(400)]),
                                 "loss_at_1500": float(v[-1])}
    return out


def production_iterations(model) -> dict:
    return {f"q{int(q*100)}": int(m.n_iter_) for q, m in model.models_.items()}


def hyper_grid(X, t, scale, y, tr, va, metric: str) -> list[dict]:
    rows = []
    for lr in (0.03, 0.06, 0.12):
        for leaves in (16, 48, 128):
            m = hgb(0.5, learning_rate=lr, max_leaf_nodes=leaves).fit(X[tr], t[tr])
            p = m.predict(X[va]) * (1 if scale is None else scale[va].to_numpy())
            e = np.abs(p - y[va].to_numpy())
            val = float(np.mean(e / y[va].to_numpy()) * 100) if metric == "mape" else float(np.mean(e))
            rows.append({"learning_rate": lr, "max_leaf_nodes": leaves, metric: val, "iterations_used": int(m.n_iter_),
                         "production": lr == 0.06 and leaves == 48})
    best = min(r[metric] for r in rows)
    for r in rows:
        r["vs_best_pct"] = 100 * (r[metric] / best - 1)
    return rows


def seg_table(models: dict, y: pd.Series, seg: pd.Series, name: str, mape=True) -> pd.DataFrame:
    parts = []
    for label, P in models.items():
        s = B.segment_scores(y, P.loc[y.index], seg, mape)
        s.insert(0, "model", label)
        parts.append(s.reset_index(names=name))
    return pd.concat(parts, ignore_index=True)


def records(df: pd.DataFrame) -> list[dict]:
    return df.to_dict("records")  # NaN becomes null in to_json


# ============================================================================ load
def run_load(sysd: pd.DataFrame) -> dict:
    y = sysd["ail_mw"]
    X = F.load_features(sysd)
    level = F.load_level(y)
    t = y / level
    ly = local_year(sysd.index)
    valid = X.notna().all(axis=1) & y.notna() & level.notna()
    te = valid & (ly >= 2023)
    years = [2023, 2024, 2025]
    yte = y[te]

    naive = F.seasonal_naive(y, 168, valid & (ly >= 2015) & (ly <= 2021))
    raw, cal, _ = F.rolling_origin(X, y, years, 7, ly, scale=level)
    final_model = F.rolling_origin.last_model
    log("load: boosted model done")
    Dl = B.linear_load_design(X)
    lraw, lcal = B.rolling_origin_linear(Dl, y, years, 7, ly, scale=level)
    solvers = B.rolling_origin_linear.log_
    log(f"load: linear quantile regression done {solvers}")

    rows = (score_rows("Seasonal naive (same hour last week)", yte, naive, years)
            + score_rows("Linear quantile regression, raw", yte, lraw, years)
            + score_rows("Linear quantile regression, conformal", yte, lcal, years)
            + score_rows("Gradient boosting, raw", yte, raw, years)
            + score_rows("Gradient boosting, conformal", yte, cal, years))
    pd.DataFrame(rows).to_csv(TABLES / "m01_load_model_comparison.csv", index=False)

    # Learning curve: years of history
    lc = []
    for k in (1, 2, 3, 5, 7, 9):
        if k == 7:
            r_raw, r_cal = raw, cal
        else:
            r_raw, r_cal, _ = F.rolling_origin(X, y, years, k, ly, scale=level)
        s_cal, s_raw = F.scores(yte, r_cal.loc[yte.index]), F.scores(yte, r_raw.loc[yte.index])
        lc.append({"train_years": k, "mape_pct": s_cal["mape_pct"], "pinball_mean": s_cal["pinball_mean"],
                   "bias": s_cal["bias"], "coverage_raw_pct": s_raw["coverage_10_90_pct"],
                   "coverage_cal_pct": s_cal["coverage_10_90_pct"], "width_cal": s_cal["width_10_90"]})
        log(f"load: learning curve {k} years -> MAPE {s_cal['mape_pct']:.2f}")
    pd.DataFrame(lc).to_csv(TABLES / "m02_load_learning_curve.csv", index=False)

    # Training curve for the 2025 model: train 2017 to 2023, validate on 2024
    trm = valid & (ly >= 2017) & (ly <= 2023)
    vam = valid & (ly == 2024)
    curve = training_curve(X, t, level, trm, vam)
    log("load: training curves done")

    # Hyperparameter check using only pre-test years: train 2015 to 2021, validate on 2022
    grid = hyper_grid(X, t, level, y, valid & (ly >= 2015) & (ly <= 2021), valid & (ly == 2022), "mape")
    pd.DataFrame(grid).to_csv(TABLES / "m03_load_hyperparameters.csv", index=False)
    log("load: hyperparameter grid done")

    # Where the errors are
    loc = yte.index.tz_convert(TZ)
    tavg = ((sysd["calgary_temp_c"] + sysd["edmonton_temp_c"]) / 2)[te]
    hol = X.loc[te, "holiday"].to_numpy() == 1
    daytype = pd.Series(np.where(hol, "Holiday", np.where(loc.dayofweek >= 5, "Weekend", "Weekday")), index=yte.index)
    tband = pd.cut(tavg, [-60, -20, -10, 0, 10, 20, 50],
                   labels=["below -20", "-20 to -10", "-10 to 0", "0 to 10", "10 to 20", "above 20"])
    models = {"Seasonal naive": naive, "Linear quantile regression": lcal, "Gradient boosting": cal}
    segs = {"hour": pd.Series(loc.hour, index=yte.index), "month": pd.Series(loc.month, index=yte.index),
            "day_type": daytype, "temperature_band_c": tband}
    seg_out = {}
    for name, s in segs.items():
        tab = seg_table(models, yte, s, name)
        tab.to_csv(TABLES / f"m04_load_errors_by_{name}.csv", index=False)
        seg_out[name] = records(tab)

    # Grouped permutation importance (2025 model, 4,000 hours of 2025)
    groups = {"Recent load (48 to 336 hours earlier)": [c for c in X.columns if c.endswith("_rel")],
              "Temperature (both cities)": [c for c in X.columns if "temp" in c],
              "Hour of day": ["hour"],
              "Day type (weekday, weekend, holiday)": ["dow", "holiday", "weekend_or_holiday"],
              "Season (month, day of year)": ["month", "doy_sin", "doy_cos"]}
    smp = X[valid & (ly == 2025)].sample(4000, random_state=0)
    lv = level[smp.index].to_numpy()
    imp = B.grouped_importance(lambda Z: final_model.models_[0.5].predict(Z) * lv, smp, y[smp.index], groups)

    return {"scores": rows, "learning_curve": lc, "training_curve": curve, "hyperparameters": grid,
            "production_iterations_2025": production_iterations(final_model),
            "segments": seg_out, "grouped_importance": imp, "linear_solvers": solvers,
            "linear_design_columns": int(Dl.shape[1]), "boosted_features": int(X.shape[1]),
            "test_hours": int(te.sum())}


# ============================================================================ price
def run_price(sysd: pd.DataFrame) -> dict:
    y = sysd["pool_price"]
    Xa = F.price_features(sysd, with_aeso=True)
    ly = local_year(sysd.index)
    valid = Xa.notna().all(axis=1) & y.notna()
    te = valid & (ly >= 2024)
    years = [2024, 2025]
    yte = y[te]
    base_tr = valid & (ly >= 2018) & (ly <= 2022)

    aeso_point = sysd["aeso_ha_price_fc"]
    err = (y - aeso_point)[base_tr]
    hour = sysd.index.tz_convert(TZ).hour
    qtab = err.groupby(hour[base_tr]).quantile(list(F.QUANTILES)).unstack()
    aeso = F.sort_quantiles(pd.DataFrame({q: aeso_point.to_numpy() + qtab[q].reindex(hour).to_numpy()
                                          for q in F.QUANTILES}, index=sysd.index))
    raw, pa, _ = F.rolling_origin(Xa, y, years, 5, ly)
    final_model = F.rolling_origin.last_model
    log("price: boosted model done")
    Dp = B.linear_price_design(Xa)
    lraw, lcal = B.rolling_origin_linear(Dp, y, years, 5, ly)
    solvers = B.rolling_origin_linear.log_
    log(f"price: linear quantile regression done {solvers}")
    aeso, raw, pa, lraw, lcal = (P.clip(PRICE_FLOOR, PRICE_CAP) for P in (aeso, raw, pa, lraw, lcal))

    rows = (score_rows("AESO hour-ahead forecast with its error spread", yte, aeso, years, mape=False)
            + score_rows("Linear quantile regression, raw", yte, lraw, years, mape=False)
            + score_rows("Linear quantile regression, conformal", yte, lcal, years, mape=False)
            + score_rows("Gradient boosting, raw", yte, raw, years, mape=False)
            + score_rows("Gradient boosting, conformal", yte, pa, years, mape=False))
    pd.DataFrame(rows).to_csv(TABLES / "m05_price_model_comparison.csv", index=False)

    trm = valid & (ly >= 2019) & (ly <= 2023)
    vam = valid & (ly == 2024)
    curve = training_curve(Xa, y, None, trm, vam)
    log("price: training curves done")
    grid = hyper_grid(Xa, y, None, y, valid & (ly >= 2018) & (ly <= 2022), valid & (ly == 2023), "mae")
    pd.DataFrame(grid).to_csv(TABLES / "m06_price_hyperparameters.csv", index=False)
    log("price: hyperparameter grid done")

    loc = yte.index.tz_convert(TZ)
    band = pd.cut(yte, [-1, 25, 50, 100, 200, 500, 1001],
                  labels=["below $25", "$25 to 50", "$50 to 100", "$100 to 200", "$200 to 500", "$500 and above"])
    models = {"AESO forecast": aeso, "Linear quantile regression": lcal, "Gradient boosting": pa}
    seg_out = {}
    for name, s in {"price_band": band, "hour": pd.Series(loc.hour, index=yte.index)}.items():
        tab = seg_table(models, yte, s, name, mape=False)
        tab.to_csv(TABLES / f"m07_price_errors_by_{name}.csv", index=False)
        seg_out[name] = records(tab)
    spike_hours = yte >= SPIKE
    share = {}
    for label, P in models.items():
        e = (P.loc[yte.index, 0.5] - yte).abs()
        share[label] = {"error_share_spike_hours_pct": float(100 * e[spike_hours].sum() / e.sum()),
                        "mae_excluding_spikes": float(e[~spike_hours].mean())}

    groups = {"AESO hour-ahead forecast": ["aeso_ha_forecast"],
              "Recent prices": [c for c in Xa.columns if c.startswith("price_")],
              "Load": ["ail_lag1", "ail_change_lag1"],
              "Wind, solar and thermal output": ["wind_lag1", "solar_lag1", "wind_share_lag1", "wind_change_lag1",
                                                 "thermal_lag1"],
              "Net imports": ["net_import_lag1"],
              "Temperature": ["calgary_temp", "edmonton_temp"],
              "Calendar": ["hour", "dow", "month", "doy_sin", "doy_cos", "holiday", "weekend_or_holiday"]}
    smp = Xa[valid & (ly == 2025)].sample(4000, random_state=0)
    imp = B.grouped_importance(lambda Z: final_model.models_[0.5].predict(Z), smp, y[smp.index], groups)

    # Spike warning: boosted classifier against logistic regression, same inputs and training windows
    p_gb = pd.Series(np.nan, index=sysd.index)
    p_lr = pd.Series(np.nan, index=sysd.index)
    base, iters = {}, {}
    for Y in years:
        tr = valid & (ly >= Y - 6) & (ly <= Y - 1)
        tm = valid & (ly == Y)
        s = (y[tr] >= SPIKE).astype(int)
        clf = F.spike_classifier().fit(Xa[tr], s)
        iters[str(Y)] = int(clf.n_iter_)
        p_gb[tm] = clf.predict_proba(Xa[tm])[:, 1]
        p_lr[tm] = B.logistic_spike_model().fit(Xa[tr], s).predict_proba(Xa[tm])[:, 1]
        base[Y] = float(s.mean())
    log("price: spike classifiers done")
    s_te = (yte >= SPIKE).astype(int).to_numpy()
    clim = np.array([base[v] for v in local_year(yte.index)])
    b_clim = F.brier(s_te, clim)
    spike = {"climatology_brier": b_clim, "classifier_iterations": iters, "models": {}}
    for label, p in (("Gradient boosting", p_gb[te].to_numpy()), ("Logistic regression", p_lr[te].to_numpy())):
        b = F.brier(s_te, p)
        prec, rec, thr = precision_recall_curve(s_te, p)
        keep = np.unique(np.linspace(0, len(thr) - 1, 200).astype(int))
        thresholds = []
        for th in (0.2, 0.3, 0.5, 0.7):
            w = p >= th
            thresholds.append({"threshold": th, "warnings": int(w.sum()),
                               "hit_rate_pct": float(100 * (w & (s_te == 1)).sum() / s_te.sum()),
                               "false_alarm_pct": float(100 * (w & (s_te == 0)).sum() / max(1, w.sum()))})
        spike["models"][label] = {"brier": b, "brier_skill_pct": 100 * (1 - b / b_clim),
                                  "roc_auc": float(roc_auc_score(s_te, p)),
                                  "average_precision": float(average_precision_score(s_te, p)),
                                  "reliability": F.reliability(s_te, p), "thresholds": thresholds,
                                  "pr_curve": {"precision": prec[keep].round(4).tolist(),
                                               "recall": rec[keep].round(4).tolist()}}
    aeso_flag = (aeso_point[te] >= SPIKE).astype(int).to_numpy()
    spike["aeso_flag"] = {"brier": F.brier(s_te, aeso_flag),
                          "hit_rate_pct": float(100 * (aeso_flag & s_te).sum() / s_te.sum()),
                          "false_alarm_pct": float(100 * ((aeso_flag == 1) & (s_te == 0)).sum() / aeso_flag.sum())}
    spike["base_rate_test_pct"] = float(100 * s_te.mean())

    return {"scores": rows, "training_curve": curve, "hyperparameters": grid,
            "production_iterations_2025": production_iterations(final_model),
            "segments": seg_out, "spike_hour_error_share": share, "grouped_importance": imp, "spike": spike,
            "linear_solvers": solvers, "linear_design_columns": int(Dp.shape[1]),
            "test_hours": int(te.sum())}


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    sysd = D.load_processed("system")
    out = {"load": run_load(sysd), "price": run_price(sysd)}
    (ROOT / "outputs" / "ml_results.json").write_text(to_json(out, indent=1))
    log("done")


if __name__ == "__main__":
    main()
