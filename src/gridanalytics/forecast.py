"""Probabilistic forecasting of Alberta Internal Load (day ahead) and pool price (hour ahead).

Quantile models are gradient-boosted trees fitted with the pinball (quantile) loss. Intervals are then
calibrated with split conformal prediction on a held-out year (conformalized quantile regression,
Romano et al., 2019), so the stated coverage holds on data the model has not seen.
"""
from __future__ import annotations

from dataclasses import dataclass

import holidays
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

TZ = "America/Edmonton"
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


# ----------------------------------------------------------------------------- features
def calendar(idx: pd.DatetimeIndex) -> pd.DataFrame:
    loc = idx.tz_convert(TZ)
    years = range(loc.year.min() - 1, loc.year.max() + 2)
    ab = holidays.Canada(subdiv="AB", years=years)
    d = pd.DataFrame(index=idx)
    d["hour"] = loc.hour
    d["dow"] = loc.dayofweek
    d["month"] = loc.month
    doy = loc.dayofyear.to_numpy()
    d["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    dates = loc.date
    d["holiday"] = np.array([dt in ab for dt in dates], dtype=int)
    d["weekend_or_holiday"] = ((d["dow"] >= 5) | (d["holiday"] == 1)).astype(int)
    return d


def load_features(sysd: pd.DataFrame) -> pd.DataFrame:
    """Features for a day-ahead load forecast.

    Load lags are 48 hours or more, so every feature is known when a forecast for the next day is issued
    the morning before. Temperature is the ERA5 reanalysis value for the hour, standing in for a day-ahead
    weather forecast; this makes the scores slightly optimistic (see the report).
    """
    y = sysd["ail_mw"]
    X = calendar(sysd.index)
    loc_date = sysd.index.tz_convert(TZ).date
    for city in ("calgary", "edmonton"):
        t = sysd[f"{city}_temp_c"]
        X[f"{city}_temp"] = t
        X[f"{city}_temp_daymean"] = t.groupby(loc_date).transform("mean")
        X[f"{city}_temp_lag24"] = t.shift(24)
    level = load_level(y)
    # Lags are expressed relative to the recent level, so the model learns shape, not size, and keeps
    # working as Alberta's load grows beyond anything in the training years.
    X["load_lag48_rel"] = y.shift(48) / level
    X["load_lag168_rel"] = y.shift(168) / level
    X["load_lag336_rel"] = y.shift(336) / level
    X["load_daymean_lag48_rel"] = y.shift(48).rolling(24).mean() / level
    return X


def load_level(y: pd.Series) -> pd.Series:
    """Average load over the week ending 48 hours before the target hour (known when the forecast is issued)."""
    return y.shift(48).rolling(168).mean()


def price_features(sysd: pd.DataFrame, with_aeso: bool) -> pd.DataFrame:
    """Features for an hour-ahead price forecast: everything is known one hour before the delivery hour."""
    p = sysd["pool_price"]
    X = calendar(sysd.index)
    for k in (1, 2, 3, 24, 168):
        X[f"price_lag{k}"] = p.shift(k)
    X["price_max24_lag1"] = p.shift(1).rolling(24).max()
    X["price_std24_lag1"] = p.shift(1).rolling(24).std()
    X["ail_lag1"] = sysd["ail_mw"].shift(1)
    X["ail_change_lag1"] = sysd["ail_mw"].diff().shift(1)
    X["wind_lag1"] = sysd["wind_mwh"].shift(1)
    X["solar_lag1"] = sysd["solar_mwh"].shift(1)
    X["wind_share_lag1"] = (sysd["wind_mwh"] / sysd["ail_mw"]).shift(1)
    X["wind_change_lag1"] = sysd["wind_mwh"].diff().shift(1)
    X["net_import_lag1"] = sysd["net_import_mw"].shift(1)
    X["thermal_lag1"] = sysd["thermal_mwh"].shift(1)
    X["calgary_temp"] = sysd["calgary_temp_c"]
    X["edmonton_temp"] = sysd["edmonton_temp_c"]
    if with_aeso:
        X["aeso_ha_forecast"] = sysd["aeso_ha_price_fc"]
    return X


# ----------------------------------------------------------------------------- models
def make_quantile_model(q: float, seed: int = 0, max_iter: int = 400) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, learning_rate=0.06, max_iter=max_iter,
                                         max_leaf_nodes=48, min_samples_leaf=60, l2_regularization=1.0,
                                         random_state=seed)


@dataclass
class QuantileForecaster:
    """One gradient-boosted model per quantile. With `scale`, the models learn y / scale and predictions
    are multiplied back, which lets trees follow a growing series they could not otherwise extrapolate."""
    quantiles: tuple = QUANTILES
    seed: int = 0
    max_iter: int = 400

    def fit(self, X: pd.DataFrame, y: pd.Series, scale: pd.Series | None = None) -> "QuantileForecaster":
        t = y if scale is None else y / scale
        self.models_ = {q: make_quantile_model(q, self.seed, self.max_iter).fit(X, t) for q in self.quantiles}
        self.columns_ = list(X.columns)
        self.adjust_ = {}
        return self

    def predict(self, X: pd.DataFrame, scale: pd.Series | None = None, adjusted: bool = True) -> pd.DataFrame:
        P = pd.DataFrame({q: m.predict(X[self.columns_]) for q, m in self.models_.items()}, index=X.index)
        if scale is not None:
            P = P.mul(scale.to_numpy(), axis=0)
        P = sort_quantiles(P)
        if adjusted:
            for (lo, hi), a in self.adjust_.items():
                P[lo] = P[lo] - a
                P[hi] = P[hi] + a
        return sort_quantiles(P)

    def conformalize(self, Xc: pd.DataFrame, yc: pd.Series, scale: pd.Series | None = None,
                     pairs=((0.1, 0.9), (0.25, 0.75))) -> dict:
        """Split conformal adjustment of each symmetric interval on a calibration set (CQR)."""
        self.adjust_ = {}
        P = self.predict(Xc, scale, adjusted=False)
        out = {}
        for lo, hi in pairs:
            alpha = 1 - (hi - lo)
            e = np.maximum(P[lo] - yc, yc - P[hi]).to_numpy()
            n = len(e)
            level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
            a = float(np.quantile(e, level, method="higher"))
            self.adjust_[(lo, hi)] = a
            out[f"{int(lo*100)}-{int(hi*100)}"] = a
        return out


def rolling_origin(X: pd.DataFrame, y: pd.Series, years: list[int], train_years: int, local_years: np.ndarray,
                   scale: pd.Series | None = None, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Retrain every year, as an operator would: for test year Y, train on the `train_years` years ending
    Y-2, calibrate the intervals on Y-1, then forecast Y. Returns raw and calibrated test quantiles."""
    raws, cals, log = [], [], []
    last = None
    ok = X.notna().all(axis=1) & y.notna() & (scale.notna() if scale is not None else True)
    for Y in years:
        tr = ok & (local_years >= Y - 1 - train_years) & (local_years <= Y - 2)
        ca = ok & (local_years == Y - 1)
        te = ok & (local_years == Y)
        if te.sum() == 0:
            continue
        sc = (lambda m: None) if scale is None else (lambda m: scale[m])
        m = QuantileForecaster(seed=seed).fit(X[tr], y[tr], sc(tr))
        raws.append(m.predict(X[te], sc(te), adjusted=False))
        adj = m.conformalize(X[ca], y[ca], sc(ca))
        cals.append(m.predict(X[te], sc(te)))
        log.append({"test_year": Y, "train": f"{Y-1-train_years} to {Y-2}", "calibration": str(Y - 1),
                    "adjustment": adj, "train_hours": int(tr.sum())})
        last = m
    rolling_origin.last_model = last
    return pd.concat(raws), pd.concat(cals), log


def sort_quantiles(P: pd.DataFrame) -> pd.DataFrame:
    """Rearrange so quantiles never cross (Chernozhukov et al., 2010)."""
    cols = sorted(P.columns)
    vals = np.sort(P[cols].to_numpy(), axis=1)
    return pd.DataFrame(vals, index=P.index, columns=cols)


def seasonal_naive(y: pd.Series, lag: int, train_mask: pd.Series, quantiles=QUANTILES) -> pd.DataFrame:
    """Same hour `lag` hours earlier as the median; spread from that method's own errors in training, by hour."""
    point = y.shift(lag)
    err = (y - point)[train_mask]
    hour = y.index.tz_convert(TZ).hour
    qtab = err.groupby(hour[train_mask]).quantile(list(quantiles)).unstack()
    P = pd.DataFrame({q: point.to_numpy() + qtab[q].reindex(hour).to_numpy() for q in quantiles}, index=y.index)
    return sort_quantiles(P)


# ----------------------------------------------------------------------------- scores
def pinball(y: np.ndarray, qhat: np.ndarray, q: float) -> float:
    d = np.asarray(y) - np.asarray(qhat)
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def scores(y: pd.Series, P: pd.DataFrame, mape: bool = True) -> dict:
    ok = y.notna() & P.notna().all(axis=1)
    y, P = y[ok], P[ok]
    med = P[0.5]
    out = {"n": int(len(y)),
           "mae": float(np.mean(np.abs(y - med))),
           "mape_pct": float(100 * np.mean(np.abs(y - med) / np.abs(y).clip(lower=1e-9))) if mape else None,
           "bias": float(np.mean(med - y)),
           "pinball_mean": float(np.mean([pinball(y, P[q], q) for q in P.columns]))}
    for lo, hi in ((0.1, 0.9), (0.25, 0.75)):
        if lo in P and hi in P:
            inside = (y >= P[lo]) & (y <= P[hi])
            out[f"coverage_{int(lo*100)}_{int(hi*100)}_pct"] = float(100 * inside.mean())
            out[f"width_{int(lo*100)}_{int(hi*100)}"] = float(np.mean(P[hi] - P[lo]))
    return out


def brier(y01: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y01)) ** 2))


def reliability(y01: np.ndarray, p: np.ndarray, bins=(0, .05, .1, .2, .3, .5, .7, 1.0001)) -> list[dict]:
    y01, p = np.asarray(y01), np.asarray(p)
    out = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (p >= a) & (p < b)
        if m.sum() >= 20:
            out.append({"bin": f"{a:.2f}-{min(b,1):.2f}", "n": int(m.sum()),
                        "forecast": float(p[m].mean()), "observed": float(y01[m].mean())})
    return out


def spike_classifier(seed: int = 0) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(learning_rate=0.06, max_iter=300, max_leaf_nodes=32,
                                          min_samples_leaf=80, l2_regularization=1.0, random_state=seed)
