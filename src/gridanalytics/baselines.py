"""Simpler models used to judge whether the gradient-boosted forecasts earn their complexity.

* Linear quantile regression for load, built like the "vanilla" benchmark of the Global Energy Forecasting
  Competitions (Hong, 2010; Hong et al., 2014): hour by day type, month, and a temperature polynomial with
  month and hour interactions, plus the same recent-load inputs the boosted model gets.
* Linear quantile regression for price on the boosted price model's inputs, with hour of day as categories and
  without day of week and month.
* Logistic regression for the spike warning.

Each linear quantile model is fitted per quantile (statsmodels; if its iterative solver does not converge, the
exact linear programme is solved with HiGHS instead) and then calibrated with the same conformal step as the
boosted models, so the comparison isolates the model, not the calibration.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .forecast import QUANTILES, QuantileForecaster, sort_quantiles


def _dummies(s: pd.Series, prefix: str) -> pd.DataFrame:
    return pd.get_dummies(s.astype(int), prefix=prefix, drop_first=True, dtype=float)


def linear_load_design(X: pd.DataFrame) -> pd.DataFrame:
    """Vanilla-style linear design built from the boosted model's load features. It averages the two cities'
    temperatures and leaves out the 24-hour lagged temperature, the day-of-year terms and day of week (beyond
    the weekend and holiday flag)."""
    t = (X["calgary_temp"] + X["edmonton_temp"]) / 2
    td = (X["calgary_temp_daymean"] + X["edmonton_temp_daymean"]) / 2
    ts = t / 10  # scaled so the cubic terms stay well conditioned
    daytype = X["weekend_or_holiday"].astype(int)
    D = pd.DataFrame(index=X.index)
    D = D.join(_dummies(X["hour"] * 2 + daytype, "hd")).join(_dummies(X["month"], "m"))
    hour = _dummies(X["hour"], "h")
    month = _dummies(X["month"], "m")
    for k in (1, 2, 3):
        D[f"t{k}"] = ts ** k
        D = D.join(hour.mul(ts ** k, axis=0).add_suffix(f"_t{k}"))
        D = D.join(month.mul(ts ** k, axis=0).add_suffix(f"_t{k}"))
    D["tday"] = td / 10
    D["tday2"] = (td / 10) ** 2
    for c in [c for c in X.columns if c.endswith("_rel")]:
        D[c] = X[c]
    D.insert(0, "const", 1.0)
    return D


def linear_price_design(X: pd.DataFrame) -> pd.DataFrame:
    """Linear design for the hour-ahead price: the boosted model's inputs with hour of day as dummies in place of
    the numeric hour, and without day of week and month."""
    D = X.drop(columns=["hour", "dow", "month"]).copy()
    D = D.join(_dummies(X["hour"], "h"))
    D.insert(0, "const", 1.0)
    return D.astype(float)


@dataclass
class LinearQuantileForecaster(QuantileForecaster):
    """Linear quantile regression per quantile, with the same `scale`, sorting and conformal step as the
    boosted forecaster (it reuses `predict` and `conformalize` from the parent class).

    Each quantile is fitted first by iteratively reweighted least squares (statsmodels). If that stops at its
    iteration limit without converging, which happens in the tails of heavy-tailed prices, the same problem is
    solved exactly as a linear programme (HiGHS, through scikit-learn), so no baseline is left half-fitted.
    `solver_` records which method fitted each quantile."""

    irls_max_iter: int = 3000

    def fit(self, X: pd.DataFrame, y: pd.Series, scale: pd.Series | None = None) -> "LinearQuantileForecaster":
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import QuantileRegressor
        from statsmodels.regression.quantile_regression import QuantReg
        from statsmodels.tools.sm_exceptions import IterationLimitWarning
        t = (y if scale is None else y / scale).to_numpy()
        A = X.to_numpy(dtype=float)
        self.models_, self.solver_ = {}, {}
        for q in self.quantiles:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                res = QuantReg(t, A).fit(q=q, max_iter=self.irls_max_iter, p_tol=1e-6)
            if any(issubclass(w.category, IterationLimitWarning) for w in caught):
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)  # refuse a failed exact solve too
                    lp = QuantileRegressor(quantile=q, alpha=0.0, fit_intercept=False, solver="highs").fit(A, t)
                self.models_[q] = _Linear(lp.coef_)
                self.solver_[q] = "exact linear programme (HiGHS)"
            else:
                self.models_[q] = _Linear(res.params)
                self.solver_[q] = f"iteratively reweighted least squares ({int(res.iterations)} iterations)"
        self.columns_ = list(X.columns)
        self.adjust_ = {}
        return self


class _Linear:
    def __init__(self, beta: np.ndarray):
        self.beta = np.asarray(beta)

    def predict(self, X) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.beta


def rolling_origin_linear(D: pd.DataFrame, y: pd.Series, years: list[int], train_years: int,
                          local_years: np.ndarray, scale: pd.Series | None = None):
    """Same retraining and calibration scheme as forecast.rolling_origin, with linear quantile models.
    The solver used for each year and quantile is kept in `rolling_origin_linear.log_`."""
    raws, cals, log = [], [], []
    ok = D.notna().all(axis=1) & y.notna() & (scale.notna() if scale is not None else True)
    sc = (lambda m: None) if scale is None else (lambda m: scale[m])
    for Y in years:
        tr = ok & (local_years >= Y - 1 - train_years) & (local_years <= Y - 2)
        ca = ok & (local_years == Y - 1)
        te = ok & (local_years == Y)
        m = LinearQuantileForecaster().fit(D[tr], y[tr], sc(tr))
        raws.append(m.predict(D[te], sc(te), adjusted=False))
        m.conformalize(D[ca], y[ca], sc(ca))
        cals.append(m.predict(D[te], sc(te)))
        log.append({"test_year": Y, "solver": {str(q): v for q, v in m.solver_.items()}})
    rolling_origin_linear.log_ = log
    return pd.concat(raws), pd.concat(cals)


def logistic_spike_model() -> object:
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))


def grouped_importance(predict, X: pd.DataFrame, y: pd.Series, groups: dict[str, list[str]],
                       repeats: int = 3, seed: int = 0) -> list[dict]:
    """Permutation importance by feature group: shuffle every column in a group together (same row order),
    so correlated inputs such as Calgary and Edmonton temperature are judged as one."""
    rng = np.random.default_rng(seed)
    base = float(np.mean(np.abs(predict(X) - y)))
    out = []
    for name, cols in groups.items():
        inc = []
        for _ in range(repeats):
            Xp = X.copy()
            order = rng.permutation(len(Xp))
            Xp[cols] = Xp[cols].to_numpy()[order]
            inc.append(float(np.mean(np.abs(predict(Xp) - y))) - base)
        out.append({"group": name, "columns": cols, "mae_increase": float(np.mean(inc)),
                    "mae_increase_pct": 100 * float(np.mean(inc)) / base if base > 0 else float("nan")})
    return sorted(out, key=lambda r: -r["mae_increase"])


def segment_scores(y: pd.Series, P: pd.DataFrame, seg: pd.Series, mape: bool = True) -> pd.DataFrame:
    """Error, bias and 80 percent coverage by segment (hour, month, day type, temperature band...)."""
    if not seg.index.equals(y.index):
        before = int(seg.isna().sum())
        seg = seg.reindex(y.index)  # align by index, never by position
        if int(seg.isna().sum()) > before:
            raise ValueError("segment labels do not cover every row of y; check that their indexes match")
    e = P[0.5] - y
    # keep the natural order of banded segments (temperature or price bands), not alphabetical order
    sv = (pd.Categorical(seg.to_numpy(), categories=seg.cat.categories, ordered=True)
          if isinstance(seg.dtype, pd.CategoricalDtype) else seg.to_numpy())
    d = pd.DataFrame({"abs": e.abs(), "err": e, "pct": e.abs() / y.abs().clip(lower=1e-9) * 100,
                      "inside": ((y >= P[0.1]) & (y <= P[0.9])) * 100.0, "seg": sv}, index=y.index)
    g = d.groupby("seg", observed=True)
    out = pd.DataFrame({"hours": g.size(), "mae": g["abs"].mean(), "bias": g["err"].mean(),
                        "coverage_80_pct": g["inside"].mean()})
    if mape:
        out.insert(1, "mape_pct", g["pct"].mean())
    return out


__all__ = ["linear_load_design", "linear_price_design", "LinearQuantileForecaster", "rolling_origin_linear",
           "logistic_spike_model", "grouped_importance", "segment_scores", "QUANTILES", "sort_quantiles"]
