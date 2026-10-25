"""Unit tests for the baseline models and diagnostics used in docs/ML_TECHNICAL_NOTE.md."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridanalytics import baselines as B, forecast as F  # noqa: E402


def test_linear_quantile_regression_recovers_known_quantiles():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.uniform(0, 10, n)
    y = pd.Series(2.0 * x + rng.uniform(-1, 1, n))
    X = pd.DataFrame({"const": 1.0, "x": x})
    m = B.LinearQuantileForecaster(quantiles=(0.1, 0.5, 0.9)).fit(X, y)
    assert m.models_[0.5].beta[1] == pytest.approx(2.0, abs=0.05)
    # uniform noise on [-1, 1]: the 10th and 90th percentiles sit 0.8 below and above the line
    assert m.models_[0.1].beta[0] == pytest.approx(-0.8, abs=0.1)
    assert m.models_[0.9].beta[0] == pytest.approx(0.8, abs=0.1)
    P = m.predict(X)
    assert (np.diff(P.to_numpy(), axis=1) >= 0).all()


def test_linear_load_design_has_no_gaps():
    idx = pd.date_range("2023-01-01", periods=24 * 30, freq="h", tz="UTC")
    X = F.calendar(idx)
    for city in ("calgary", "edmonton"):
        X[f"{city}_temp"] = np.linspace(-20, 10, len(idx))
        X[f"{city}_temp_daymean"] = 0.0
        X[f"{city}_temp_lag24"] = 0.0
    for c in ("load_lag48_rel", "load_lag168_rel", "load_lag336_rel", "load_daymean_lag48_rel"):
        X[c] = 1.0
    Dm = B.linear_load_design(X)
    assert Dm.notna().all().all()
    assert "const" in Dm and "load_lag48_rel" in Dm
    assert Dm.shape[0] == len(idx)


def test_grouped_importance_finds_the_signal():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=2000), "b": rng.normal(size=2000), "c": rng.normal(size=2000)})
    y = 3 * X["a"] + rng.normal(0, 0.1, 2000)
    imp = B.grouped_importance(lambda Z: 3 * Z["a"].to_numpy(), X, y, {"signal": ["a"], "noise": ["b", "c"]})
    assert imp[0]["group"] == "signal" and imp[0]["mae_increase"] > 1
    assert imp[1]["mae_increase"] == pytest.approx(0.0, abs=1e-12)


def test_segment_scores_split_by_group():
    y = pd.Series([100.0, 100.0, 200.0, 200.0])
    P = pd.DataFrame({0.1: [90, 90, 150, 150], 0.5: [110, 90, 200, 220], 0.9: [120, 120, 250, 210]}, dtype=float)
    s = B.segment_scores(y, P, pd.Series(["x", "x", "y", "y"]))
    assert s.loc["x", "mae"] == pytest.approx(10.0)
    assert s.loc["y", "mape_pct"] == pytest.approx(5.0)
    assert s.loc["x", "coverage_80_pct"] == pytest.approx(100.0)
    # banded segments keep their natural order, not alphabetical order
    band = pd.Series(pd.Categorical(["low", "low", "high", "high"], categories=["low", "high"], ordered=True))
    assert list(B.segment_scores(y, P, band).index) == ["low", "high"]
    # labels are matched to rows by index, not by position
    shuffled = pd.Series(["y", "y", "x", "x"], index=[3, 2, 1, 0])
    assert B.segment_scores(y, P, shuffled).loc["x", "mae"] == pytest.approx(10.0)


def test_published_ml_results_are_consistent():
    p = ROOT / "outputs" / "ml_results.json"
    if not p.exists():
        pytest.skip("run run_ml_analysis.py first")
    R = json.loads(p.read_text())
    forecast = json.loads((ROOT / "outputs" / "forecast_results.json").read_text())
    sc = {(r["model"], r["period"]): r for r in R["load"]["scores"]}
    # the boosted model in the ML note is the same model as in the report
    assert sc[("Gradient boosting, conformal", "all test")]["mape_pct"] == pytest.approx(
        forecast["load"]["headline"]["mape_gbm_pct"], rel=1e-6)
    lc = {r["train_years"]: r for r in R["load"]["learning_curve"]}
    assert lc[7]["mape_pct"] == pytest.approx(forecast["load"]["headline"]["mape_gbm_pct"], rel=1e-6)
    for m in R["price"]["spike"]["models"].values():
        assert 0.5 < m["roc_auc"] <= 1.0
    # the figures quoted in docs/ML_TECHNICAL_NOTE.md
    assert R["load"]["linear_design_columns"] == 170
    lin = sc[("Linear quantile regression, conformal", "all test")]
    assert round(lin["mape_pct"], 2) == 1.68
    psc = {(r["model"], r["period"]): r for r in R["price"]["scores"]}
    assert round(psc[("Linear quantile regression, conformal", "all test")]["mae"], 2) == 17.42
    assert round(psc[("Gradient boosting, conformal", "all test")]["mae"], 2) == 13.11
    assert round(R["price"]["spike"]["models"]["Gradient boosting"]["brier_skill_pct"]) == 69


def test_linear_quantile_falls_back_to_the_exact_solver():
    rng = np.random.default_rng(2)
    n = 600
    x = rng.uniform(0, 10, n)
    y = pd.Series(1.0 + 0.5 * x + rng.standard_t(2, n))
    X = pd.DataFrame({"const": 1.0, "x": x})
    exact = B.LinearQuantileForecaster(quantiles=(0.9,), irls_max_iter=1).fit(X, y)  # forced to stop early
    assert exact.solver_[0.9].startswith("exact")
    irls = B.LinearQuantileForecaster(quantiles=(0.9,)).fit(X, y)
    assert irls.solver_[0.9].startswith("iteratively")
    # both reach the same minimum of the pinball loss
    lx = F.pinball(y, exact.models_[0.9].predict(X), 0.9)
    li = F.pinball(y, irls.models_[0.9].predict(X), 0.9)
    assert lx <= li + 1e-6 and li == pytest.approx(lx, rel=1e-3)
