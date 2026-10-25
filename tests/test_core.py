"""Unit tests for the forecasting, renewables, congestion, quality and agent tool code.

Tests that need the raw AESO files skip when data/raw is empty (as on GitHub Actions); the rest run on
synthetic data or on the published result files in outputs/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridanalytics import congestion as C, forecast as F, quality as Q, renewables as R  # noqa: E402
from gridanalytics import agent_tools as T, data as D  # noqa: E402

UTC_HOURS = pd.date_range("2022-01-01", periods=24 * 400, freq="h", tz="UTC")


# ----------------------------------------------------------------------------- forecasting
def test_pinball_known_values():
    y = np.array([10.0, 10.0])
    assert F.pinball(y, np.array([8.0, 12.0]), 0.5) == pytest.approx(1.0)
    # under-forecast by 2 at q=0.9 costs 1.8, over-forecast by 2 costs 0.2
    assert F.pinball(np.array([10.0]), np.array([8.0]), 0.9) == pytest.approx(1.8)
    assert F.pinball(np.array([10.0]), np.array([12.0]), 0.9) == pytest.approx(0.2)


def test_sort_quantiles_removes_crossing():
    P = pd.DataFrame({0.1: [5.0, 1.0], 0.5: [3.0, 2.0], 0.9: [4.0, 3.0]})
    S = F.sort_quantiles(P)
    assert (np.diff(S.to_numpy(), axis=1) >= 0).all()
    assert list(S.columns) == [0.1, 0.5, 0.9]


def test_seasonal_naive_uses_the_lag():
    y = pd.Series(np.arange(len(UTC_HOURS), dtype=float), index=UTC_HOURS)
    P = F.seasonal_naive(y, 168, pd.Series(True, index=UTC_HOURS))
    # y rises by 1 each hour, so the error of the same hour last week is always +168: the median is exact
    assert P[0.5].iloc[500] == pytest.approx(y.iloc[500])


def test_conformal_intervals_reach_nominal_coverage():
    rng = np.random.default_rng(1)
    n = 6000
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(2 * X.a + rng.normal(scale=1 + np.abs(X.b), size=n))
    tr, ca, te = slice(0, 3000), slice(3000, 4500), slice(4500, n)
    m = F.QuantileForecaster(quantiles=(0.1, 0.5, 0.9), max_iter=120).fit(X.iloc[tr], y.iloc[tr])
    m.conformalize(X.iloc[ca], y.iloc[ca], pairs=((0.1, 0.9),))
    P = m.predict(X.iloc[te])
    cover = ((y.iloc[te] >= P[0.1]) & (y.iloc[te] <= P[0.9])).mean()
    assert 0.76 <= cover <= 0.86


def test_scale_lets_trees_follow_growth():
    rng = np.random.default_rng(2)
    idx = pd.RangeIndex(4000)
    level = pd.Series(np.linspace(100, 200, 4000), index=idx)
    X = pd.DataFrame({"x": rng.uniform(size=4000)}, index=idx)
    y = level * (1 + 0.1 * X.x)
    m = F.QuantileForecaster(quantiles=(0.5,), max_iter=100).fit(X.iloc[:3000], y.iloc[:3000], level.iloc[:3000])
    pred = m.predict(X.iloc[3000:], level.iloc[3000:])[0.5]
    # level-relative model keeps up with growth beyond the training range
    assert np.mean(np.abs(pred - y.iloc[3000:]) / y.iloc[3000:]) < 0.02


def test_calendar_marks_alberta_holidays():
    idx = pd.DatetimeIndex(["2024-02-19 18:00", "2024-02-20 18:00"], tz="UTC")  # Family Day, then a Tuesday
    cal = F.calendar(idx)
    assert cal.holiday.tolist() == [1, 0]
    assert cal.hour.tolist() == [11, 11]   # 18:00 UTC is 11:00 Mountain Standard Time


def _toy_system(n=24 * 30):
    idx = pd.date_range("2024-01-01 07:00", periods=n, freq="h", tz="UTC")   # midnight in Alberta
    rng = np.random.default_rng(3)
    s = pd.DataFrame({"ail_mw": 9000 + 500 * np.sin(np.arange(n) / 24 * 2 * np.pi) + rng.normal(0, 50, n),
                      "pool_price": rng.uniform(10, 100, n), "wind_mwh": rng.uniform(0, 3000, n),
                      "solar_mwh": rng.uniform(0, 500, n), "net_import_mw": rng.uniform(-200, 200, n),
                      "thermal_mwh": rng.uniform(5000, 8000, n), "aeso_ha_price_fc": rng.uniform(10, 100, n),
                      "calgary_temp_c": rng.normal(0, 10, n), "edmonton_temp_c": rng.normal(0, 10, n)}, index=idx)
    return s


def test_features_do_not_leak_the_target():
    s = _toy_system()
    t = s.index[400]
    s2 = s.copy()
    s2.loc[t, ["ail_mw", "pool_price"]] = [99999.0, 999.0]
    a, b = F.load_features(s), F.load_features(s2)
    pa, pb = F.price_features(s, True), F.price_features(s2, True)
    assert a.loc[t].equals(b.loc[t])           # load at t never feeds its own forecast
    assert pa.loc[t].equals(pb.loc[t])         # price at t never feeds its own forecast
    # and the load level uses load at least 48 hours old
    la, lb = F.load_level(s.ail_mw), F.load_level(s2.ail_mw)
    assert la.loc[t + pd.Timedelta(hours=47)] == lb.loc[t + pd.Timedelta(hours=47)]
    assert la.loc[t + pd.Timedelta(hours=48)] != lb.loc[t + pd.Timedelta(hours=48)]


def test_brier_and_reliability():
    y = np.array([0, 0, 1, 1])
    assert F.brier(y, np.array([0, 0, 1, 1])) == 0
    assert F.brier(y, np.full(4, 0.5)) == pytest.approx(0.25)
    rel = F.reliability(np.r_[np.zeros(80), np.ones(20)], np.full(100, 0.2))
    assert rel[0]["observed"] == pytest.approx(0.2)


# ----------------------------------------------------------------------------- renewables
def test_capture_price_and_value_factor():
    price = pd.Series([10.0, 100.0])
    gen = pd.Series([3.0, 1.0])
    assert R.capture_price(gen, price) == pytest.approx(32.5)
    s = _toy_system()
    for g in ("wind", "solar", "storage"):
        s[f"{g}_cap_mw"] = 4000.0
    s["storage_mwh"] = 0.0
    a = R.annual_summary(s).iloc[0]
    assert a.wind_value_factor == pytest.approx(a.wind_capture / a.avg_price)


def test_ramps_and_wind_forecast_eval():
    s = _toy_system(24 * 400)
    s["wind_cap_mw"] = 3000.0
    r = R.ramps(s)
    assert (r.wind_ramp_down_p1_mw < 0).all() and (r.wind_ramp_up_p99_mw > 0).all()
    wf = pd.DataFrame({"time_utc": s.index, "wpf_ml": s.wind_mwh, "wpf_min": s.wind_mwh - 10, "wpf_max": s.wind_mwh + 10})
    tab, extra = R.wind_forecast_eval(wf, s)
    assert tab.mae_mw.max() == pytest.approx(0.0)
    assert extra["overall_band_coverage_pct"] == pytest.approx(100.0)


# ----------------------------------------------------------------------------- congestion
def test_constrained_energy_and_curtailment():
    s = _toy_system()
    c = pd.DataFrame({"time_utc": s.index[:10], "fuel": "WIND", "planning_area": 4, "constrained_mw": 50.0, "minutes": 60})
    ce = C.constrained_energy(c)
    assert ce.mwh.sum() == pytest.approx(500.0)
    cr = C.curtailment_rate(ce, s, [2024]).set_index("group")
    gen = s.wind_mwh.sum()
    assert cr.loc["Wind", "curtailment_pct"] == pytest.approx(100 * 500 / (gen + 500))
    pa = C.by_planning_area(ce, [2024])
    assert pa.Total.iloc[0] == pytest.approx(0.5)   # GWh


def test_intertie_binding_share():
    t = pd.date_range("2018-01-01", periods=4, freq="h")
    it = pd.DataFrame({"time_local": t, "path": "BC_MATL_IMPORT", "sched_mw": [500, 100, 600, 0],
                       "atc_mw": [500, 500, 600, 500], "ttc_mw": [1000] * 4, "limit_reason": ["A", "A", "B", None]})
    tab, reasons = C.intertie_limits(it)
    assert tab.import_at_limit_pct.iloc[0] == pytest.approx(50.0)
    assert tab.atc_share_pct.iloc[0] == pytest.approx(52.5)
    assert reasons.hours.sum() == 4


# ----------------------------------------------------------------------------- quality
def test_continuity_and_stuck_values():
    idx = pd.DatetimeIndex(list(UTC_HOURS[:5]) + list(UTC_HOURS[6:10]) + [UTC_HOURS[9]])
    c = Q.hourly_continuity(idx)
    assert c["duplicates"] == 1 and c["gaps"] >= 1
    s = pd.Series([1.0] * 30 + [2.0, 3.0] + [0.0] * 40)
    assert Q.stuck_values(s, 24) == 1     # zeros do not count as stuck
    assert Q.price_checks(pd.Series([0.0, 50.0, 999.99]))["at_cap_hours"] == 1


# ----------------------------------------------------------------------------- published outputs and agent tools
def test_published_results_are_consistent():
    fr = json.loads((ROOT / "outputs" / "forecast_results.json").read_text())
    L, P = fr["load"]["headline"], fr["price"]["headline"]
    assert L["mape_gbm_pct"] < L["mape_naive_pct"]
    assert 75 <= L["coverage_cal_pct"] <= 88
    assert P["mae_gbm_aeso"] < P["mae_aeso"]
    mk = json.loads((ROOT / "outputs" / "market_results.json").read_text())
    years = [r["year"] for r in mk["annual"]]
    assert years == list(range(2010, 2026))


def test_planning_area_map_covers_constraint_areas():
    gj = json.loads((ROOT / "static_data" / "planning_areas.geojson").read_text())
    ids = {int(f["properties"]["Area_ID"]) for f in gj["features"]}
    assert len(ids) == 42
    mk = json.loads((ROOT / "outputs" / "market_dashboard.json").read_text())
    for y, rows in mk["pa"].items():
        assert {int(r["planning_area"]) for r in rows} <= ids


def test_agent_tools_answer_from_the_tables():
    one = T.constrained_energy(2025, "Medicine Hat")
    top = T.constrained_energy(2025)
    assert one["planning_area"] == 4 and one["total_gwh"] == pytest.approx(top["top_areas"][0]["total_gwh"])
    h = T.forecast_at("2024-01-13 18:00")
    assert h["load"]["q10"] <= h["load"]["q50"] <= h["load"]["q90"]
    assert 0 <= h["price"]["spike_probability"] <= 1
    assert T.forecast_at("2019-06-01 12:00")["load"] is None
    assert T.asset_value("TVS1")["group"] == "Solar"
    with pytest.raises(ValueError):
        T.constrained_energy(2019)
    assert set(T.grid_summary()) >= {"load_forecast", "price_forecast", "congestion"}


def test_mcp_server_registers_every_tool():
    pytest.importorskip("mcp")
    import asyncio
    from gridanalytics.mcp_server import mcp
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert names == {fn.__name__ for fn in T.TOOLS}


@pytest.mark.skipif(not (D.RAW / D.FILES["metered_2020_2025"][0]).exists(), reason="raw AESO data not downloaded")
def test_raw_files_match_published_checksums():
    for key, (name, _, sha) in D.FILES.items():
        assert Q.sha256(D.RAW / name) == sha, name
