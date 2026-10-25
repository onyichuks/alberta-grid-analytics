"""Static figures for the README and report (figures/*.png), drawn from the analysis outputs.

Usage:  python scripts/make_figures.py
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, PowerNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
TZ = ZoneInfo("America/Edmonton")
BLUE, ORANGE, AMBER, INK, INK2, LINE = "#2a78d6", "#eb6834", "#eda100", "#0f1a24", "#475665", "#d6dee5"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": LINE, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": LINE, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})


def series_window(s: dict, start: str, days: int):
    t0 = datetime.strptime(s["t0_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    times = [(t0 + timedelta(hours=i)).astimezone(TZ) for i in range(len(s["actual"]))]
    a = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=TZ)
    b = a + timedelta(days=days)
    idx = [i for i, t in enumerate(times) if a <= t < b]
    return [times[i].replace(tzinfo=None) for i in idx], idx


def fan(ax, t, s, idx, ylabel):
    g = lambda k: [s[k][i] for i in idx]  # noqa: E731
    ax.fill_between(t, g("q10"), g("q90"), color=BLUE, alpha=0.15, lw=0, label="10 to 90% range")
    ax.fill_between(t, g("q25"), g("q75"), color=BLUE, alpha=0.28, lw=0, label="25 to 75% range")
    ax.plot(t, g("q50"), color=BLUE, lw=1.8, label="Forecast (median)")
    ax.plot(t, g("actual"), color=INK, lw=1.2, label="Actual")
    ax.set_ylabel(ylabel)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    fd = json.loads((OUT / "forecast_dashboard.json").read_text())
    fr = json.loads((OUT / "forecast_results.json").read_text())
    mk = json.loads((OUT / "market_dashboard.json").read_text())
    mr = json.loads((OUT / "market_results.json").read_text())

    # 1. Load forecast, January 2024 cold snap
    t, idx = series_window(fd["load"], "2024-01-08", 7)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    fan(ax, t, fd["load"], idx, "Alberta Internal Load (MW)")
    h = fr["load"]["headline"]
    ax.set_title(f"Day-ahead load forecast, 8 to 14 January 2024 (2023 to 2025: {h['mape_gbm_pct']:.1f}% error, "
                 f"{h['coverage_cal_pct']:.0f}% of hours inside the 80% range)", fontsize=10, loc="left", color=INK)
    ax.legend(loc="lower right", fontsize=8, frameon=False, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG / "fig01_load_forecast.png")
    plt.close(fig)

    # 2. Price forecast and spike probability, same week
    t, idx = series_window(fd["price"], "2024-01-08", 7)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 4.8), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    fan(a1, t, fd["price"], idx, "Pool price (\\$/MWh)")
    a1.plot(t, [fd["price"]["aeso"][i] for i in idx], color=ORANGE, lw=1, ls=":", label="AESO hour-ahead forecast")
    a1.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
    p = fr["price"]["headline"]
    a1.set_title(f"Hour-ahead price forecast (2024 to 2025: mean error \\${p['mae_gbm_aeso']:.1f} against "
                 f"\\${p['mae_aeso']:.1f}/MWh for AESO's forecast)", fontsize=10, loc="left", color=INK)
    a2.bar(t, [100 * fd["price"]["p_spike"][i] for i in idx], width=1 / 24, color=ORANGE)
    a2.set_ylabel("P(price ≥ \\$200)\n(%)")
    a2.set_ylim(0, 100)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG / "fig02_price_forecast.png")
    plt.close(fig)

    # 3. Wind and solar share and value factor
    A = [r for r in mk["annual"]]
    yrs = [r["year"] for r in A]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    a1.bar(yrs, [r["wind_share_pct"] for r in A], color=BLUE, label="Wind")
    a1.bar(yrs, [r["solar_share_pct"] for r in A], bottom=[r["wind_share_pct"] for r in A], color=AMBER, label="Solar")
    a1.set_title("Share of Alberta Internal Load (%)", fontsize=10, loc="left", color=INK)
    a1.legend(frameon=False, fontsize=8)
    a2.plot(yrs, [r["wind_value_factor"] for r in A], color=BLUE, marker="o", ms=4, label="Wind")
    a2.plot(yrs, [r["solar_value_factor"] for r in A], color=AMBER, marker="o", ms=4, label="Solar")
    a2.axhline(1, color=INK2, lw=0.8, ls=":")
    a2.set_ylim(0, None)
    a2.set_title("Value factor (capture price / average price)", fontsize=10, loc="left", color=INK)
    a2.legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.01, "2025 covers January to July.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "fig03_wind_solar_value.png")
    plt.close(fig)

    # 4. Constrained energy by planning area, 2025
    pa = {int(r["planning_area"]): r["Total"] for r in mk["pa"]["2025"]}
    polys, vals, names = [], [], []
    for f in mk["geo"]["features"]:
        g = f["geometry"]
        rings = g["coordinates"] if g["type"] == "Polygon" else [p[0] for p in g["coordinates"]]
        aid = int(f["properties"]["Area_ID"])
        for r in rings:
            polys.append([(x, y * 1.7) for x, y in r])
            vals.append(pa.get(aid, 0.0))
        names.append((aid, " ".join(f["properties"]["NAME"].split()), rings[0]))
    cmap = LinearSegmentedColormap.from_list("seq", ["#eaf0f4", "#cde2fb", "#3987e5", "#0d366b"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4.6), gridspec_kw={"width_ratios": [1, 1.3]})
    pc = PolyCollection(polys, array=vals, cmap=cmap, norm=PowerNorm(0.5, vmin=0, vmax=max(vals)), edgecolor="white", linewidth=0.5)
    a1.add_collection(pc)
    a1.autoscale_view()
    a1.set_aspect("equal")
    a1.axis("off")
    cb = fig.colorbar(pc, ax=a1, shrink=0.6, pad=0.02)
    cb.set_label("GWh constrained, 2025", color=INK2)
    top = sorted(mk["pa"]["2025"], key=lambda r: -r["Total"])[:8][::-1]
    nm = {aid: n for aid, n, _ in names}
    a2.barh([nm.get(int(r["planning_area"]), str(r["planning_area"])) for r in top], [r["Total"] for r in top], color=BLUE)
    a2.set_xlabel("GWh")
    a2.grid(axis="y", visible=False)
    c = mr["congestion"]
    a2.set_title(f"Generation held back by transmission limits, 2025\n({c['constrained_gwh']['2025']:,.0f} GWh in total; "
                 f"{c['south_share_2025_pct']:.0f}% in the South)", fontsize=10, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig04_constraints_2025.png")
    plt.close(fig)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
