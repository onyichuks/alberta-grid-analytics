"""Build the interactive dashboard (docs/index.html) from the analysis outputs.

Usage:  python scripts/build_dashboard.py [--repo-url URL]
Run python scripts/check_data.py, python run_forecast.py and python run_market.py first.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridanalytics.outputs import to_json  # noqa: E402

DEFAULT_REPO = "https://github.com/onyichuks/alberta-grid-analytics"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-url", default=DEFAULT_REPO)
    args = ap.parse_args()
    out_dir = ROOT / "outputs"
    data = {
        "fc": json.loads((out_dir / "forecast_dashboard.json").read_text()),
        "fr": json.loads((out_dir / "forecast_results.json").read_text()),
        "mk": json.loads((out_dir / "market_dashboard.json").read_text()),
        "dq": json.loads((out_dir / "data_quality.json").read_text()),
    }
    html = (ROOT / "dashboard" / "template.html").read_text(encoding="utf-8")
    payload = to_json(data, compact=True).replace("</", "<\\/")
    html = html.replace("/*__DATA__*/null", payload)
    html = html.replace("{{REPO_URL}}", args.repo_url.rstrip("/"))
    html = html.replace("{{REPO_LABEL}}", args.repo_url.rstrip("/").replace("https://", ""))
    out = ROOT / "docs" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    (ROOT / "docs" / ".nojekyll").write_text("")
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
