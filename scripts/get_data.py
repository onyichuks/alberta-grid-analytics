"""Download the public AESO files into data/raw and check them against their published SHA-256 checksums.

Usage:  python scripts/get_data.py
If a download is blocked on your network, save the files from the URLs below into data/raw by hand;
the checksum check runs either way. The weather file and the current asset list are built from public
pages rather than single downloads; see static_data/README.md for how to rebuild them.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridanalytics import data as D, quality as Q  # noqa: E402


def main() -> None:
    D.RAW.mkdir(parents=True, exist_ok=True)
    bad = []
    for key, (name, url, sha) in D.FILES.items():
        p = D.RAW / name
        if not p.exists():
            print(f"downloading {name}")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "alberta-grid-analytics"})
                with urllib.request.urlopen(req, timeout=300) as r, open(p, "wb") as f:
                    f.write(r.read())
            except Exception as e:  # noqa: BLE001
                print(f"  could not download ({e}); save it by hand from {url}")
                continue
        ok = Q.sha256(p) == sha
        print(f"{'ok      ' if ok else 'MISMATCH'} {name}")
        if not ok:
            bad.append(name)
    for extra in (D.WEATHER_FILE, D.CSD_CURRENT_FILE):
        if not (D.RAW / extra).exists():
            print(f"missing {extra}: copy it from static_data/ (see static_data/README.md)")
    if bad:
        print("Checksum mismatch: AESO may have revised these files. Results may differ from the published ones.")


if __name__ == "__main__":
    main()
