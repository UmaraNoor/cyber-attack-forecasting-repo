"""
acquire_data.py
===============
Obtain the two upstream sources needed to rebuild the unified corpus.

    VERIS Community Database   fetched automatically from its public repository
    Hackmageddon timelines     MUST be obtained by you, from the official site

This repository does not redistribute either source. VCDB is fetched at run
time from the maintainers' own repository under their licence; Hackmageddon
timelines are published on the author's site under terms this project makes no
claim about, so they are not downloaded, scraped or mirrored here. You are
responsible for obtaining them and for complying with the applicable terms.

Usage:
    python acquire_data.py                 # fetch VCDB, report what is missing
    python acquire_data.py --check         # report status only, fetch nothing
"""

import argparse
import io
import json
import sys
import zipfile
from datetime import date
from pathlib import Path

import requests

from paths import DATA

RAW = DATA / "raw"
VCDB_DIR = RAW / "vcdb"
HACK_DIR = RAW / "hackmageddon"

# Pinned to the VCDB commit used to build the published corpus (master branch on 17 December 2024).
VCDB_COMMIT = "becd8c62905ea2330451cd335e0fefe8ac9f3be2"
VCDB_ZIP = f"https://github.com/vz-risk/VCDB/archive/{VCDB_COMMIT}.zip"
VCDB_HOME = "https://github.com/vz-risk/VCDB"
HACK_HOME = "https://www.hackmageddon.com/"

HACK_INSTRUCTIONS = f"""
  Hackmageddon timelines are NOT downloaded by this script.

  To obtain them:

    1. Visit {HACK_HOME}
    2. Locate the timeline archives covering 2018-2023. The site publishes
       these as periodic posts, each with an accompanying spreadsheet.
    3. Review the site's terms of use and confirm that your intended use is
       permitted. This project makes no representation about those terms.
    4. Save the files, unmodified, into:

           {HACK_DIR}

    5. Record the access date in data/raw/ACCESS_LOG.json (this script will
       create a template for you).

  The corpus reported in the paper was built from timelines accessed on
  7 August 2023. Timelines published after that date will produce a corpus
  that differs from the published one; validate_corpus.py will tell you by
  how much.
"""


def fetch_vcdb() -> bool:
    """Download and extract the VERIS Community Database JSON records."""
    VCDB_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(VCDB_DIR.rglob("*.json"))
    if existing:
        print(f"  VCDB already present: {len(existing):,} JSON records in {VCDB_DIR}")
        return True

    print(f"  fetching VCDB from {VCDB_ZIP} ...")
    try:
        r = requests.get(VCDB_ZIP, timeout=180)
        r.raise_for_status()
    except Exception as exc:                                  # noqa: BLE001
        print(f"  FAILED: {exc}")
        print(f"  Fetch manually from {VCDB_HOME} and unpack into {VCDB_DIR}")
        return False

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        members = [m for m in z.namelist() if m.endswith(".json") and "/data/json/" in m]
        for m in members:
            target = VCDB_DIR / Path(m).name
            with z.open(m) as src, open(target, "wb") as dst:
                dst.write(src.read())
    print(f"  extracted {len(members):,} VCDB records to {VCDB_DIR}")
    return True


def check_hackmageddon() -> bool:
    HACK_DIR.mkdir(parents=True, exist_ok=True)
    files = [p for p in HACK_DIR.iterdir()
             if p.suffix.lower() in {".xlsx", ".xls", ".csv"}]
    if files:
        print(f"  Hackmageddon: {len(files)} file(s) found in {HACK_DIR}")
        for p in sorted(files)[:8]:
            print(f"    - {p.name}")
        if len(files) > 8:
            print(f"    ... and {len(files) - 8} more")
        return True
    print("  Hackmageddon: NOT PRESENT")
    print(HACK_INSTRUCTIONS)
    return False


def write_access_log_template() -> None:
    log = RAW / "ACCESS_LOG.json"
    if log.exists():
        return
    log.write_text(json.dumps({
        "note": "Record when and from where each source was obtained. "
                "Required for citation and for interpreting any divergence "
                "from the published corpus statistics.",
        "veris_vcdb": {
            "source": VCDB_HOME,
            "accessed": str(date.today()),
            "licence": "see upstream repository",
        },
        "hackmageddon": {
            "source": HACK_HOME,
            "accessed": "FILL IN",
            "coverage": "2018-2023",
            "terms_reviewed": False,
            "note": "Published corpus used timelines accessed 2023-08-07.",
        },
    }, indent=2), encoding="utf8")
    print(f"  wrote access-log template to {log}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report status only; do not download anything")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    print("Upstream sources")
    print("-" * 60)

    ok_v = check_vcdb_only() if args.check else fetch_vcdb()
    ok_h = check_hackmageddon()
    write_access_log_template()

    print("-" * 60)
    if ok_v and ok_h:
        print("Both sources present. Next: python build_corpus.py")
        return 0
    print("One or more sources missing. The corpus cannot be rebuilt until")
    print("both are available. See the instructions above.")
    return 1


def check_vcdb_only() -> bool:
    n = len(list(VCDB_DIR.rglob("*.json"))) if VCDB_DIR.exists() else 0
    print(f"  VCDB: {n:,} JSON records" if n else "  VCDB: NOT PRESENT")
    return n > 0


if __name__ == "__main__":
    sys.exit(main())
