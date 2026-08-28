"""
build_corpus.py
===============
Rebuild the unified corpus from the two upstream sources.

    data/raw/vcdb/*.json          VERIS Community Database  (1971-2017)
    data/raw/hackmageddon/*.xlsx  Hackmageddon timelines    (2018-2023)
        |
        v
    data/derived/unified_corpus.csv   20,279 x 64

This implements the harmonisation described in Section 3 of the paper:
field-name alignment, temporal decomposition, country-name expansion,
sentinel filling, and cross-source attribute reconstruction by Random Forest.

--------------------------------------------------------------------------
IMPORTANT - READ BEFORE RELYING ON THIS SCRIPT
--------------------------------------------------------------------------
This is a reimplementation written from the methodology as documented, not
the original authors' notebook, which was not retained in runnable form. It
has NOT been executed against raw upstream files, because the raw files were
not available to whoever wrote this module - only the already-merged corpus
was. It is therefore a specification of the procedure that should be treated
as untested until someone runs it end to end.

Run validate_corpus.py afterwards. It compares whatever this produces against
the published corpus statistics and will tell you where the rebuild diverges.
Divergence is expected if your Hackmageddon timelines were accessed on a
different date; divergence in schema or in the VERIS-era row count is a bug
and should be reported as an issue.
--------------------------------------------------------------------------

Usage:
    python build_corpus.py
    python build_corpus.py --skip-imputation   # intersection schema only
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from paths import DATA

RAW = DATA / "raw"
DERIVED = DATA / "derived"
OUT = DERIVED / "unified_corpus.csv"

# ---- schema ---------------------------------------------------------------
DATE_COLS = ["timeline.incident.day", "timeline.incident.month",
             "timeline.incident.year"]
KEY_COLS = ["victim.victim_id", "victim.industry.name", "victim.country"]
TARGETS = ["Attack", "pattern"]

BINARY_COLS = [
    "action.Malware", "action.Hacking", "action.Social", "action.Physical",
    "action.Misuse", "action.Error", "action.Environmental", "action.ABC",
    "attribute.Confidentiality", "attribute.Integrity", "attribute.Availability",
    "actor.External", "actor.Internal", "actor.Partner", "actor.ABC",
    "asset.variety.Server", "asset.variety.Network", "asset.variety.User Dev",
    "asset.variety.Media", "asset.variety.Person", "asset.variety.Kiosk/Term",
    "asset.variety.ABC", "asset.variety.Embedded",
]
for _who in ("external", "internal", "partner"):
    for _m in ("Convenience", "Espionage", "Fear", "Financial", "Fun", "Grudge",
               "Ideology", "NA", "Other", "Secondary", "ABC"):
        BINARY_COLS.append(f"actor.{_who}.motive.{_m}")

SCHEMA = DATE_COLS + KEY_COLS + BINARY_COLS + TARGETS
SENTINEL = "Unknown"


# ---- VERIS ----------------------------------------------------------------
def _flag(node, *path) -> int:
    cur = node
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return 0
        cur = cur[p]
    if isinstance(cur, dict):
        return int(bool(cur))
    if isinstance(cur, list):
        return int(len(cur) > 0)
    return int(bool(cur))


def load_veris(d: Path) -> pd.DataFrame:
    files = sorted(d.rglob("*.json"))
    if not files:
        raise SystemExit(f"no VCDB records under {d}; run acquire_data.py first")
    rows = []
    for f in files:
        try:
            inc = json.loads(f.read_text(encoding="utf8"))
        except Exception:                                     # noqa: BLE001
            continue
        tl = inc.get("timeline", {}).get("incident", {})
        vic = (inc.get("victim") or {})
        if isinstance(vic, list):
            vic = vic[0] if vic else {}
        r = {
            "timeline.incident.day": tl.get("day"),
            "timeline.incident.month": tl.get("month"),
            "timeline.incident.year": tl.get("year"),
            "victim.victim_id": vic.get("victim_id", SENTINEL),
            "victim.industry.name": vic.get("industry", SENTINEL),
            "victim.country": (vic.get("country") or [SENTINEL])[0]
            if isinstance(vic.get("country"), list) else vic.get("country", SENTINEL),
            "pattern": inc.get("pattern", SENTINEL),
            "Attack": np.nan,                       # absent from VERIS
        }
        for c in BINARY_COLS:
            parts = c.split(".")
            r[c] = _flag(inc, *parts)
        rows.append(r)
    df = pd.DataFrame(rows)
    # completeness criterion from Section 3.1
    df = df[~(df["victim.country"].isna() & df["victim.victim_id"].isna())]
    return df.reset_index(drop=True)


# ---- Hackmageddon ---------------------------------------------------------
def load_hackmageddon(d: Path) -> pd.DataFrame:
    files = [p for p in d.iterdir()
             if p.suffix.lower() in {".xlsx", ".xls", ".csv"}]
    if not files:
        raise SystemExit(f"no Hackmageddon timelines under {d}; see acquire_data.py")

    frames = []
    for f in sorted(files):
        t = pd.read_csv(f) if f.suffix.lower() == ".csv" else pd.read_excel(f)
        t.columns = [str(c).strip() for c in t.columns]
        frames.append(t)
    raw = pd.concat(frames, ignore_index=True)

    def pick(*names):
        for n in names:
            for c in raw.columns:
                if c.lower().replace(" ", "") == n.lower().replace(" ", ""):
                    return raw[c]
        return pd.Series([np.nan] * len(raw))

    dt = pd.to_datetime(pick("Date", "DateReported", "Date Reported"), errors="coerce")
    df = pd.DataFrame({
        "timeline.incident.day": dt.dt.day,
        "timeline.incident.month": dt.dt.month,
        "timeline.incident.year": dt.dt.year,
        "victim.victim_id": pick("Target"),
        "victim.industry.name": pick("Target Class", "TargetClass"),
        "victim.country": pick("Country"),
        "Attack": pick("Attack"),
        "pattern": np.nan,                          # absent from Hackmageddon
    })
    df = df[~(df["victim.country"].isna() & df["victim.victim_id"].isna())]

    # Section 3.1: split multi-country incidents into one row per country
    df["victim.country"] = df["victim.country"].astype(str)
    df = (df.assign(**{"victim.country": df["victim.country"].str.split(r"[;,/]")})
            .explode("victim.country"))
    df["victim.country"] = df["victim.country"].str.strip()
    df = df[df["victim.country"].ne("")]

    try:                                            # expand ISO codes
        import pycountry
        def expand(v):
            if isinstance(v, str) and 2 <= len(v) <= 3 and v.isupper():
                c = (pycountry.countries.get(alpha_2=v)
                     or pycountry.countries.get(alpha_3=v))
                return c.name if c else v
            return v
        df["victim.country"] = df["victim.country"].map(expand)
    except ImportError:
        print("  pycountry not installed; country codes left unexpanded")

    for c in BINARY_COLS:
        df[c] = np.nan                              # reconstructed below
    return df.reset_index(drop=True)


# ---- cross-source reconstruction ------------------------------------------
def reconstruct(donor: pd.DataFrame, recipient: pd.DataFrame,
                fields, shared) -> pd.DataFrame:
    """Fit a Random Forest per missing field on the donor, apply to recipient."""
    enc = {}
    D, R = donor.copy(), recipient.copy()
    for c in shared:
        if D[c].dtype == object or R[c].dtype == object:
            le = LabelEncoder().fit(
                pd.concat([D[c], R[c]]).astype(str).fillna(SENTINEL))
            D[c] = le.transform(D[c].astype(str).fillna(SENTINEL))
            R[c] = le.transform(R[c].astype(str).fillna(SENTINEL))
            enc[c] = le
    Xd = D[shared].fillna(0).values
    Xr = R[shared].fillna(0).values

    for f in fields:
        y = donor[f]
        mask = y.notna()
        if mask.sum() < 50:
            print(f"  skipping {f}: only {int(mask.sum())} donor rows")
            continue
        yl = LabelEncoder().fit(y[mask].astype(str))
        rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=1)
        rf.fit(Xd[mask.values], yl.transform(y[mask].astype(str)))
        recipient[f] = yl.inverse_transform(rf.predict(Xr))
    return recipient


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-imputation", action="store_true",
                    help="emit the intersection schema without reconstructed fields")
    args = ap.parse_args()

    DERIVED.mkdir(parents=True, exist_ok=True)
    print("Building unified corpus")
    print("-" * 60)

    veris = load_veris(RAW / "vcdb")
    print(f"  VERIS         {len(veris):>7,} records")
    hack = load_hackmageddon(RAW / "hackmageddon")
    print(f"  Hackmageddon  {len(hack):>7,} records")

    shared = DATE_COLS + KEY_COLS

    if not args.skip_imputation:
        print("  reconstructing Hackmageddon-side VERIS fields ...")
        hack = reconstruct(veris, hack, ["pattern"] + BINARY_COLS, shared)
        print("  reconstructing VERIS-side attack labels ...")
        veris = reconstruct(hack, veris, ["Attack"], shared)

    corpus = pd.concat([veris, hack], ignore_index=True)
    for c in SCHEMA:
        if c not in corpus.columns:
            corpus[c] = SENTINEL
    corpus = corpus[SCHEMA]

    for c in DATE_COLS:
        corpus[c] = corpus[c].ffill()
    for c in KEY_COLS + TARGETS:
        corpus[c] = corpus[c].fillna(SENTINEL)

    corpus.to_csv(OUT, index=False)
    print("-" * 60)
    print(f"  wrote {OUT}  ({len(corpus):,} rows x {corpus.shape[1]} columns)")
    print("  next: python validate_corpus.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
