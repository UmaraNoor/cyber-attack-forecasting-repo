"""
validate_corpus.py
==================
Compare a locally rebuilt corpus against the statistics of the corpus used in
the paper. The corpus itself is not redistributed, so this fingerprint is how
you check that your rebuild is equivalent.

Usage:
    python validate_corpus.py
    python validate_corpus.py --corpus /path/to/unified_corpus.csv
"""

import argparse
import sys

import pandas as pd

from paths import CORPUS

# Fingerprint of the corpus reported in the paper. Hackmageddon timelines
# accessed 7 August 2023; VCDB as of the same period.
EXPECTED = {
    "n_rows": 20279,
    "n_cols": 64,
    "year_min": 1971,
    "year_max": 2023,
    "n_countries": 189,
    "n_patterns": 11,
    "n_raw_attack_labels": 588,
    "veris_rows": 7745,          # year <= 2017
    "hackmageddon_rows": 12534,  # year >= 2018
}

TOLERANCE = 0.02   # 2% on row counts; schema must match exactly


def check(name, got, want, exact=False):
    if exact or isinstance(want, str):
        ok = got == want
    else:
        ok = abs(got - want) <= max(1, TOLERANCE * want)
    flag = "OK  " if ok else "FAIL"
    delta = "" if ok or isinstance(want, str) else f"  ({got - want:+,})"
    print(f"  [{flag}] {name:24s} got {got:>10,}   expected {want:>10,}{delta}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(CORPUS))
    args = ap.parse_args()

    try:
        df = pd.read_csv(args.corpus, low_memory=False)
    except FileNotFoundError:
        print(f"corpus not found at {args.corpus}")
        print("Run acquire_data.py then build_corpus.py first.")
        return 1

    print(f"Validating {args.corpus}")
    print("-" * 74)
    yr = df["timeline.incident.year"]
    results = [
        check("rows", len(df), EXPECTED["n_rows"]),
        check("columns", df.shape[1], EXPECTED["n_cols"], exact=True),
        check("earliest year", int(yr.min()), EXPECTED["year_min"], exact=True),
        check("latest year", int(yr.max()), EXPECTED["year_max"], exact=True),
        check("distinct countries", df["victim.country"].nunique(), EXPECTED["n_countries"]),
        check("distinct patterns", df["pattern"].nunique(), EXPECTED["n_patterns"], exact=True),
        check("raw attack labels", df["Attack"].nunique(), EXPECTED["n_raw_attack_labels"]),
        check("VERIS-era rows", int((yr <= 2017).sum()), EXPECTED["veris_rows"]),
        check("Hackmageddon-era rows", int((yr >= 2018).sum()), EXPECTED["hackmageddon_rows"]),
    ]
    print("-" * 74)

    if 2021 in set(yr):
        print("  NOTE  your rebuild contains 2021 records; the published corpus")
        print("        contains none. Chronological results will differ from the")
        print("        paper, which trains on records to 2020 for that reason.")

    n_ok = sum(results)
    print(f"\n{n_ok}/{len(results)} checks passed")
    if n_ok == len(results):
        print("Rebuild is equivalent to the published corpus. Proceed to the "
              "experiment scripts in docs/EXPERIMENTS.md.")
        return 0

    print("""
Rebuild diverges. Likely causes, in order of probability:

  1. Hackmageddon timelines accessed on a different date. The site is updated
     continuously; row counts in the 2018-2023 range will differ. Results will
     be directionally consistent but numerically different from the paper.
  2. A different subset of timeline files was downloaded. Confirm coverage of
     2018-2023 and note that the published corpus contains no 2021 records.
  3. A schema mismatch, i.e. the column count differs. This is a bug in
     build_corpus.py rather than a data issue; please open an issue.

Divergence in row counts does not invalidate your run. It does mean your
numbers will not match the paper exactly, and you should say so if you
report them.""")
    return 1


if __name__ == "__main__":
    sys.exit(main())
