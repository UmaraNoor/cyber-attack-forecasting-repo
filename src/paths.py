"""
paths.py
========
Single source of truth for file locations, so every script runs from a fresh
clone with no editing. Override the repository root with the CAF_ROOT
environment variable if you relocate the data.
"""

import os
from pathlib import Path

ROOT = Path(os.environ.get("CAF_ROOT", Path(__file__).resolve().parent.parent))

DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
TAXONOMY = DATA / "taxonomy"

# The harmonised corpus. NOT redistributed in this repository: it is derived
# from Hackmageddon and VERIS and must be rebuilt locally from those sources.
#   python acquire_data.py  ->  python build_corpus.py  ->  python validate_corpus.py
CORPUS = DATA / "derived" / "unified_corpus.csv"

RAW = DATA / "raw"

# Tier-1 mapping recovered empirically (181 raw labels, 97.5% of records).
RECOVERED_MAPPING = TAXONOMY / "recovered_mapping.csv"

# Full 588-label mapping with per-label provenance (empirical | rule-based).
FULL_MAPPING = TAXONOMY / "attack_mapping_full.csv"

for _d in (RESULTS, FIGURES, DATA / "raw", DATA / "derived"):
    _d.mkdir(parents=True, exist_ok=True)
