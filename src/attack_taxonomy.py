"""
attack_taxonomy.py
==================
Consolidates the 588 raw Attack labels in Dataset A into the 11 forecasting
categories used in the manuscript.

The mapping is built in two tiers:

  Tier 1 (empirical, 181 labels, 97.5% of rows)
      Recovered by matching Dataset A records to their counterparts in
      Final_Data.csv on all 63 non-Attack columns, then reading off the
      consolidated label the original authors assigned. Zero ambiguity:
      every recovered raw label mapped to exactly one category.

  Tier 2 (rule-based, 407 labels, 2.5% of rows)
      The residual long tail, unmatched because those records were not
      sampled into Final_Data.csv. Assigned by case-insensitive keyword
      rules derived from the Tier-1 category definitions, applied in
      priority order. Anything unmatched falls to 'other', which is
      consistent with how Tier 1 treats miscellaneous techniques.

Every assignment is recorded with its tier so the mapping can be audited
and reported as supplementary material.
"""

import re
import pandas as pd

from paths import CORPUS, RECOVERED_MAPPING, FULL_MAPPING

CATEGORIES = [
    "malware", "hijacking", "vulnerability", "ddos", "sqli", "defacement",
    "targeted attack", "fake account/website",
    "coordinated inauthentic behavior", "unknown", "other",
]

# Rules are evaluated top to bottom; first match wins. Order matters:
# more specific patterns must precede more general ones.
RULES = [
    ("unknown",                          r"^\s*unknown\s*$"),
    ("sqli",                             r"\bsql\s*i\b|\bsql injection\b"),
    ("ddos",                             r"\bd?dos\b|denial[- ]of[- ]service|black\s*box"),
    ("defacement",                       r"\bdeface"),
    ("coordinated inauthentic behavior", r"\bcib\b|\bcid\b|coordinated inauthentic"),
    ("fake account/website",             r"\bfake\b|counterfeit|typosquat|spoofed (site|website|page)"),
    ("targeted attack",                  r"targeted attack|\bapt\b"),
    ("vulnerability",                    r"vulnerab|\bcve\b|cve-\d|0[- ]day|zero[- ]day|\bexploit|misconfigur"),
    ("malware",                          r"malware|ransomware|trojan|spyware|worm|rootkit|cryptojack|coinmin|keylog|botnet|\bbot attack\b|skimmer|skimming|backdoor"),
    ("hijacking",                        r"hijack|takeover|\bbec\b|business email compromise|credential stuffing|brute[- ]?force|account compromise|session (fixation|hijack)|dns poison|bgp"),
]

_COMPILED = [(cat, re.compile(pat, re.I)) for cat, pat in RULES]


def _rule_map(label: str) -> str:
    """Assign a raw label to a category using the keyword rules."""
    text = str(label)
    for cat, rx in _COMPILED:
        if rx.search(text):
            return cat
    return "other"


def build_mapping(recovered_csv: str, raw_labels) -> pd.DataFrame:
    """
    Build the full raw-label -> category mapping.

    Parameters
    ----------
    recovered_csv : path to the Tier-1 mapping recovered from Final_Data.csv
                    (two columns: raw label, consolidated category)
    raw_labels    : iterable of every distinct raw Attack label in Dataset A

    Returns
    -------
    DataFrame indexed by raw label with columns ['category', 'tier'].
    """
    tier1 = pd.read_csv(recovered_csv, index_col=0).iloc[:, 0].to_dict()

    rows = []
    for lab in sorted(set(raw_labels)):
        if lab in tier1:
            rows.append((lab, tier1[lab], "empirical"))
        else:
            rows.append((lab, _rule_map(lab), "rule-based"))

    out = pd.DataFrame(rows, columns=["raw_label", "category", "tier"])
    out = out.set_index("raw_label")

    bad = set(out["category"]) - set(CATEGORIES)
    if bad:
        raise ValueError(f"mapping produced unknown categories: {bad}")
    return out


def apply_mapping(series: pd.Series, mapping: pd.DataFrame) -> pd.Series:
    """Map a Series of raw Attack labels to consolidated categories."""
    return series.map(mapping["category"]).astype("object")


if __name__ == "__main__":
    import sys
    A = pd.read_csv(CORPUS, low_memory=False)
    m = build_mapping(str(RECOVERED_MAPPING), A["Attack"].unique())
    m.to_csv(str(FULL_MAPPING))

    consolidated = apply_mapping(A["Attack"], m)
    print(f"raw labels mapped : {len(m)}")
    print(f"  empirical  : {(m['tier']=='empirical').sum()}")
    print(f"  rule-based : {(m['tier']=='rule-based').sum()}")
    print(f"unmapped rows     : {consolidated.isna().sum()}")
    print("\nresulting class distribution:")
    print(consolidated.value_counts().to_string())
