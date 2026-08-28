# Data

## Dataset Notice

> This repository does not redistribute the original Hackmageddon dataset.
> Researchers should obtain the dataset directly from its original source and
> comply with the applicable terms of use and licensing conditions. This
> repository provides scripts and documentation for preprocessing and
> reproducing the experiments.

---

## What is and is not here

| Path | Contents | Redistributed? |
|---|---|---|
| `data/raw/hackmageddon/` | empty — you populate it | **No.** Obtain from the official source. |
| `data/raw/vcdb/` | empty — populated by `acquire_data.py` | No. Fetched at run time from the upstream repository. |
| `data/derived/unified_corpus.csv` | empty — produced by `build_corpus.py` | No. Built locally from the two sources above. |
| `data/taxonomy/*.csv` | attack-label → category mapping | **Yes.** Original derived work; see below. |

Neither upstream dataset is mirrored, scraped or vendored in this repository.
The corpus analysed in the paper is a derived product of both and is likewise
not distributed; it is rebuilt locally by the scripts in `src/`.

---

## Source 1 — Hackmageddon

| | |
|---|---|
| **Dataset** | Hackmageddon — Information Security Timelines and Statistics |
| **Author** | Paolo Passeri |
| **Source** | https://www.hackmageddon.com/ |
| **Coverage used** | 2018–2023 timelines |
| **Date accessed** | 7 August 2023 |
| **Records after cleaning** | 12,534 (after multi-country decomposition) |
| **Redistributed here** | **No** |

**Citation**

```
P. Passeri, HACKMAGEDDON — Information Security Timelines and Statistics.
https://www.hackmageddon.com/ (accessed 7 August 2023).
```

**How to obtain.** Visit the site, locate the timeline archives covering
2018–2023, review the site's terms of use, satisfy yourself that your intended
use is permitted, and save the files unmodified into `data/raw/hackmageddon/`.
`acquire_data.py` will print these instructions and will not download anything
from the site on your behalf.

Timelines accessed after August 2023 will yield a corpus that differs from the
published one. `validate_corpus.py` reports the divergence.

---

## Source 2 — VERIS Community Database

| | |
|---|---|
| **Dataset** | VERIS Community Database (VCDB) |
| **Maintainer** | Verizon / VERIS community |
| **Source** | https://github.com/vz-risk/VCDB |
| **Schema** | https://verisframework.org/ |
| **Coverage used** | 1971–2017 |
| **Records after cleaning** | 7,745 |
| **Redistributed here** | **No** — fetched at run time by `acquire_data.py` |

**Citation**

```
VERIS Community Database (VCDB), vz-risk/VCDB.
https://github.com/vz-risk/VCDB (accessed 17 December 2024).
```

Consult the upstream repository for its licence before redistributing any
part of it.

---

## Derived artefacts that *are* included

`data/taxonomy/recovered_mapping.csv` and `attack_mapping_full.csv` map the 588
raw attack-label strings appearing in the merged corpus to 11 analytical
categories, each assignment tagged as empirically recovered or rule-assigned.

These are included because they are original analytical work: a label
vocabulary and its classification, not incident records. They contain no
victim, organisation, date or country information from either source, and no
incident can be reconstructed from them. If you consider even the label
vocabulary to fall under an upstream term, remove the files and regenerate
them with `attack_taxonomy.py` once you hold the sources.

---

## Rebuilding the corpus

```bash
cd src
python acquire_data.py      # fetches VCDB; tells you how to obtain Hackmageddon
python build_corpus.py      # harmonises both into data/derived/unified_corpus.csv
python validate_corpus.py   # compares your rebuild to the published statistics
```

`validate_corpus.py` checks row counts, schema, era split, and category counts
against the published corpus. Exact agreement is expected only if your
Hackmageddon timelines match the 7 August 2023 access; otherwise the script
reports where and by how much you diverge.

**Note on `build_corpus.py`.** It implements the harmonisation described in
Section 3 of the paper, but it was written from that description rather than
from the original notebook, and it has not been executed against raw upstream
files. Treat it as a specification of the procedure until you have run it end
to end, and please open an issue if the schema check in `validate_corpus.py`
fails, since that would indicate a defect in the script rather than in your
data.

---

## Corpus schema (once built)

20,279 rows × 64 columns.

| Group | Count | Fields |
|---|---|---|
| Temporal | 3 | `timeline.incident.{day,month,year}` |
| Victim | 3 | `victim.{victim_id,industry.name,country}` |
| Targets | 2 | `Attack`, `pattern` |
| Binary indicators | 56 | `action.*`, `attribute.*`, `actor.*`, `asset.variety.*`, `actor.*.motive.*` |

**Label provenance.** Because the sources cover disjoint periods and each
lacks fields the other records, roughly half the labels for each target were
reconstructed during merging:

| Target | Observed | Reconstructed |
|---|---|---|
| `Attack` | Hackmageddon era (≥ 2018) | VERIS era (≤ 2017) |
| `pattern` | VERIS era (≤ 2017) | Hackmageddon era (≥ 2018) |
| `victim.country` | all records | — |

Any evaluation on this corpus should report metrics stratified by provenance.
This is the central finding of the accompanying paper; see Section 5.

**Known gap.** The published corpus contains no records dated 2021.
