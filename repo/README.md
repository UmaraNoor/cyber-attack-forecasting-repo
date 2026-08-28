# Cyber-Attack Forecasting on Merged Incident Repositories

Corpus, code and results for:

> **Entity Embeddings and Ensemble Learning for Multi-Attribute Cyber-Attack Forecasting: A Benchmark and Evaluation Protocol for Merged Incident Repositories**
> I. Fiza, U. Zahid — Department of Computer Science, International Islamic University, Islamabad

This repository contains the code, taxonomy and per-seed results behind every
number, table and figure in the paper.

Nothing runs out of the box: the corpus is not redistributed, so you must
obtain the two upstream sources and rebuild it first (see the Dataset Notice
below). Once rebuilt, every experiment can be re-executed. Two caveats apply
and are documented where relevant: `build_corpus.py` has not been executed
against raw upstream files by its author, and results will differ from the
paper if your Hackmageddon timelines were accessed after 7 August 2023.

---

## Dataset Notice

> This repository does not redistribute the original Hackmageddon dataset.
> Researchers should obtain the dataset directly from its original source and
> comply with the applicable terms of use and licensing conditions. This
> repository provides scripts and documentation for preprocessing and
> reproducing the experiments.

The corpus analysed in the paper is derived from **Hackmageddon**
(https://www.hackmageddon.com/, accessed 7 August 2023) and the **VERIS
Community Database** (https://github.com/vz-risk/VCDB). Neither is mirrored
here, and nor is the merged corpus. Both are obtained locally and the corpus is
rebuilt by the scripts in `src/`. Full source documentation, citations and
access dates: [`data/README.md`](data/README.md).

---

## What this is

Cyber-attack forecasting is limited by data. Intrusion-detection corpora describe
testbed traffic rather than real incidents; incident repositories are individually
either temporally dated or categorically shallow. The usual response is to merge
repositories and impute each source's missing attributes from the other.

We show that this practice creates a specific evaluation hazard. After merging,
a substantial fraction of labels are themselves model outputs computed from the
same features later used as predictors, so a downstream model can score well by
approximating an earlier model rather than by learning the phenomenon. Pooled
metrics do not reveal this.

The repository provides:

- **the merged corpus** — 20,279 incidents, 1971–2023, 64 harmonised attributes,
  built from the VERIS Community Database and Hackmageddon timelines;
- **the recovered attack taxonomy** — all 588 raw labels mapped to 11 categories,
  each assignment tagged with its provenance;
- **a three-part evaluation protocol** — provenance stratification, feature-group
  ablation, and a masked-imputation control that isolates circularity from
  correlated era effects;
- **all experiments** in the paper, as runnable scripts.

---

## Headline results

| Finding | Result |
|---|---|
| Circularity, corpus as constructed | Reconstructed labels inflate attack-type macro F1 by **43.83 points** (26.58 → 70.41, p < 0.0001) |
| Circularity, era held constant | Inflation persists at **16.86 points** (p = 0.012) — imputation isolated as cause |
| Imputer fidelity | Inflation scales **inversely** with imputer accuracy (42.2% fidelity → +16.9; 91.3% → +7.5) |
| Representation | Entity embeddings recover **+18.57 macro F1** over label encoding at fixed architecture |
| Cardinality | 189-class country target caps at **7.32% macro F1** despite 66.79% accuracy |
| Ensemble weighting | No scheme, including CAWPE, beats uniform averaging (all p > 0.25) |
| Forward-looking forecasting | Chronological split: **38.44% accuracy, 19.32% macro F1**; model ordering reverses in the deep model's favour (p < 0.003) |

---

## Repository layout

```
.
├── data/
│   ├── README.md                   dataset notice, sources, citations
│   ├── raw/
│   │   ├── hackmageddon/           EMPTY - you obtain these (not redistributed)
│   │   └── vcdb/                   EMPTY - fetched by acquire_data.py
│   ├── derived/
│   │   └── unified_corpus.csv      EMPTY - built by build_corpus.py
│   └── taxonomy/
│       ├── recovered_mapping.csv   181 labels recovered empirically
│       └── attack_mapping_full.csv all 588 labels + provenance tier
├── src/
│   ├── paths.py                    file locations (override with CAF_ROOT)
│   ├── acquire_data.py             fetch VCDB; instructions for Hackmageddon
│   ├── build_corpus.py             harmonise both sources into the corpus
│   ├── validate_corpus.py          check a rebuild against published statistics
│   ├── attack_taxonomy.py          588 -> 11 consolidation
│   ├── model_v2.py                 entity-embedding model + training loop
│   ├── ensemble_v3.py              three encoders + RL weighting layer
│   ├── rebuild_pipeline.py         leak-free benchmark (label-encoded)
│   ├── cawpe_baseline.py           ensemble weighting comparison
│   ├── mask_impute_control.py      masked-imputation control (Diagnostic 3)
│   ├── caveats.py                  provenance split + feature ablation
│   ├── chronological_split.py      forward-looking evaluation
│   ├── circularity_check.py        single-seed walkthrough of Diagnostics 1-2
│   ├── leakage_audit.py            audit of the pre-correction implementation
│   │                               (requires the original files; not shipped)
│   ├── make_figures.py             regenerates Figures 1-7 from results/
│   └── make_graphical_abstract.py  regenerates the graphical abstract
├── results/                        per-seed outputs (CSV)
├── figures/                        publication figures (300 dpi PNG)
└── docs/
    └── EXPERIMENTS.md              paper section -> command mapping
```

---

## Installation

Python 3.10 or later.

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>
pip install -r requirements.txt
```

### Obtain the data and build the corpus

The corpus is not distributed. Build it before running any experiment:

```bash
cd src
python acquire_data.py      # fetches VCDB; prints how to obtain Hackmageddon
                            # -> then place Hackmageddon files in data/raw/hackmageddon/
python build_corpus.py      # writes data/derived/unified_corpus.csv
python validate_corpus.py   # compares your rebuild to the published statistics
```

`validate_corpus.py` reports any divergence from the corpus used in the paper.
Divergence is expected if your Hackmageddon timelines were accessed after
7 August 2023; results will then be directionally consistent but numerically
different, and you should say so if you report them.

CPU is sufficient. No GPU is required and none was used for the published
results; the largest single experiment takes roughly one minute per seed on one
core.

---

## Reproducing the paper

Run from the `src/` directory. Results append to `results/`.

```bash
cd src
```

**Table 6, 7 — benchmark and ensemble weighting**

```bash
for s in 0 1 2 3 4; do
  for t in Attack pattern victim.country; do
    python ensemble_v3.py --target $t --seed $s
  done
done
```

**Table 8 — ensemble weighting schemes including CAWPE**

```bash
for s in 0 1 2 3 4; do
  python cawpe_baseline.py --target Attack --seed $s
  python cawpe_baseline.py --target victim.country --seed $s
done
```

**Table 9 — representation ablation (label encoding baseline)**

```bash
for s in 0 1 2; do
  for t in Attack pattern victim.country; do
    python rebuild_pipeline.py --target $t --seed $s
  done
done
```

**Table 10, Fig. 3 — provenance stratification (Diagnostic 1)**

```bash
for s in 0 1 2 3 4; do python caveats.py --mode source --seed $s; done
```

**Table 11, Fig. 4 — masked-imputation control (Diagnostic 3)**

```bash
for s in 0 1 2 3 4; do
  python mask_impute_control.py --target Attack  --seed $s
  python mask_impute_control.py --target pattern --seed $s
done
```

**Table 12, Fig. 5 — feature-group ablation (Diagnostic 2)**

```bash
for s in 0 1 2; do
  python caveats.py --mode ablate --target pattern        --seed $s
  python caveats.py --mode ablate --target victim.country --seed $s
done
```

**Table 13, Fig. 7 — chronological evaluation**

```bash
for s in 0 1 2 3 4; do
  python chronological_split.py --target Attack         --config full     --seed $s
  python chronological_split.py --target Attack         --config observed --seed $s
  python chronological_split.py --target victim.country --config full     --seed $s
done
```

**Figures**

```bash
python make_figures.py
```

**Taxonomy reconstruction**

```bash
python attack_taxonomy.py     # rebuilds data/taxonomy/attack_mapping_full.csv
```

Expected wall-clock on one CPU core: about 90 minutes for the experiment
suite, excluding corpus construction. `build_corpus.py` fits Random Forest
imputers over 57 fields and may take considerably longer; it has not been
timed. Peak memory stays under roughly 3 GB except for Random Forest on the
189-class country target, which is the most memory-hungry step.

---

## Verifying a single result quickly

The masked-imputation control is the paper's central experiment and runs in
under a minute:

```bash
cd src && python mask_impute_control.py --target Attack --seed 0
```

Expected output:

```
  Attack   [hackmageddon] observed       n= 1517 acc= 43.70  macroF1= 22.78
  Attack   [hackmageddon] reconstructed  n=  990 acc= 57.88  macroF1= 36.12
  imputer reproduced 41.5% of masked labels exactly
```

The gap between the two strata, with era and source held constant, is the
circularity effect.

---

## The corpus

`data/unified_corpus.csv` — 20,279 rows × 64 columns.

| Field group | Count | Notes |
|---|---|---|
| Temporal | 3 | day, month, year (1971–2023) |
| Victim | 3 | country (189 values), industry, organisation |
| Targets | 2 | `Attack` (11 classes after consolidation), `pattern` (11) |
| Binary indicators | 56 | threat actor, motive, asset, compromised attribute |

**Provenance matters when using this corpus.** Because the two sources cover
disjoint periods and each lacks fields the other records, roughly half the labels
for each target were reconstructed at merge time:

| Target | Observed | Reconstructed |
|---|---|---|
| `Attack` | Hackmageddon era (≥ 2018) | VERIS era (≤ 2017) |
| `pattern` | VERIS era (≤ 2017) | Hackmageddon era (≥ 2018) |
| `victim.country` | all records | — |

Any evaluation on this corpus should report metrics stratified by provenance.
See `docs/EXPERIMENTS.md` and Section 5 of the paper.

**Known gaps.** The corpus contains no records dated 2021, despite the
Hackmageddon collection nominally spanning 2018–2023. This constrains
chronological evaluation, which trains on records to 2020.

---

## Licensing and redistribution

**No licence has been chosen yet.** See [`LICENSE.md`](LICENSE.md). Until a
licence is added, default copyright applies (all rights reserved), which is
incompatible with the reproducibility claims in the paper. This must be
resolved before the repository is made public.

Neither upstream dataset is redistributed here, and nor is the merged corpus
derived from them. See the Dataset Notice above and
[`data/README.md`](data/README.md) for sources, access dates and citations.

The taxonomy mapping tables in `data/taxonomy/` **are** included. They map
attack-label strings to analytical categories and contain no incident records,
victims, dates or locations; they are original analytical work rather than a
reproduction of either source. They can be regenerated with
`attack_taxonomy.py` if you prefer not to rely on the shipped copies.

Users are responsible for complying with the terms of both upstream sources.

## Citation

```bibtex
@article{fiza2026forecasting,
  title   = {Entity Embeddings and Ensemble Learning for Multi-Attribute
             Cyber-Attack Forecasting: A Benchmark and Evaluation Protocol
             for Merged Incident Repositories},
  author  = {Fiza, Inbasat and Zahid, Umara},
  journal = {Under review},
  year    = {2026}
}
```

---

## Contact

Corresponding author: Dr Umara Zahid, Department of Computer Science, Faculty of
Computing and Information Technology, International Islamic University,
H-10, Islamabad 44000, Pakistan.
