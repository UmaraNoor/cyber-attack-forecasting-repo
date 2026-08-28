# Experiments: paper section → command → expected output

Every table and figure in the paper, with the command that produces it and the
value you should see. All figures are means over the stated number of seeds;
individual seeds vary within roughly the reported standard deviation.

**Prerequisite.** The corpus is not distributed with this repository. Build it
first:

```bash
cd src
python acquire_data.py      # fetches VCDB; instructions for Hackmageddon
python build_corpus.py      # writes data/derived/unified_corpus.csv
python validate_corpus.py   # confirms your rebuild matches the published corpus
```

If `validate_corpus.py` reports divergence, the expected values below will not
match exactly. See `data/README.md`.

Run all commands from `src/`.

`model_v2.py` can also be run directly as a single-model baseline; it writes to
`results/results_v2.csv`, which is not shipped and is not used by any table in
the paper.

---

## Section 7.1 — Benchmark (Table 6)

```bash
for s in 0 1 2 3 4; do
  for t in Attack pattern victim.country; do
    python ensemble_v3.py --target $t --seed $s
  done
done
```

Writes `results/results_v3.csv`. Expected (macro F1, mean ± SD, 5 seeds):

| Target | Deep ensemble (RL) | Random Forest |
|---|---|---|
| Attack type | 32.57 ± 2.62 | 35.46 ± 1.07 |
| Pattern | 88.95 ± 3.36 | 87.17 ± 3.15 |
| Country | 4.95 ± 0.63 | 7.32 ± 1.03 |

---

## Section 7.1 — Ensemble weighting schemes (Table 8)

```bash
for s in 0 1 2 3 4; do
  python cawpe_baseline.py --target Attack --seed $s
  python cawpe_baseline.py --target victim.country --seed $s
done
```

Writes `results/cawpe_results.csv`. Expected: no scheme beats uniform averaging.
Attack-type macro F1 spans 32.18 (best single) to 32.57 (policy gradient) against
a uniform reference of 32.25; all paired p > 0.25.

---

## Section 7.2 — Representation ablation (Table 9, Fig. 2)

The embedding arm comes from `ensemble_v3.py` above. The label-encoding arm:

```bash
for s in 0 1 2; do
  for t in Attack pattern victim.country; do
    python rebuild_pipeline.py --target $t --seed $s
  done
done
```

Writes `results/results.csv`. Expected macro F1 under label encoding: attack
14.00 ± 1.59, pattern 81.51 ± 4.21, country 1.79 ± 0.17 — i.e. +18.57, +7.44 and
+3.16 points respectively for entity embeddings.

---

## Section 7.3 — Provenance stratification, Diagnostic 1 (Table 10, Fig. 3)

```bash
for s in 0 1 2 3 4; do python caveats.py --mode source --seed $s; done
```

Writes `results/caveats_source.csv`. Expected:

| Target | Observed | Reconstructed | Inflation |
|---|---|---|---|
| Attack type | 26.58 ± 1.32 | 70.41 ± 5.84 | +43.83 (p < 0.0001) |
| Pattern | 84.24 ± 3.10 | 94.27 ± 5.05 | +10.03 (p = 0.029) |

---

## Section 7.4 — Masked-imputation control, Diagnostic 3 (Table 11, Fig. 4)

```bash
for s in 0 1 2 3 4; do
  python mask_impute_control.py --target Attack  --seed $s
  python mask_impute_control.py --target pattern --seed $s
done
```

Writes `results/mask_impute_results.csv`. Expected, with era held constant:

| Target | Imputer fidelity | Observed | Reconstructed | Inflation |
|---|---|---|---|---|
| Attack type | 42.2% | 24.03 ± 2.07 | 40.89 ± 7.16 | +16.86 (p = 0.012) |
| Pattern | 91.3% | 82.57 ± 3.68 | 90.05 ± 5.28 | +7.49 (p = 0.063) |

This is the paper's central experiment. It isolates circularity from the
era effects that confound Section 7.3, because both strata come from the same
source and the same period.

---

## Section 7.5 — Feature-group ablation, Diagnostic 2 (Table 12, Fig. 5)

```bash
for s in 0 1 2; do
  python caveats.py --mode ablate --target pattern        --seed $s
  python caveats.py --mode ablate --target victim.country --seed $s
done
```

Writes `results/caveats_ablate.csv`. Expected macro F1:

| Features withheld | Pattern | Country |
|---|---|---|
| None | 87.05 ± 4.48 | 5.73 ± 0.76 |
| `action.*` | 81.46 ± 4.31 | 5.28 ± 1.01 |
| `action/actor/asset/attribute` | 43.30 ± 1.34 | 4.38 ± 0.56 |

Pattern loses 43.75 points, confirming it is largely taxonomy-rule recovery.
Country is insensitive to every group, so its low ceiling is genuine difficulty.

---

## Section 7.7 — Chronological evaluation (Table 13, Fig. 7)

```bash
for s in 0 1 2 3 4; do
  python chronological_split.py --target Attack         --config full     --seed $s
  python chronological_split.py --target Attack         --config observed --seed $s
  python chronological_split.py --target victim.country --config full     --seed $s
done
```

Writes `results/chrono_results.csv`. Train ≤ 2020, validate 2022, test 2023.
Expected:

| Target | Training set | Deep model | Random Forest | Majority |
|---|---|---|---|---|
| Attack | Full | 18.29 ± 1.00 | 13.58 ± 0.30 | 5.96 |
| Attack | Observed only | 19.32 ± 1.59 | 14.51 ± 0.44 | 5.96 |
| Country | Full | 3.65 ± 0.20 | 3.60 ± 0.11 | 0.46 |

Note the reversal: the deep model loses to Random Forest under random
partitioning and wins under chronological partitioning (p = 0.0008 full,
p = 0.0029 observed-only).

Pattern is deliberately excluded from Table 13 of the paper. Observed pattern
labels end in 2017 while the test window is 2023, so a chronological evaluation
of pattern would measure a model recovering an imputer's output from imputed
inputs.

`results/chrono_results.csv` nonetheless contains pattern rows
(macro F1 98.40 +- 0.43 for the deep model, 89.21 +- 8.04 for Random Forest,
3 seeds). They were produced during development and are retained for
completeness. **They are not a forecasting result and must not be reported as
one** - the figure is high precisely because the target and its inputs are both
reconstructed on the test window.

---

## Figures

```bash
python make_figures.py
```

Regenerates all seven paper figures from the CSVs in `results/`:
`fig_pipeline`, `fig_representation`, `fig_circularity`, `fig_control`,
`fig_ablation`, `fig_cardinality`, `fig_chrono`.

```bash
python make_graphical_abstract.py
```

Regenerates `figures/graphical_abstract.png` (1,932 x 738 px).

---

## Taxonomy reconstruction

```bash
python attack_taxonomy.py
```

Rebuilds `data/taxonomy/attack_mapping_full.csv` from
`recovered_mapping.csv` plus keyword rules. Expected: 588 labels mapped,
181 empirical and 407 rule-based, 0 unmapped, 11 categories, largest class
`malware` (6,734) and smallest `sqli` (22).

---

## Audit of the original implementation

```bash
python leakage_audit.py --a <original.xlsx> --b <Final_Data.csv>
```

Reproduces the diagnostic findings that motivated the rebuild: exact target
reconstruction from engineered lag features, target presence in its own feature
matrix, duplicate-record contamination across the split, and the unreachable
resampling branch.

**This script cannot be run from this repository alone.** It requires the two
pre-correction files (the original imbalanced spreadsheet and the GAN-augmented
`Final_Data.csv`), neither of which is distributed here. It is included as a
record of the audit method rather than as a runnable check.

---

## Notes on reproducibility

- All scripts accept `--seed`; partitions are derived deterministically from it.
- Results append rather than overwrite. Delete the CSV in `results/` before
  re-running a sweep, or pass `--out` to write elsewhere.
- PyTorch is pinned to CPU with a single thread for determinism; small variation
  across BLAS implementations is expected.
- The complete suite takes roughly 90 minutes on one CPU core.
