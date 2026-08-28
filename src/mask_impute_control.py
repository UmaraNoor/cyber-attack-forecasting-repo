"""
mask_impute_control.py
======================
Control experiment for Major 3: isolates imputation-induced circularity from
era-specific predictability.

THE PROBLEM
-----------
In the unified corpus, label provenance is perfectly collinear with source era:

    attack type : observed for Hackmageddon rows (>=2018)
                  reconstructed for VERIS rows   (<=2017)
    pattern     : the reverse

So the +43.83 macro-F1 gap reported in Section 7.3 admits three explanations
that the stratified comparison alone cannot separate: circularity, era-specific
predictability, and label-distribution differences between the sources.

THE CONTROL
-----------
Work entirely WITHIN a single era, where every label is genuinely observed:

  1. restrict the corpus to one era (all labels observed, one source only)
  2. freeze a train / val / test split
  3. randomly designate a fraction of ALL rows as "masked"
  4. fit the SAME Random Forest imputer used at merge time, on the *unmasked
     training rows only*, predicting the target from the shared features
  5. overwrite the target on every masked row with the imputer's prediction
  6. train the forecasting model on the resulting mixed-provenance training set
  7. evaluate on the held-out test set, stratified into
        observed      (label never touched)
        reconstructed (label overwritten by the imputer)

Era, source, taxonomy and time period are identical across the two strata.
The only difference is whether the label passed through the imputer. Any gap
is therefore attributable to imputation and nothing else.

Usage:
    python mask_impute_control.py --target Attack --seed 0
    python mask_impute_control.py --target pattern --seed 0
"""

import argparse, os, time
import numpy as np
import pandas as pd
import torch

from paths import CORPUS, RECOVERED_MAPPING, RESULTS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)

from model_v2 import load, build_features, train, evaluate

# Era in which each target is genuinely OBSERVED in the real corpus.
OBSERVED_ERA = {
    "Attack":  ("hackmageddon", lambda y: y >= 2018),
    "pattern": ("veris",        lambda y: y <= 2017),
}

# Fraction masked. Chosen to mirror the real corpus, in which 7,745 of 20,279
# records (38%) carry a reconstructed attack label.
MASK_FRAC = 0.40


def imputer_matrix(df, target):
    """Feature matrix available to the merge-time imputer: everything the two
    repositories share, i.e. all numeric columns except the target itself and
    the free-text victim identifier."""
    drop = [c for c in [target, "victim.victim_id"] if c in df.columns]
    X = df.drop(columns=drop).select_dtypes(include=[np.number])
    return X.values.astype(np.float32)


def score(y, p):
    return dict(
        accuracy=accuracy_score(y, p) * 100,
        macro_precision=precision_score(y, p, average="macro", zero_division=0) * 100,
        macro_recall=recall_score(y, p, average="macro", zero_division=0) * 100,
        macro_f1=f1_score(y, p, average="macro", zero_division=0) * 100,
        weighted_f1=f1_score(y, p, average="weighted", zero_division=0) * 100,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=list(OBSERVED_ERA))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mask_frac", type=float, default=MASK_FRAC)
    ap.add_argument("--out", default=str(RESULTS / "mask_impute_results.csv"))
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    # ---- 1. restrict to the era in which the target is observed -------------
    df = load()
    era_name, era_test = OBSERVED_ERA[args.target]
    df = df[era_test(df["timeline.incident.year"].values)].reset_index(drop=True)
    n = len(df)

    # ---- 2. frozen split ----------------------------------------------------
    idx = np.arange(n)
    tmp, te = train_test_split(idx, test_size=0.20, random_state=args.seed)
    tr, va = train_test_split(tmp, test_size=0.20, random_state=args.seed)

    # ---- 3. designate masked rows ------------------------------------------
    masked = np.zeros(n, dtype=bool)
    masked[rng.choice(n, size=int(round(args.mask_frac * n)), replace=False)] = True

    # ---- 4. fit the imputer on UNMASKED TRAINING rows only ------------------
    Ximp = imputer_matrix(df, args.target)
    y_true = df[args.target].values.copy()
    fit_rows = tr[~masked[tr]]
    imp = RandomForestClassifier(n_estimators=100, random_state=args.seed,
                                 n_jobs=1).fit(Ximp[fit_rows], y_true[fit_rows])

    # ---- 5. overwrite masked labels with imputer output --------------------
    y_mixed = y_true.copy()
    if masked.any():
        y_mixed[masked] = imp.predict(Ximp[masked])
    agree = (y_mixed[masked] == y_true[masked]).mean() * 100

    df[args.target] = y_mixed

    # ---- 6. train the forecasting model on mixed-provenance labels ---------
    Xc, Xd, cards, y = build_features(df, args.target)
    n_cls = int(max(y.max(), y_true.max())) + 1
    model, _ = train(Xc, Xd, y, tr, va, n_cls, args.seed,
                     hidden=(512, 256), use_weights=False)
    XcT, XdT = torch.tensor(Xc), torch.tensor(Xd)

    # ---- 7. evaluate, stratified by provenance -----------------------------
    rows = []
    strata = [("observed", te[~masked[te]]), ("reconstructed", te[masked[te]])]
    for label, sub in strata:
        if len(sub) == 0:
            continue
        _, m = evaluate(model, XcT[sub], XdT[sub], y[sub])
        rows.append(dict(target=args.target, seed=args.seed, era=era_name,
                         stratum=label, n=len(sub),
                         imputer_agreement=round(agree, 2), **m))
        print(f"  {args.target:8s} [{era_name}] {label:14s} n={len(sub):5d} "
              f"acc={m['accuracy']:6.2f}  macroF1={m['macro_f1']:6.2f}")

    out = pd.DataFrame(rows)
    out.to_csv(args.out, mode="a", header=not os.path.exists(args.out), index=False)
    print(f"  imputer reproduced {agree:.1f}% of masked labels exactly "
          f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
