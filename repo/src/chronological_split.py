"""
chronological_split.py
======================
Forward-looking evaluation for Major 8.

All results in the manuscript use random partitioning, which for a forecasting
task permits interpolation within the observation window rather than
extrapolation beyond it. This script re-evaluates under a strictly
chronological split:

    train       <= 2020   (the corpus contains no 2021 records)
    validation  == 2022
    test        == 2023

Two configurations are run:

  full      train on the whole corpus up to the cutoff. For attack type this
            means most training labels are RF-reconstructed (VERIS era) while
            every test label is observed, since the 2023 test window lies
            wholly within Hackmageddon coverage.

  observed  train only on records whose target label was observed, i.e. the
            Hackmageddon era for attack type and the VERIS era for pattern.
            For attack type this yields the cleanest available estimate:
            observed labels in training, observed labels in test, and a
            strictly forward-looking split.

The comparison between the two isolates what training on reconstructed labels
costs when the test set is real and future.

Usage:
    python chronological_split.py --target Attack --config full --seed 0
"""

import argparse, os, time
import numpy as np
import pandas as pd
import torch

from paths import CORPUS, RECOVERED_MAPPING, RESULTS
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)

from model_v2 import load, build_features, train, evaluate

TRAIN_MAX, VAL_YEAR, TEST_YEAR = 2020, 2022, 2023

# era in which each target's label is genuinely observed
OBSERVED = {
    "Attack":         lambda y: y >= 2018,
    "pattern":        lambda y: y <= 2017,
    "victim.country": lambda y: np.ones_like(y, dtype=bool),   # always observed
}


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
    ap.add_argument("--target", required=True, choices=list(OBSERVED))
    ap.add_argument("--config", default="full", choices=["full", "observed"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(RESULTS / "chrono_results.csv"))
    args = ap.parse_args()

    t0 = time.time()
    df = load()
    year = df["timeline.incident.year"].values

    Xc, Xd, cards, y = build_features(df, args.target)
    n_cls = int(y.max()) + 1

    tr = np.where(year <= TRAIN_MAX)[0]
    va = np.where(year == VAL_YEAR)[0]
    te = np.where(year == TEST_YEAR)[0]

    if args.config == "observed":
        obs = OBSERVED[args.target](year)
        tr = tr[obs[tr]]
        va = va[obs[va]]

    if len(tr) < 200 or len(va) < 50 or len(te) < 50:
        print(f"  SKIP {args.target}/{args.config}: "
              f"train={len(tr)} val={len(va)} test={len(te)}")
        return

    XcT, XdT = torch.tensor(Xc), torch.tensor(Xd)

    # deep model
    model, _ = train(Xc, Xd, y, tr, va, n_cls, args.seed,
                     hidden=(512, 256), use_weights=False)
    _, m_deep = evaluate(model, XcT[te], XdT[te], y[te])

    # Random Forest baseline on the same split
    Xflat = np.column_stack([Xc, Xd])
    rf = RandomForestClassifier(n_estimators=100, random_state=args.seed,
                                n_jobs=1).fit(Xflat[tr], y[tr])
    m_rf = score(y[te], rf.predict(Xflat[te]))

    # majority-class floor, taken from the training window
    maj = np.bincount(y[tr], minlength=n_cls).argmax()
    m_maj = score(y[te], np.full(len(te), maj))

    rows = []
    for name, m in [("EmbedMLP", m_deep), ("RandomForest", m_rf),
                    ("MajorityClass", m_maj)]:
        rows.append(dict(target=args.target, config=args.config, seed=args.seed,
                         model=name, n_train=len(tr), n_val=len(va),
                         n_test=len(te), **m))
        print(f"  {args.target:14s} {args.config:8s} {name:14s} "
              f"acc={m['accuracy']:6.2f}  macroF1={m['macro_f1']:6.2f}")

    pd.DataFrame(rows).to_csv(args.out, mode="a",
                              header=not os.path.exists(args.out), index=False)
    print(f"  train={len(tr)} val={len(va)} test={len(te)}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
