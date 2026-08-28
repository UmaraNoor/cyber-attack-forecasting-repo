"""
circularity_check.py
====================
Two diagnostics on the unified corpus, both aimed at the same question:
how much of the measured performance reflects genuine forecasting signal,
and how much reflects structure introduced during dataset construction?

Diagnostic 1 - feature-group ablation
    In the VERIS schema, `pattern` is largely a derived label: it follows
    from the combination of action / actor / asset / attribute indicators
    by taxonomy rule. If removing those groups collapses pattern accuracy,
    the task is closer to recovering a rule than to forecasting.

Diagnostic 2 - per-source split
    Dataset A merges two sources with disjoint eras:
        VERIS        1971-2017   (Attack was RF-IMPUTED; pattern observed)
        Hackmageddon 2018-2023   (pattern was RF-IMPUTED; Attack observed)
    So for each target, roughly half the corpus carries an observed label
    and half carries a label that a Random Forest predicted from these same
    features. Scoring separately on the two halves shows whether the model
    is recovering the earlier imputation rather than learning the phenomenon.

Usage:
    python circularity_check.py
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from paths import CORPUS
from model_v2 import load, build_features, train, evaluate, CAT_COLS, DATE_COLS

GROUPS = {
    "action.*": "action.",
    "actor.*": "actor.",
    "asset.*": "asset.",
    "attribute.*": "attribute.",
}


def run(df, target, drop_prefixes=(), seed=0, tag=""):
    d = df.copy()
    if drop_prefixes:
        cols = [c for c in d.columns
                if any(c.startswith(p) for p in drop_prefixes)]
        d = d.drop(columns=cols)
    Xc, Xd, cards, y = build_features(d, target)
    n_cls = int(y.max()) + 1
    idx = np.arange(len(y))
    tmp, te = train_test_split(idx, test_size=0.20, random_state=seed)
    tr, va = train_test_split(tmp, test_size=0.20, random_state=seed)
    model, _ = train(Xc, Xd, y, tr, va, n_cls, seed,
                     hidden=(512, 256), use_weights=False)
    XcT, XdT = torch.tensor(Xc), torch.tensor(Xd)
    _, m = evaluate(model, XcT[te], XdT[te], y[te])
    print(f"  {tag:34s} acc={m['accuracy']:6.2f}  macroF1={m['macro_f1']:6.2f}")
    return model, te, y, Xc, Xd


def per_source(df, target, seed=0):
    """Score separately on VERIS-era and Hackmageddon-era test rows."""
    Xc, Xd, cards, y = build_features(df, target)
    n_cls = int(y.max()) + 1
    idx = np.arange(len(y))
    tmp, te = train_test_split(idx, test_size=0.20, random_state=seed)
    tr, va = train_test_split(tmp, test_size=0.20, random_state=seed)
    model, _ = train(Xc, Xd, y, tr, va, n_cls, seed,
                     hidden=(512, 256), use_weights=False)
    XcT, XdT = torch.tensor(Xc), torch.tensor(Xd)

    year = df["timeline.incident.year"].values
    veris = te[year[te] <= 2017]
    hack = te[year[te] >= 2018]

    for name, sub, note in [("VERIS era (<=2017)", veris, ""),
                            ("Hackmageddon era (>=2018)", hack, "")]:
        if len(sub) == 0:
            continue
        _, m = evaluate(model, XcT[sub], XdT[sub], y[sub])
        imputed = ((target == "Attack" and name.startswith("VERIS")) or
                   (target == "pattern" and name.startswith("Hack")))
        flag = "  <-- label was RF-IMPUTED" if imputed else "  (label observed)"
        print(f"    {name:28s} n={len(sub):5d}  acc={m['accuracy']:6.2f}  "
              f"macroF1={m['macro_f1']:6.2f}{flag}")


if __name__ == "__main__":
    df = load()

    print("=" * 70)
    print("DIAGNOSTIC 1 - feature-group ablation, target = pattern")
    print("=" * 70)
    run(df, "pattern", (), tag="all features")
    run(df, "pattern", ("action.",), tag="minus action.*")
    run(df, "pattern", ("action.", "actor.", "asset.", "attribute."),
        tag="minus action/actor/asset/attribute")

    print()
    print("=" * 70)
    print("DIAGNOSTIC 1b - same ablation, target = Attack")
    print("=" * 70)
    run(df, "Attack", (), tag="all features")
    run(df, "Attack", ("action.",), tag="minus action.*")
    run(df, "Attack", ("action.", "actor.", "asset.", "attribute."),
        tag="minus action/actor/asset/attribute")

    print()
    print("=" * 70)
    print("DIAGNOSTIC 2 - performance by source era")
    print("=" * 70)
    for t in ["pattern", "Attack"]:
        print(f"  target = {t}")
        per_source(df, t)
        print()
