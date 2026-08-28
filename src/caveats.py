"""
caveats.py
==========
Multi-seed confirmation of the two diagnostics reported single-seed:

  --mode source    per-source (observed vs RF-imputed label) performance
  --mode ablate    feature-group ablation, any target

Results append to caveats_source.csv / caveats_ablate.csv.

Usage:
    python caveats.py --mode source --seed 0
    python caveats.py --mode ablate --target victim.country --seed 0
"""

import argparse, os
import numpy as np
import pandas as pd
import torch

from paths import CORPUS, RECOVERED_MAPPING, RESULTS
from sklearn.model_selection import train_test_split

from model_v2 import load, build_features, train, evaluate

ABLATIONS = {
    "all features": (),
    "minus action.*": ("action.",),
    "minus action/actor/asset/attribute": ("action.", "actor.", "asset.", "attribute."),
}


def fit_for(df, target, seed, drop_prefixes=()):
    d = df.drop(columns=[c for c in df.columns
                         if any(c.startswith(p) for p in drop_prefixes)]) \
        if drop_prefixes else df
    Xc, Xd, cards, y = build_features(d, target)
    n_cls = int(y.max()) + 1
    idx = np.arange(len(y))
    tmp, te = train_test_split(idx, test_size=0.20, random_state=seed)
    tr, va = train_test_split(tmp, test_size=0.20, random_state=seed)
    model, _ = train(Xc, Xd, y, tr, va, n_cls, seed,
                     hidden=(512, 256), use_weights=False)
    return model, torch.tensor(Xc), torch.tensor(Xd), y, te


def mode_source(df, seed, out=str(RESULTS / "caveats_source.csv")):
    rows = []
    year = df["timeline.incident.year"].values
    for target in ["Attack", "pattern"]:
        model, XcT, XdT, y, te = fit_for(df, target, seed)
        for era, sub in [("VERIS<=2017", te[year[te] <= 2017]),
                         ("Hack>=2018", te[year[te] >= 2018])]:
            if len(sub) == 0:
                continue
            _, m = evaluate(model, XcT[sub], XdT[sub], y[sub])
            imputed = (target == "Attack" and era.startswith("VERIS")) or \
                      (target == "pattern" and era.startswith("Hack"))
            rows.append(dict(target=target, seed=seed, era=era,
                             label="imputed" if imputed else "observed",
                             n=len(sub), **m))
            print(f"  {target:8s} {era:12s} {'IMPUTED' if imputed else 'observed':9s} "
                  f"n={len(sub):5d} acc={m['accuracy']:6.2f} macroF1={m['macro_f1']:6.2f}")
    pd.DataFrame(rows).to_csv(out, mode="a", header=not os.path.exists(out), index=False)


def mode_ablate(df, target, seed, out=str(RESULTS / "caveats_ablate.csv")):
    rows = []
    for tag, pref in ABLATIONS.items():
        model, XcT, XdT, y, te = fit_for(df, target, seed, pref)
        _, m = evaluate(model, XcT[te], XdT[te], y[te])
        rows.append(dict(target=target, seed=seed, ablation=tag, **m))
        print(f"  {target:14s} {tag:36s} acc={m['accuracy']:6.2f} macroF1={m['macro_f1']:6.2f}")
    pd.DataFrame(rows).to_csv(out, mode="a", header=not os.path.exists(out), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["source", "ablate"])
    ap.add_argument("--target", default="victim.country")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    df = load()
    if a.mode == "source":
        mode_source(df, a.seed)
    else:
        mode_ablate(df, a.target, a.seed)
