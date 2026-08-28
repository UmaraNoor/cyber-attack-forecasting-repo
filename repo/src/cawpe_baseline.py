"""
cawpe_baseline.py
=================
Baseline experiment for Major 5.

The manuscript reports that a policy-gradient ensemble weighting layer fails to
beat uniform averaging. That null result is uninterpretable on its own: it may
reflect a property of the task, or it may reflect a weak weighting
implementation. Distinguishing the two requires comparing against the
established reference point for learned ensemble weighting.

CAWPE (Large, Lines & Bagnall, Data Mining and Knowledge Discovery 33, 2019)
weights each ensemble member by an exponentiated estimate of its accuracy:

    w_i  =  acc_i ^ alpha            (alpha = 4 in the original paper)
    p(c) =  sum_i w_i * p_i(c) / sum_i w_i

with acc_i estimated by cross-validation on the training data. Here we estimate
acc_i on the held-out validation partition, which is the same signal the RL
agent receives, so the comparison isolates the weighting *rule* rather than the
information available to it.

Schemes compared, all over the identical three trained encoders:
    uniform        equal weights                       (current reference)
    accuracy       w_i proportional to validation accuracy
    cawpe          w_i proportional to (validation accuracy)^alpha
    cawpe_f1       as cawpe but on validation macro F1
    rl             policy-gradient weights            (the manuscript's layer)
    best_single    the single best member on validation (selection, not blending)

Usage:
    python cawpe_baseline.py --target Attack --seed 0
"""

import argparse, os, time
import numpy as np
import pandas as pd
import torch

from paths import CORPUS, RECOVERED_MAPPING, RESULTS
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)

from model_v2 import load, build_features
from ensemble_v3 import fit, probs, REINFORCEWeights

ALPHA = 4.0
torch.set_num_threads(1)


def score(y, p):
    return dict(
        accuracy=accuracy_score(y, p) * 100,
        macro_precision=precision_score(y, p, average="macro", zero_division=0) * 100,
        macro_recall=recall_score(y, p, average="macro", zero_division=0) * 100,
        macro_f1=f1_score(y, p, average="macro", zero_division=0) * 100,
        weighted_f1=f1_score(y, p, average="weighted", zero_division=0) * 100,
    )


def combine(P, w):
    w = np.asarray(w, dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.ones(len(w)) / len(w)
    return np.tensordot(w, P, axes=([0], [0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(RESULTS / "cawpe_results.csv"))
    args = ap.parse_args()

    t0 = time.time()
    df = load()
    Xc, Xd, cards, y = build_features(df, args.target)
    cards = [int(Xc[:, i].max()) + 1 for i in range(Xc.shape[1])]
    n_cls = int(y.max()) + 1
    idx = np.arange(len(y))
    tmp, te = train_test_split(idx, test_size=0.20, random_state=args.seed)
    tr, va = train_test_split(tmp, test_size=0.20, random_state=args.seed)
    XcT, XdT = torch.tensor(Xc), torch.tensor(Xd)

    # ---- train the three encoders once; every scheme reuses them ----------
    names = ["MLP", "CNN", "LSTM"]
    Pv, Pt, val_acc, val_f1 = [], [], [], []
    for nm in names:
        m = fit(nm, cards, Xc, Xd, y, tr, va, n_cls, args.seed)
        pv = probs(m, XcT[va], XdT[va], len(va))
        pt = probs(m, XcT[te], XdT[te], len(te))
        Pv.append(pv); Pt.append(pt)
        val_acc.append(accuracy_score(y[va], pv.argmax(1)))
        val_f1.append(f1_score(y[va], pv.argmax(1), average="macro", zero_division=0))
        print(f"  {nm:5s} val acc={val_acc[-1]*100:6.2f}  val macroF1={val_f1[-1]*100:6.2f}")
    Pv, Pt = np.array(Pv), np.array(Pt)
    val_acc, val_f1 = np.array(val_acc), np.array(val_f1)

    schemes = {}
    schemes["uniform"] = np.ones(3) / 3
    schemes["accuracy"] = val_acc
    schemes["cawpe"] = val_acc ** ALPHA
    schemes["cawpe_f1"] = np.maximum(val_f1, 1e-9) ** ALPHA
    best = np.zeros(3); best[int(val_f1.argmax())] = 1.0
    schemes["best_single"] = best

    np.random.seed(args.seed)
    agent = REINFORCEWeights(3, seed=args.seed).fit(Pv, y[va])
    schemes["rl"] = agent.w

    rows = []
    for name, w in schemes.items():
        pred = combine(Pt, w).argmax(1)
        rows.append(dict(target=args.target, seed=args.seed, scheme=name,
                         w_mlp=round(float(w[0] / np.sum(w)), 4),
                         w_cnn=round(float(w[1] / np.sum(w)), 4),
                         w_lstm=round(float(w[2] / np.sum(w)), 4),
                         **score(y[te], pred)))
        print(f"  {name:12s} acc={rows[-1]['accuracy']:6.2f}  "
              f"macroF1={rows[-1]['macro_f1']:6.2f}  "
              f"w=({rows[-1]['w_mlp']:.2f},{rows[-1]['w_cnn']:.2f},{rows[-1]['w_lstm']:.2f})")

    pd.DataFrame(rows).to_csv(args.out, mode="a",
                              header=not os.path.exists(args.out), index=False)
    print(f"  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
