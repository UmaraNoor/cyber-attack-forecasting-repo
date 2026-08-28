"""
ensemble_v3.py
==============
Deep ensemble with a reinforcement learning weighting layer that actually
trains, built on the embedding representation from model_v2.

Ensemble members share the entity-embedding front end but differ in how they
model interactions between embedded fields:

  MLP       - concatenates all field embeddings, dense interaction
  CNN       - 1D convolution across the sequence of field-embedding tokens,
              capturing local groups of related fields
  LSTM      - recurrent pass over the field-embedding tokens

Treating the embedded fields as a token sequence is the defensible version of
the original CNN/RNN/LSTM design. The original convolved across raw
label-encoded feature indices, which imposes an ordering that does not exist.

RL weighting layer
------------------
The original agent defined update_weights() but never called it, so weights
stayed at their 1/3 initialisation and the "RL ensemble" was a uniform mean.
Here the weights are parameterised as a softmax over logits and trained by
REINFORCE on the validation set:

    w      = softmax(theta)
    m ~ w                       (sample an ensemble member)
    r      = +1 if member m classifies the instance correctly else -1
    theta += alpha * r * (onehot(m) - w)        [policy gradient]

with a running-mean baseline to reduce variance. This converges to weights
proportional to validated per-member competence.

Usage:
    python ensemble_v3.py --target Attack --seed 0
"""

import argparse, os, time
import numpy as np
import pandas as pd
import torch

from paths import CORPUS, RECOVERED_MAPPING, RESULTS
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)

from model_v2 import load, build_features

torch.set_num_threads(1)


# ------------------------------------------------------------- encoders
class Base(nn.Module):
    def __init__(self, cards, n_dense, emb=32):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, emb) for c in cards])
        self.emb_dim, self.n_fields, self.n_dense = emb, len(cards), n_dense

    def tokens(self, xc):
        return torch.stack([e(xc[:, i]) for i, e in enumerate(self.embs)], dim=1)


class MLPNet(Base):
    def __init__(self, cards, n_dense, n_cls, emb=32, p=0.3):
        super().__init__(cards, n_dense, emb)
        d = self.n_fields * emb + n_dense
        self.head = nn.Sequential(
            nn.Linear(d, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(p),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(p),
            nn.Linear(256, n_cls))

    def forward(self, xc, xd):
        return self.head(torch.cat([self.tokens(xc).flatten(1), xd], 1))


class CNNNet(Base):
    def __init__(self, cards, n_dense, n_cls, emb=32, p=0.3):
        super().__init__(cards, n_dense, emb)
        self.conv = nn.Sequential(
            nn.Conv1d(emb, 64, 2, padding=1), nn.ReLU(), nn.BatchNorm1d(64),
            nn.AdaptiveMaxPool1d(1))
        self.head = nn.Sequential(
            nn.Linear(64 + n_dense, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(p), nn.Linear(256, n_cls))

    def forward(self, xc, xd):
        h = self.conv(self.tokens(xc).transpose(1, 2)).squeeze(-1)
        return self.head(torch.cat([h, xd], 1))


class LSTMNet(Base):
    def __init__(self, cards, n_dense, n_cls, emb=32, p=0.3):
        super().__init__(cards, n_dense, emb)
        self.lstm = nn.LSTM(emb, 128, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(256 + n_dense, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(p), nn.Linear(256, n_cls))

    def forward(self, xc, xd):
        h, _ = self.lstm(self.tokens(xc))
        return self.head(torch.cat([h[:, -1], xd], 1))


NETS = {"MLP": MLPNet, "CNN": CNNNet, "LSTM": LSTMNet}


# ------------------------------------------------------------- training
def probs(model, Xc, Xd, n):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, n, 1024):
            out.append(torch.softmax(model(Xc[i:i+1024], Xd[i:i+1024]), 1).numpy())
    return np.vstack(out)


def fit(name, cards, Xc, Xd, y, tr, va, n_cls, seed, epochs=60, patience=10):
    torch.manual_seed(seed); np.random.seed(seed)
    model = NETS[name](cards, Xd.shape[1], n_cls)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    XcT, XdT, yT = torch.tensor(Xc), torch.tensor(Xd), torch.tensor(y)
    best, state, bad = -1, None, 0
    for _ in range(epochs):
        perm = np.random.permutation(tr)
        model.train()
        for i in range(0, len(perm), 256):
            b = perm[i:i+256]
            opt.zero_grad(); lossf(model(XcT[b], XdT[b]), yT[b]).backward(); opt.step()
        P = probs(model, XcT[va], XdT[va], len(va))
        f = f1_score(y[va], P.argmax(1), average="macro", zero_division=0)
        if f > best:
            best, bad = f, 0
            state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(state)
    return model


# ------------------------------------------------------------ RL layer
class REINFORCEWeights:
    """Policy-gradient ensemble weighting. Unlike the original agent, this
    one is actually invoked and its weights move away from uniform."""

    def __init__(self, n_models, alpha=0.05, seed=0):
        self.theta = np.zeros(n_models)
        self.alpha, self.rng = alpha, np.random.default_rng(seed)

    @property
    def w(self):
        e = np.exp(self.theta - self.theta.max())
        return e / e.sum()

    def fit(self, P_val, y_val, passes=8):
        n, M = len(y_val), len(self.theta)
        baseline = 0.0
        for _ in range(passes):
            for i in self.rng.permutation(n):
                w = self.w
                m = self.rng.choice(M, p=w)
                r = 1.0 if P_val[m, i].argmax() == y_val[i] else -1.0
                baseline = 0.99 * baseline + 0.01 * r
                adv = r - baseline
                onehot = np.zeros(M); onehot[m] = 1.0
                self.theta += self.alpha * adv * (onehot - w)
        return self

    def predict(self, P):
        return np.tensordot(self.w, P, axes=([0], [0]))


def score(y, p):
    return dict(accuracy=accuracy_score(y, p) * 100,
                macro_precision=precision_score(y, p, average="macro", zero_division=0) * 100,
                macro_recall=recall_score(y, p, average="macro", zero_division=0) * 100,
                macro_f1=f1_score(y, p, average="macro", zero_division=0) * 100,
                weighted_f1=f1_score(y, p, average="weighted", zero_division=0) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(RESULTS / "results_v3.csv"))
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

    rows, Pv, Pt = [], [], []
    for name in ["MLP", "CNN", "LSTM"]:
        m = fit(name, cards, Xc, Xd, y, tr, va, n_cls, args.seed)
        pv = probs(m, XcT[va], XdT[va], len(va))
        pt = probs(m, XcT[te], XdT[te], len(te))
        Pv.append(pv); Pt.append(pt)
        rows.append(dict(target=args.target, seed=args.seed, model=name,
                         **score(y[te], pt.argmax(1))))
        print(f"  {name:18s} acc={rows[-1]['accuracy']:6.2f} "
              f"macroF1={rows[-1]['macro_f1']:6.2f}  [{time.time()-t0:.0f}s]")

    Pv, Pt = np.array(Pv), np.array(Pt)
    rows.append(dict(target=args.target, seed=args.seed, model="Ens-uniform",
                     **score(y[te], Pt.mean(0).argmax(1))))

    agent = REINFORCEWeights(3, seed=args.seed).fit(Pv, y[va])
    rows.append(dict(target=args.target, seed=args.seed, model="Ens-RL",
                     w_mlp=agent.w[0], w_cnn=agent.w[1], w_lstm=agent.w[2],
                     **score(y[te], agent.predict(Pt).argmax(1))))
    print(f"  RL weights MLP/CNN/LSTM = {agent.w[0]:.3f}/{agent.w[1]:.3f}/{agent.w[2]:.3f}")

    Xflat = np.column_stack([Xc, Xd])
    rf = RandomForestClassifier(n_estimators=100, random_state=args.seed,
                                n_jobs=1).fit(Xflat[tr], y[tr])
    rows.append(dict(target=args.target, seed=args.seed, model="RandomForest",
                     **score(y[te], rf.predict(Xflat[te]))))

    out = pd.DataFrame(rows)
    out.to_csv(args.out, mode="a", header=not os.path.exists(args.out), index=False)
    print(out[["model", "accuracy", "macro_f1"]].to_string(index=False))
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
