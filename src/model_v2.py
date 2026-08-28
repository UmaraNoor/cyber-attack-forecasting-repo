"""
model_v2.py
===========
Rescue attempt for the deep forecasting framework.

Diagnosis of why the original deep models lost to Random Forest:

  (a) The 61-element feature vector was reshaped to (n_features, 1) and fed
      to Conv1D / RNN / LSTM. Feature index is not a time axis, so the
      convolution and recurrence were modelling an ordering that does not
      exist.
  (b) High-cardinality categoricals (victim.country = 189 levels,
      victim.industry.name, victim.victim_id) were label-encoded and then
      StandardScaler-ed, which asserts that country 47 lies numerically
      between 46 and 48. Trees split arbitrarily and are immune; dense
      layers are not.
  (c) Unweighted cross-entropy on a 306:1 imbalance drives the network to
      majority classes, which is what the macro-F1 collapse reflects.
  (d) A 16-unit bottleneck before an 11-way softmax is very little capacity.

Fixes applied here:

  1. Entity embeddings for every categorical field, sized
     min(50, (cardinality+1)//2), learned jointly with the classifier.
  2. Binary indicators passed straight through (already 0/1, no scaling).
  3. Cyclical encoding for day and month; year scaled to [0,1].
  4. Class-weighted cross-entropy (inverse frequency).
  5. Wider MLP head with batch norm and dropout.
  6. Early stopping on validation macro F1, restoring best weights.

Usage:
    python model_v2.py --target Attack --seed 0
"""

import argparse, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from attack_taxonomy import build_mapping, apply_mapping

from paths import CORPUS, RECOVERED_MAPPING, RESULTS
CAT_COLS = ["victim.country", "victim.industry.name"]
DATE_COLS = ["timeline.incident.day", "timeline.incident.month", "timeline.incident.year"]
torch.set_num_threads(1)


def load():
    if not CORPUS.exists():
        raise SystemExit(
            f"Corpus not found at {CORPUS}.\n"
            "This repository does not redistribute the corpus. Build it first:\n"
            "    python acquire_data.py\n"
            "    python build_corpus.py\n"
            "    python validate_corpus.py\n"
            "See data/README.md for the dataset notice and source terms.")
    df = pd.read_csv(CORPUS, low_memory=False)
    m = build_mapping(str(RECOVERED_MAPPING), df["Attack"].unique())
    df["Attack"] = apply_mapping(df["Attack"], m)
    for c in ["Attack", "pattern"] + CAT_COLS:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))
    return df


def build_features(df, target):
    """Returns categorical index matrix, dense matrix, cardinalities, y."""
    cats = [c for c in CAT_COLS if c != target]
    Xc = df[cats].values.astype(np.int64)
    cards = [int(df[c].max()) + 1 for c in cats]

    d, mth = df["timeline.incident.day"].values, df["timeline.incident.month"].values
    yr = df["timeline.incident.year"].values.astype(np.float32)
    dense = [np.sin(2*np.pi*d/31), np.cos(2*np.pi*d/31),
             np.sin(2*np.pi*mth/12), np.cos(2*np.pi*mth/12),
             (yr - yr.min()) / max(yr.max() - yr.min(), 1)]

    bin_cols = [c for c in df.columns
                if c not in CAT_COLS + DATE_COLS + ["Attack", "pattern", "victim.victim_id"]]
    binaries = df[bin_cols].values.astype(np.float32)
    Xd = np.column_stack(dense + [binaries]).astype(np.float32)

    # 'pattern' is a legitimate predictor for other targets; include as embedding
    if target != "pattern":
        Xc = np.column_stack([Xc, df["pattern"].values])
        cards.append(int(df["pattern"].max()) + 1)

    y = df[target].values.astype(np.int64)
    return Xc, Xd, cards, y


class EmbedMLP(nn.Module):
    def __init__(self, cards, n_dense, n_cls, hidden=(256, 128), p=0.3):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, min(50, (c + 1) // 2)) for c in cards])
        emb_dim = sum(e.embedding_dim for e in self.embs)
        layers, prev = [], emb_dim + n_dense
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
            prev = h
        layers.append(nn.Linear(prev, n_cls))
        self.mlp = nn.Sequential(*layers)

    def forward(self, xc, xd):
        e = torch.cat([emb(xc[:, i]) for i, emb in enumerate(self.embs)], dim=1)
        return self.mlp(torch.cat([e, xd], dim=1))


def evaluate(model, Xc, Xd, y):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(y), 1024):
            out.append(torch.softmax(model(Xc[i:i+1024], Xd[i:i+1024]), 1).numpy())
    P = np.vstack(out)
    p = P.argmax(1)
    return P, dict(
        accuracy=accuracy_score(y, p) * 100,
        macro_precision=precision_score(y, p, average="macro", zero_division=0) * 100,
        macro_recall=recall_score(y, p, average="macro", zero_division=0) * 100,
        macro_f1=f1_score(y, p, average="macro", zero_division=0) * 100,
        weighted_f1=f1_score(y, p, average="weighted", zero_division=0) * 100)


def train(Xc, Xd, y, tr, va, n_cls, seed, epochs=80, patience=12,
          hidden=(256, 128), use_weights=True, lr=1e-3, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    model = EmbedMLP([int(Xc[:, i].max()) + 1 for i in range(Xc.shape[1])],
                     Xd.shape[1], n_cls, hidden)

    cnt = np.bincount(y[tr], minlength=n_cls).astype(np.float32)
    w = (len(tr) / (n_cls * np.maximum(cnt, 1))) if use_weights else np.ones(n_cls, np.float32)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    XcT, XdT, yT = torch.tensor(Xc), torch.tensor(Xd), torch.tensor(y)
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(tr)
        for i in range(0, len(perm), 256):
            b = perm[i:i+256]
            opt.zero_grad()
            lossf(model(XcT[b], XdT[b]), yT[b]).backward()
            opt.step()
        _, m = evaluate(model, XcT[va], XdT[va], y[va])
        if m["macro_f1"] > best:
            best, bad = m["macro_f1"], 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(RESULTS / "results_v2.csv"))
    args = ap.parse_args()

    t0 = time.time()
    df = load()
    Xc, Xd, cards, y = build_features(df, args.target)
    n_cls = int(y.max()) + 1
    idx = np.arange(len(y))
    tmp, te = train_test_split(idx, test_size=0.20, random_state=args.seed)
    tr, va = train_test_split(tmp, test_size=0.20, random_state=args.seed)

    model, bestval = train(Xc, Xd, y, tr, va, n_cls, args.seed)
    XcT, XdT = torch.tensor(Xc), torch.tensor(Xd)
    _, m = evaluate(model, XcT[te], XdT[te], y[te])

    row = dict(target=args.target, seed=args.seed, model="EmbedMLP", **m)
    pd.DataFrame([row]).to_csv(args.out, mode="a",
                               header=not os.path.exists(args.out), index=False)
    print(f"  EmbedMLP  acc={m['accuracy']:.2f}  macroF1={m['macro_f1']:.2f} "
          f"(best val macroF1 {bestval:.2f})  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
