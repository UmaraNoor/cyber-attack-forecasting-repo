"""
rebuild_pipeline.py
===================
Leak-free reimplementation of the cyber-attack forecasting framework.

Corrections applied relative to the original implementation:

  1. lag_1 and diff are NOT created. In the original these were
     Attack.shift(1) and Attack.diff(), so Attack == lag_1 + diff exactly,
     making the target an arithmetic function of two inputs.
  2. Each target is excluded from its own feature matrix. In the original,
     victim.country was a target but remained in X.
  3. StandardScaler is fitted on the training partition only.
  4. Train / validation / test indices are frozen once and reused by every
     configuration, so all comparisons are on identical partitions.
  5. Evaluation is on real, naturally distributed data. No synthetic
     records, no resampling, no duplicated rows.
  6. Macro-averaged precision/recall/F1 are reported alongside accuracy,
     since accuracy alone conceals majority-class collapse.

Usage:
    python rebuild_pipeline.py --target Attack --seed 0
"""

import argparse, json, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)

from attack_taxonomy import build_mapping, apply_mapping

from paths import CORPUS, RECOVERED_MAPPING, RESULTS
TARGETS = ["Attack", "victim.country", "pattern"]
CAT_FEATURES = ["victim.country", "victim.industry.name", "victim.victim_id"]
EPOCHS, BATCH = 40, 64
torch.set_num_threads(1)


# ------------------------------------------------------------------ data
def load_data():
    if not CORPUS.exists():
        raise SystemExit(
            f"Corpus not found at {CORPUS}. Run acquire_data.py, then "
            "build_corpus.py. See data/README.md.")
    df = pd.read_csv(CORPUS, low_memory=False)
    mapping = build_mapping(str(RECOVERED_MAPPING), df["Attack"].unique())
    df["Attack"] = apply_mapping(df["Attack"], mapping)
    for c in CAT_FEATURES + ["Attack", "pattern"]:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))
    return df


def make_xy(df, target):
    """Feature matrix excluding the target itself and the free-text victim id.
    Critically: no lag_1, no diff."""
    drop = {target, "victim.victim_id"}
    X = df.drop(columns=[c for c in drop if c in df.columns])
    X = X.select_dtypes(include=[np.number])
    return X.values.astype(np.float32), df[target].values.astype(np.int64), list(X.columns)


def split_indices(n, seed):
    idx = np.arange(n)
    tmp, test = train_test_split(idx, test_size=0.20, random_state=seed)
    train, val = train_test_split(tmp, test_size=0.20, random_state=seed)
    return train, val, test


# ---------------------------------------------------------------- models
class CNN(nn.Module):
    def __init__(self, n_feat, n_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 3), nn.ReLU(),
            nn.MaxPool1d(2), nn.BatchNorm1d(32), nn.Dropout(0.3), nn.Flatten(),
            nn.Linear(32 * ((n_feat - 2) // 2), 16), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(16, n_cls))

    def forward(self, x):
        return self.net(x.unsqueeze(1))


class RNN(nn.Module):
    def __init__(self, n_feat, n_cls):
        super().__init__()
        self.r1 = nn.RNN(1, 32, batch_first=True, nonlinearity="relu")
        self.r2 = nn.RNN(32, 16, batch_first=True, nonlinearity="relu")
        self.bn = nn.BatchNorm1d(16)
        self.do = nn.Dropout(0.3)
        self.fc = nn.Linear(16, n_cls)

    def forward(self, x):
        h, _ = self.r1(x.unsqueeze(-1))
        h, _ = self.r2(h)
        return self.fc(self.do(self.bn(h[:, -1])))


class LSTM(nn.Module):
    def __init__(self, n_feat, n_cls):
        super().__init__()
        self.l = nn.LSTM(1, 128, batch_first=True)
        self.do = nn.Dropout(0.3)
        self.fc = nn.Linear(128, n_cls)

    def forward(self, x):
        h, _ = self.l(x.unsqueeze(-1))
        return self.fc(self.do(h[:, -1]))


ARCH = {"CNN": CNN, "RNN": RNN, "LSTM": LSTM}


def train_model(name, Xtr, ytr, Xva, yva, n_cls, seed):
    torch.manual_seed(seed)
    model = ARCH[name](Xtr.shape[1], n_cls)
    opt = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    lossf = nn.CrossEntropyLoss()
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
    Xva_t, yva_t = torch.tensor(Xva), torch.tensor(yva)
    n = len(Xtr_t)

    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[b]), ytr_t[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = lossf(model(Xva_t), yva_t).item()
        sched.step(vl)
    return model


@torch.no_grad()
def probs(model, X):
    model.eval()
    out = []
    Xt = torch.tensor(X)
    for i in range(0, len(Xt), 512):
        out.append(torch.softmax(model(Xt[i:i + 512]), dim=1).numpy())
    return np.vstack(out)


# ------------------------------------------------------------ RL ensemble
class RLAgent:
    """Policy-gradient weighting of ensemble members, per Eqs. 5-7."""

    def __init__(self, n_models, lr=0.005):
        self.w = np.ones(n_models) / n_models
        self.lr = lr

    def fit(self, P_val, y_val, passes=3):
        n = len(y_val)
        for _ in range(passes):
            for i in np.random.permutation(n):
                pred = np.tensordot(self.w, P_val[:, i, :], axes=([0], [0]))
                correct = pred.argmax() == y_val[i]
                r = np.array([1.0 if P_val[m, i].argmax() == y_val[i] else -1.0
                              for m in range(len(self.w))])
                if not correct:
                    r *= 1.0
                self.w += self.lr * r / n
                self.w = np.clip(self.w, 0, 1)
                s = self.w.sum()
                self.w = self.w / s if s > 0 else np.ones_like(self.w) / len(self.w)
        return self

    def predict(self, P):
        return np.tensordot(self.w, P, axes=([0], [0]))


# --------------------------------------------------------------- scoring
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
    ap.add_argument("--target", required=True, choices=TARGETS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(RESULTS / "results.csv"))
    args = ap.parse_args()

    t0 = time.time()
    df = load_data()
    X, y, cols = make_xy(df, args.target)
    n_cls = int(y.max()) + 1
    tr, va, te = split_indices(len(X), args.seed)

    sc = StandardScaler().fit(X[tr])                 # fitted on TRAIN only
    Xtr, Xva, Xte = sc.transform(X[tr]), sc.transform(X[va]), sc.transform(X[te])
    ytr, yva, yte = y[tr], y[va], y[te]

    rows = []

    # individual architectures + collected probabilities
    P_val, P_test = [], []
    for name in ["CNN", "RNN", "LSTM"]:
        m = train_model(name, Xtr, ytr, Xva, yva, n_cls, args.seed)
        pv, pt = probs(m, Xva), probs(m, Xte)
        P_val.append(pv); P_test.append(pt)
        rows.append(dict(target=args.target, seed=args.seed, model=name,
                         **score(yte, pt.argmax(1))))
        print(f"  {name:22s} acc={rows[-1]['accuracy']:6.2f} "
              f"macroF1={rows[-1]['macro_f1']:6.2f}  [{time.time()-t0:.0f}s]")

    P_val, P_test = np.array(P_val), np.array(P_test)

    # uniform-average ensemble
    rows.append(dict(target=args.target, seed=args.seed, model="Ensemble-uniform",
                     **score(yte, P_test.mean(0).argmax(1))))

    # majority vote
    votes = np.stack([p.argmax(1) for p in P_test])
    maj = np.array([np.bincount(votes[:, i], minlength=n_cls).argmax()
                    for i in range(votes.shape[1])])
    rows.append(dict(target=args.target, seed=args.seed, model="Ensemble-majority",
                     **score(yte, maj)))

    # RL-weighted ensemble
    np.random.seed(args.seed)
    agent = RLAgent(3).fit(P_val, yva)
    rows.append(dict(target=args.target, seed=args.seed, model="Ensemble-RL",
                     w_cnn=agent.w[0], w_rnn=agent.w[1], w_lstm=agent.w[2],
                     **score(yte, agent.predict(P_test).argmax(1))))
    print(f"  RL weights CNN/RNN/LSTM = "
          f"{agent.w[0]:.3f}/{agent.w[1]:.3f}/{agent.w[2]:.3f}")

    # Random Forest baseline
    rf = RandomForestClassifier(n_estimators=100, random_state=args.seed,
                                n_jobs=1).fit(Xtr, ytr)
    rows.append(dict(target=args.target, seed=args.seed, model="RandomForest",
                     **score(yte, rf.predict(Xte))))

    # majority-class reference
    maj_cls = np.bincount(ytr).argmax()
    rows.append(dict(target=args.target, seed=args.seed, model="MajorityClass",
                     **score(yte, np.full_like(yte, maj_cls))))

    out = pd.DataFrame(rows)
    hdr = not os.path.exists(args.out)
    out.to_csv(args.out, mode="a", header=hdr, index=False)
    print(out[["model", "accuracy", "macro_f1"]].to_string(index=False))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
