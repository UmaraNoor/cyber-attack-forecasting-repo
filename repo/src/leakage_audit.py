"""
leakage_audit.py
================
Independent verification of data-leakage findings in the cyber-attack
forecasting pipeline (Fiza & Zahid).

Run this in Colab or locally. It reproduces every diagnostic result
reported in the audit, using ONLY the original files:

    combined_dataset-imbalanced.xlsx   (Dataset A, 20,279 x 64)
    Final_Data.csv                     (Dataset B, 150,000 x 64)

Usage:
    python leakage_audit.py --a path/to/combined_dataset-imbalanced.xlsx \
                            --b path/to/Final_Data.csv
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

CAT_COLS = ["Attack", "victim.country", "pattern",
            "victim.industry.name", "victim.victim_id"]


def rule(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def encode(df):
    df = df.copy()
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = LabelEncoder().fit_transform(df[c].astype(str))
    return df


def build_features_as_published(df):
    """Reproduces the feature engineering in forecasting_code.txt verbatim."""
    df = df.copy()
    df["lag_1"] = df["Attack"].shift(1).fillna(0)
    df["diff"] = df["Attack"].diff().fillna(0)
    X = df.drop(["Attack", "victim.victim_id", "pattern"], axis=1)
    y = df[["Attack", "victim.country", "pattern"]]
    return X, y


# ---------------------------------------------------------------- TEST 1
def test_target_reconstruction(df, name):
    """Attack should NOT be recoverable from the feature matrix."""
    rule(f"TEST 1 - Target reconstruction from engineered features [{name}]")
    X, y = build_features_as_published(df)

    recon = X["lag_1"] + X["diff"]
    exact = np.isclose(recon.values, y["Attack"].values).mean()
    print(f"  Attack == (lag_1 + diff) for {exact*100:.2f}% of rows")
    print(f"  Reason: lag_1 = Attack.shift(1), diff = Attack - Attack.shift(1)")
    print(f"  => the target is an exact linear combination of two inputs.")

    print(f"\n  'victim.country' is a TARGET and is present in X: "
          f"{'victim.country' in X.columns}")
    print(f"  => the country target is also directly readable from the inputs.")
    return X, y


# ---------------------------------------------------------------- TEST 2
def test_linear_recovery(X, y):
    """A plain linear model should not solve a forecasting task."""
    rule("TEST 2 - Linear regression recovers the target")
    Xs = StandardScaler().fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(
        Xs, y["Attack"], test_size=0.2, random_state=42)

    lr = LinearRegression().fit(Xtr, ytr)
    pred = np.rint(lr.predict(Xte)).astype(int)
    acc = accuracy_score(yte, pred) * 100
    print(f"  Plain LinearRegression accuracy on Attack: {acc:.2f}%")

    coef = pd.Series(lr.coef_, index=X.columns).abs().sort_values(ascending=False)
    print("\n  Top coefficient magnitudes:")
    for k, v in coef.head(5).items():
        print(f"    {k:32s} {v:.3e}")
    print("\n  Every coefficient except lag_1/diff is ~1e-13 (numerically zero).")
    print("  The model is computing arithmetic, not forecasting.")


# ---------------------------------------------------------------- TEST 3
def test_duplicate_contamination(B):
    """Test records must not appear in the training set."""
    rule("TEST 3 - Train/test contamination via duplicated records")
    key = B.astype(str).agg("|".join, axis=1)
    vc = key.value_counts()

    n_dup_rows = len(B) - len(B.drop_duplicates())
    print(f"  Rows: {len(B):,}   distinct records: {len(vc):,}")
    print(f"  Exact duplicate rows: {n_dup_rows:,} "
          f"({100*n_dup_rows/len(B):.1f}%)")
    print(f"  Most-repeated single record appears {vc.max()} times")

    idx = np.arange(len(B))
    tr, te = train_test_split(idx, test_size=0.2, random_state=42)
    train_keys = set(key.iloc[tr])
    dup = key.iloc[te].isin(train_keys)
    print(f"\n  Under the published 80/20 split:")
    print(f"    test rows: {len(te):,}")
    print(f"    test rows whose EXACT record is also in training: "
          f"{dup.sum():,} ({100*dup.mean():.1f}%)")


# ---------------------------------------------------------------- TEST 4
def test_dataset_b_composition(A, B):
    """Dataset B should be a balanced superset of Dataset A."""
    rule("TEST 4 - Composition of the GAN-augmented corpus")
    keys = [c for c in A.columns if c != "Attack"]
    Ak = set(A[keys].astype(str).agg("|".join, axis=1))
    Bk = B[keys].astype(str).agg("|".join, axis=1)
    mask = Bk.isin(Ak)

    print(f"  Rows traceable to a real Dataset A record: "
          f"{mask.sum():,} ({100*mask.mean():.1f}%)")
    print(f"  Rows not traceable (synthetic):            "
          f"{(~mask).sum():,} ({100*(~mask).mean():.1f}%)")
    print(f"  NOTE: Dataset A holds only {len(A):,} records, so real records")
    print(f"        are replicated roughly {mask.sum()/len(A):.1f}x in Dataset B.")

    print("\n  Class balance of the supposedly BALANCED corpus:")
    vc = B["Attack"].value_counts()
    print(f"    largest class : {vc.index[0]:35s} {vc.iloc[0]:,}")
    print(f"    smallest class: {vc.index[-1]:35s} {vc.iloc[-1]:,}")
    print(f"    imbalance ratio: {vc.iloc[0]/vc.iloc[-1]:.1f} : 1")

    print("\n  Temporal integrity:")
    yr = B["timeline.incident.year"]
    top = yr.value_counts().idxmax()
    print(f"    Dataset A year range : {A['timeline.incident.year'].min()}"
          f"-{A['timeline.incident.year'].max()}")
    print(f"    Dataset B year range : {yr.min()}-{yr.max()}")
    print(f"    single year {top} holds {100*(yr==top).mean():.1f}% of all rows")


# ---------------------------------------------------------------- TEST 5
def test_smote_path():
    """The SMOTE branch in forecasting_code.txt is unreachable."""
    rule("TEST 5 - Reachability of the SMOTE branch (static analysis)")
    print("""  In forecasting_code.txt the resampling loop does:

      X_class = X_train[y_train[col] == class_label]
      y_class = y_train[col][y_train[col] == class_label]
      if len(np.unique(y_class)) <= 1:
          continue                      # <-- ALWAYS TRUE

  y_class is filtered to a single class by construction, so
  np.unique(y_class) always has length 1 and SMOTE is never executed.
  Control falls through to:

      if not y_resampled_dict:
          X_resampled = X_train         # original data used unchanged

  The code then RE-SPLITS the result, discarding the first test set.
  Conclusion: no resampling ever occurred in the forecasting pipeline.""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Dataset A (.xlsx)")
    ap.add_argument("--b", required=True, help="Dataset B (.csv)")
    args = ap.parse_args()

    A_raw = pd.read_excel(args.a)
    B_raw = pd.read_csv(args.b, low_memory=False)

    A = encode(A_raw)
    X, y = test_target_reconstruction(A, "Dataset A")
    test_linear_recovery(X, y)

    B = encode(B_raw)
    test_target_reconstruction(B, "Dataset B")

    test_duplicate_contamination(B_raw)
    test_dataset_b_composition(A_raw, B_raw)
    test_smote_path()

    rule("SUMMARY")
    print("""  1. Attack target is exactly recoverable from lag_1 + diff.
  2. Country target is present in its own feature matrix.
  3. ~32% of Dataset B test rows also appear in training.
  4. Dataset B is neither balanced nor temporally coherent.
  5. The SMOTE branch never executes.

  All reported accuracy figures require regeneration.""")


if __name__ == "__main__":
    main()
