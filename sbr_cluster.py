#!/usr/bin/env python3
"""
Does removing per-scanner offsets rescue the SBR signal?

    python sbr_cluster.py analyze --features features.csv --labels ~/niftis/train_labels.csv
    python sbr_cluster.py export  --features features.csv --labels ~/niftis/train_labels.csv \
                                  --key xy_spacing --mode rank --out feature_model.json

Standardisation uses FEATURES ONLY, never labels, so applying it to the test set
at inference is legitimate transductive normalisation, not leakage.
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sbr_features import FEATURE_NAMES as FEATS

MIN_CLUSTER = 25   # below this, fall back to global statistics


def add_keys(df):
    df["k_shape_spacing"] = df["shape"] + "@" + df["spacing"]
    df["k_spacing"] = df["spacing"]
    df["k_xy_spacing"] = df["spacing"].str.split("x").str[0]
    return df


def standardize(df, keycol, mode, ref=None):
    """
    mode 'none' | 'z' (within-cluster z-score) | 'rank' (within-cluster quantile).
    `ref` supplies fallback global stats when a cluster is too small.
    """
    X = df[FEATS].copy()
    if mode == "none":
        return X.values
    counts = df[keycol].value_counts()
    out = X.copy()
    for c in FEATS:
        for k, sub in df.groupby(keycol):
            idx = sub.index
            v = X.loc[idx, c]
            if len(sub) < MIN_CLUSTER:
                if ref is not None:
                    out.loc[idx, c] = (v - ref[c]["mean"]) / (ref[c]["std"] + 1e-8)
                else:
                    out.loc[idx, c] = (v - X[c].mean()) / (X[c].std() + 1e-8)
            elif mode == "z":
                out.loc[idx, c] = (v - v.mean()) / (v.std() + 1e-8)
            else:  # rank -> approx normal scores, robust to outliers & cluster size
                r = v.rank(pct=True).clip(0.001, 0.999)
                out.loc[idx, c] = np.sqrt(2) * np.vectorize(_erfinv)(2 * r - 1)
    return out.values


def _erfinv(x):
    from scipy.special import erfinv
    return float(erfinv(x))


def evaluate(df, y, keycol, mode, tag):
    X = standardize(df, keycol, mode)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    p = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    rand_ll, rand_auc = log_loss(y, p), roc_auc_score(y, p)

    # leave-one-cluster-out, standardising the held-out cluster on its own stats
    big = df["k_shape_spacing"].value_counts()
    oof, oofy = [], []
    for cl in big[big >= 40].index:
        m_te = df["k_shape_spacing"] == cl
        if df.loc[m_te, "y"].nunique() < 2:
            continue
        mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        mdl.fit(X[~m_te.values], y[~m_te.values])
        oof.append(mdl.predict_proba(X[m_te.values])[:, 1])
        oofy.append(y[m_te.values])
    if oof:
        o, oy = np.concatenate(oof), np.concatenate(oofy)
        loco_ll, loco_auc = log_loss(oy, o), roc_auc_score(oy, o)
    else:
        loco_ll = loco_auc = float("nan")

    print(f"  {tag:32s}  rand5f {rand_ll:.4f} / {rand_auc:.4f}"
          f"   LOCO {loco_ll:.4f} / {loco_auc:.4f}")
    return rand_ll


def cmd_analyze(args):
    df = add_keys(pd.read_csv(args.features)).merge(
        pd.read_csv(args.labels), on="uid").query("ok == 1").dropna(subset=FEATS)
    df = df.reset_index(drop=True)
    df["y"] = df["is_pathologic"].astype(int)
    y = df["y"].values

    for k in ["k_shape_spacing", "k_spacing", "k_xy_spacing"]:
        n = df[k].nunique()
        big = (df[k].value_counts() >= MIN_CLUSTER).sum()
        cov = df[k].value_counts()[df[k].value_counts() >= MIN_CLUSTER].sum()
        print(f"{k:18s} {n:4d} clusters, {big:3d} with n>={MIN_CLUSTER} "
              f"covering {cov}/{len(df)} rows")

    print("\n                                    logloss / AUC")
    evaluate(df, y, "k_shape_spacing", "none", "no standardisation")
    for k in ["k_shape_spacing", "k_spacing", "k_xy_spacing"]:
        for mode in ["z", "rank"]:
            evaluate(df, y, k, mode, f"{k[2:]} + {mode}")

    print("\n--- per-cluster AUC of standardised sbr (should now be sign-consistent) ---")
    Xs = standardize(df, "k_xy_spacing", "rank")
    df["_sbr_s"] = Xs[:, FEATS.index("sbr")]
    for k, sub in df.groupby("k_shape_spacing"):
        if len(sub) >= 40 and sub["y"].nunique() == 2:
            print(f"  n={len(sub):4d}  {k:38s} AUC {roc_auc_score(sub['y'], sub['_sbr_s']):.4f}")


def cmd_export(args):
    df = add_keys(pd.read_csv(args.features)).merge(
        pd.read_csv(args.labels), on="uid").query("ok == 1").dropna(subset=FEATS)
    df = df.reset_index(drop=True)
    y = df["is_pathologic"].astype(int).values
    keycol = "k_" + args.key

    X = standardize(df, keycol, args.mode)
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000).fit(scaler.transform(X), y)

    raw = df[FEATS]
    model = {
        "features": FEATS,
        "cluster_key": args.key,
        "mode": args.mode,
        "min_cluster": MIN_CLUSTER,
        "global_stats": {c: {"mean": float(raw[c].mean()), "std": float(raw[c].std())}
                         for c in FEATS},
        "post_mean": scaler.mean_.tolist(),
        "post_std": scaler.scale_.tolist(),
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "clip": [0.02, 0.98],
    }
    with open(args.out, "w") as f:
        json.dump(model, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    s = ap.add_subparsers(dest="cmd", required=True)
    a = s.add_parser("analyze")
    a.add_argument("--features", required=True)
    a.add_argument("--labels", required=True)
    a.set_defaults(func=cmd_analyze)
    e = s.add_parser("export")
    e.add_argument("--features", required=True)
    e.add_argument("--labels", required=True)
    e.add_argument("--key", default="xy_spacing",
                   choices=["shape_spacing", "spacing", "xy_spacing"])
    e.add_argument("--mode", default="rank", choices=["none", "z", "rank"])
    e.add_argument("--out", default="feature_model.json")
    e.set_defaults(func=cmd_export)
    args = ap.parse_args()
    args.func(args)