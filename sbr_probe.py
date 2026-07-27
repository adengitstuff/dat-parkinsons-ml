#!/usr/bin/env python3
"""
SBR proxy probe: registration-free scalar features + geometry audit.

Usage:
    python sbr_probe.py extract --niftis data/niftis --out features.csv [--workers 24]
    python sbr_probe.py analyze --features features.csv --labels data/train_labels.csv

Design notes:
  - Every volume is reoriented to canonical RAS, so axis 0 is always Left-Right.
  - The foreground mask and the midline are derived from BINARY mask geometry only,
    never from intensity magnitude, so they cannot shift with the label.
  - All intensity features are RATIOS (self-referencing), so they are invariant to
    the raw-vs-rescaled uint16 split.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage, stats


# ---------------------------------------------------------------- extraction

def foreground_mask(vol):
    """Head/brain mask from a low intensity threshold + largest connected component."""
    p99 = np.percentile(vol, 99.0)
    if p99 <= 0:
        return np.zeros_like(vol, dtype=bool)
    mask = vol > (0.15 * p99)
    mask = ndimage.binary_opening(mask, iterations=1)
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def find_midline(mask, lr_axis=0):
    """
    Index along lr_axis that best balances mask volume left vs right.
    Uses the binary mask only -> label-neutral.
    """
    profile = mask.sum(axis=tuple(i for i in range(mask.ndim) if i != lr_axis))
    total = profile.sum()
    if total == 0:
        return mask.shape[lr_axis] // 2
    cum = np.cumsum(profile)
    return int(np.searchsorted(cum, total / 2.0))


def side_ratio(vol, mask, bg, sl):
    """Mean of the top-1% brightest in-mask voxels on one side, over background."""
    sub = vol[sl][mask[sl]]
    if sub.size < 50 or bg <= 0:
        return np.nan
    return float(sub[sub >= np.percentile(sub, 99.0)].mean() / bg)


def extract_one(path):
    uid = os.path.basename(path).split(".")[0]
    try:
        img = nib.as_closest_canonical(nib.load(path))
        vol = np.asarray(img.dataobj, dtype=np.float32)
        spacing = np.abs(img.header.get_zooms()[:3])

        mask = foreground_mask(vol)
        n_mask = int(mask.sum())
        if n_mask < 500:
            return {"uid": uid, "ok": 0}

        vals = vol[mask]
        # --- reference region: robust mid-band of in-brain voxels (non-specific binding)
        lo, hi = np.percentile(vals, [25.0, 60.0])
        band = vals[(vals >= lo) & (vals <= hi)]
        bg = float(band.mean()) if band.size else np.nan

        top05 = float(vals[vals >= np.percentile(vals, 99.5)].mean())
        top1 = float(vals[vals >= np.percentile(vals, 99.0)].mean())
        top10 = float(vals[vals >= np.percentile(vals, 90.0)].mean())

        # --- laterality, canonicalised as order statistics (affected / less-affected)
        mid = find_midline(mask, 0)
        l_sl = (slice(0, mid),)
        r_sl = (slice(mid, None),)
        sbr_l, sbr_r = side_ratio(vol, mask, bg, l_sl), side_ratio(vol, mask, bg, r_sl)
        sbr_hi = np.nanmax([sbr_l, sbr_r])
        sbr_lo = np.nanmin([sbr_l, sbr_r])

        # --- geometry audit (does the stored spacing imply a plausible head?)
        idx = np.where(mask)
        vox_extent = np.array([i.max() - i.min() + 1 for i in idx], dtype=float)

        return {
            "uid": uid,
            "ok": 1,
            # intensity features (all scale-free ratios)
            "sbr": top1 / bg,
            "sbr_peak": top05 / bg,
            "peak_sharpness": top05 / top10,
            "hot_frac": float((vals > 2.0 * bg).sum()) / n_mask,
            "sbr_hi": sbr_hi,
            "sbr_lo": sbr_lo,
            "asym": (sbr_hi - sbr_lo) / (sbr_hi + sbr_lo + 1e-8),
            "skew": float(stats.skew(vals)),
            "kurt": float(stats.kurtosis(vals)),
            # audit fields
            "shape": "x".join(map(str, vol.shape)),
            "spacing": "x".join(f"{s:.3f}" for s in spacing),
            "raw_max": float(vol.max()),
            "mask_frac": n_mask / float(vol.size),
            "extent_lr_mm": vox_extent[0] * spacing[0],
            "extent_ap_mm": vox_extent[1] * spacing[1],
            "extent_is_mm": vox_extent[2] * spacing[2],
            "extent_lr_vox": vox_extent[0],
            "implied_spacing_lr": 175.0 / max(vox_extent[0], 1.0),
        }
    except Exception as e:
        return {"uid": uid, "ok": 0, "err": type(e).__name__ + ": " + str(e)[:80]}


def cmd_extract(args):
    paths = sorted(
        os.path.join(args.niftis, f)
        for f in os.listdir(args.niftis)
        if f.endswith(".nii.gz") or f.endswith(".nii")
    )
    print(f"found {len(paths)} volumes")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(extract_one, paths, chunksize=4))
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({int(df['ok'].sum())} ok / {len(df)})")


# ------------------------------------------------------------------ analysis

FEATS = ["sbr", "sbr_peak", "peak_sharpness", "hot_frac",
         "sbr_hi", "sbr_lo", "asym", "skew", "kurt"]


def cmd_analyze(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    f = pd.read_csv(args.features)
    y_df = pd.read_csv(args.labels)
    df = f.merge(y_df, on="uid").query("ok == 1").dropna(subset=FEATS)
    y = df["is_pathologic"].values.astype(int)
    print(f"\nmerged: {len(df)} rows, positive rate {y.mean():.3f}")

    print("\n--- geometry audit: stored spacing vs. implied spacing, by cluster ---")
    g = df.groupby(["shape", "spacing"]).agg(
        n=("uid", "size"),
        extent_lr_mm=("extent_lr_mm", "median"),
        extent_lr_vox=("extent_lr_vox", "median"),
        implied_spacing=("implied_spacing_lr", "median"),
        mask_frac=("mask_frac", "median"),
        pos_rate=("is_pathologic", "mean"),
    ).sort_values("n", ascending=False)
    print(g.head(12).to_string())
    print("\n  (extent_lr_mm should be ~150-190 for a real head. "
          "implied_spacing = what the spacing WOULD be if the head were 175mm wide.)")

    print("\n--- univariate AUC (0.5 = useless, <0.5 = inversely predictive) ---")
    for c in FEATS:
        print(f"  {c:16s} AUC {roc_auc_score(y, df[c].values):.4f}")

    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    print("\n--- univariate logistic regression, 5-fold CV ---")
    for c in ["sbr", "sbr_lo", "hot_frac"]:
        pipe = make_pipeline(StandardScaler(), LogisticRegression())
        p = cross_val_predict(pipe, df[[c]].values, y, cv=cv, method="predict_proba")[:, 1]
        print(f"  {c:16s} logloss {log_loss(y, p):.4f}   AUC {roc_auc_score(y, p):.4f}")

    print("\n--- all features, logistic regression, 5-fold CV ---")
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    p = cross_val_predict(pipe, df[FEATS].values, y, cv=cv, method="predict_proba")[:, 1]
    print(f"  logloss {log_loss(y, p):.4f}   AUC {roc_auc_score(y, p):.4f}")

    print("\n--- per-cluster AUC of `sbr` (does the feature work WITHIN a scanner?) ---")
    for (shp, spc), sub in df.groupby(["shape", "spacing"]):
        if len(sub) >= 40 and sub["is_pathologic"].nunique() == 2:
            print(f"  n={len(sub):4d}  {shp:>15s} @ {spc:<18s} "
                  f"AUC {roc_auc_score(sub['is_pathologic'], sub['sbr']):.4f}   "
                  f"median sbr {sub['sbr'].median():.3f}")

    print("\n--- leave-one-cluster-out CV (the honest site-shift estimate) ---")
    df["_cl"] = df["shape"] + "@" + df["spacing"]
    big = df["_cl"].value_counts()
    keep = set(big[big >= 40].index)
    oof, oofy = [], []
    for cl in keep:
        tr, te = df[df["_cl"] != cl], df[df["_cl"] == cl]
        if te["is_pathologic"].nunique() < 2:
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        m.fit(tr[FEATS].values, tr["is_pathologic"].values)
        oof.append(m.predict_proba(te[FEATS].values)[:, 1])
        oofy.append(te["is_pathologic"].values)
    if oof:
        oof, oofy = np.concatenate(oof), np.concatenate(oofy)
        print(f"  logloss {log_loss(oofy, oof):.4f}   AUC {roc_auc_score(oofy, oof):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--niftis", required=True)
    e.add_argument("--out", default="features.csv")
    e.add_argument("--workers", type=int, default=16)
    e.set_defaults(func=cmd_extract)
    a = sub.add_parser("analyze")
    a.add_argument("--features", required=True)
    a.add_argument("--labels", required=True)
    a.set_defaults(func=cmd_analyze)
    args = ap.parse_args()
    args.func(args)