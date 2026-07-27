"""
Shared, registration-free scalar feature extraction.

Imported by BOTH the local analysis script and the container submission,
so the training-time and inference-time features cannot drift apart.

NOTE: this module must never print anything. It runs inside the container.
"""

import os

import nibabel as nib
import numpy as np
from scipy import ndimage, stats

FEATURE_NAMES = ["sbr", "sbr_peak", "peak_sharpness", "hot_frac",
                 "sbr_hi", "sbr_lo", "asym", "skew", "kurt"]


def foreground_mask(vol):
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
    profile = mask.sum(axis=tuple(i for i in range(mask.ndim) if i != lr_axis))
    total = profile.sum()
    if total == 0:
        return mask.shape[lr_axis] // 2
    return int(np.searchsorted(np.cumsum(profile), total / 2.0))


def _side_ratio(vol, mask, bg, sl):
    sub = vol[sl][mask[sl]]
    if sub.size < 50 or not np.isfinite(bg) or bg <= 0:
        return np.nan
    return float(sub[sub >= np.percentile(sub, 99.0)].mean() / bg)


def features_from_path(path):
    """Returns (uid, feature_dict). Never raises; returns ok=0 on failure."""
    uid = os.path.basename(path).split(".")[0]
    try:
        img = nib.as_closest_canonical(nib.load(path))
        vol = np.asarray(img.dataobj, dtype=np.float32)
        spacing = np.abs(img.header.get_zooms()[:3])

        mask = foreground_mask(vol)
        n_mask = int(mask.sum())
        if n_mask < 500:
            return uid, {"uid": uid, "ok": 0}

        vals = vol[mask]
        lo, hi = np.percentile(vals, [25.0, 60.0])
        band = vals[(vals >= lo) & (vals <= hi)]
        bg = float(band.mean()) if band.size else np.nan

        top05 = float(vals[vals >= np.percentile(vals, 99.5)].mean())
        top1 = float(vals[vals >= np.percentile(vals, 99.0)].mean())
        top10 = float(vals[vals >= np.percentile(vals, 90.0)].mean())

        mid = find_midline(mask, 0)
        sbr_l = _side_ratio(vol, mask, bg, (slice(0, mid),))
        sbr_r = _side_ratio(vol, mask, bg, (slice(mid, None),))
        sbr_hi = float(np.nanmax([sbr_l, sbr_r]))
        sbr_lo = float(np.nanmin([sbr_l, sbr_r]))

        return uid, {
            "uid": uid, "ok": 1,
            "sbr": top1 / bg,
            "sbr_peak": top05 / bg,
            "peak_sharpness": top05 / top10,
            "hot_frac": float((vals > 2.0 * bg).sum()) / n_mask,
            "sbr_hi": sbr_hi,
            "sbr_lo": sbr_lo,
            "asym": (sbr_hi - sbr_lo) / (sbr_hi + sbr_lo + 1e-8),
            "skew": float(stats.skew(vals)),
            "kurt": float(stats.kurtosis(vals)),
            # cluster keys
            "shape": "x".join(map(str, vol.shape)),
            "spacing": "x".join(f"{s:.3f}" for s in spacing),
            "xy_spacing": f"{spacing[0]:.3f}",
        }
    except Exception:
        return uid, {"uid": uid, "ok": 0}


def cluster_key(row, kind):
    if kind == "shape_spacing":
        return f"{row['shape']}@{row['spacing']}"
    if kind == "spacing":
        return str(row["spacing"])
    if kind == "xy_spacing":
        return str(row["xy_spacing"])
    return "ALL"