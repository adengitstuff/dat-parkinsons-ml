#!/usr/bin/env python3
"""
Precompute 2.5D striatal images from 3D volumes.

    python prep2d.py --niftis ~/niftis/niftis_extracted --out ~/niftis/cache2d

Pipeline per volume (all numpy, no gradients anywhere):
  1. canonical RAS  -> axis0=L/R, axis1=P/A, axis2=I/S (axial slices along axis2)
  2. foreground mask -> bounding box crop        (kills FOV/padding variation)
  3. resize crop to a FIXED voxel grid           (kills the broken header spacing)
  4. divide by the in-brain reference band       (keeps magnitude = the signal)
  5. find the peak axial slab, project to 3 channels

Step 3 is why we never need to know the true mm spacing: every brain ends up the
same size in voxels because it was scaled by its own anatomy, not by metadata.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

HW = 224          # in-plane output size
NZ = 64           # axial slices after resampling
SLAB = 9          # slices in the striatal slab (~20mm of a ~150mm brain)
CLIP = 8.0        # max multiple of background we keep


def _mask(vol, thr):
    p99 = np.percentile(vol, 99.0)
    if p99 <= 0:
        return None
    m = vol > (thr * p99)
    if m.sum() < 500:
        return None
    m = ndimage.binary_opening(m, iterations=1)
    lab, n = ndimage.label(m)
    if n == 0:
        return None
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def volume_to_image(path, thr=0.30):
    """Returns (3, HW, HW) float32, or None on failure."""
    img = nib.as_closest_canonical(nib.load(path))
    vol = np.asarray(img.dataobj, dtype=np.float32)

    mask = _mask(vol, thr)
    if mask is None:                       # fall back to a looser threshold
        mask = _mask(vol, 0.15)
    if mask is None:
        return None

    # --- reference-band normalisation, computed BEFORE cropping distorts percentiles
    vals = vol[mask]
    lo, hi = np.percentile(vals, [25.0, 60.0])
    band = vals[(vals >= lo) & (vals <= hi)]
    bg = float(band.mean()) if band.size else np.nan
    if not np.isfinite(bg) or bg <= 0:
        return None
    vol = np.clip(vol / bg, 0.0, CLIP)

    # --- bounding box crop
    idx = np.where(mask)
    sl = tuple(slice(i.min(), i.max() + 1) for i in idx)
    vol = vol[sl]

    # --- resize to a fixed grid
    z = (HW / vol.shape[0], HW / vol.shape[1], NZ / vol.shape[2])
    vol = ndimage.zoom(vol, z, order=1).astype(np.float32)
    if vol.shape != (HW, HW, NZ):          # zoom rounding
        pad = [(0, max(0, t - s)) for s, t in zip(vol.shape, (HW, HW, NZ))]
        vol = np.pad(vol, pad)[:HW, :HW, :NZ]

    # --- peak striatal slab
    sums = vol.sum(axis=(0, 1))
    zc = int(np.argmax(sums))
    a = int(np.clip(zc - SLAB // 2, 0, NZ - SLAB))
    slab = vol[:, :, a:a + SLAB]

    return np.stack([slab.mean(2), slab.max(2), vol[:, :, zc]]).astype(np.float32)


def _one(path):
    uid = os.path.basename(path).split(".")[0]
    try:
        im = volume_to_image(path)
    except Exception:
        im = None
    if im is None:
        return uid, None, ""
    sp = np.abs(nib.load(path).header.get_zooms()[:3])
    return uid, im.astype(np.float16), f"{sp[0]:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--niftis", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    paths = sorted(os.path.join(args.niftis, f)
                   for f in os.listdir(args.niftis) if f.endswith((".nii", ".nii.gz")))
    print(f"found {len(paths)} volumes")

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(_one, paths, chunksize=4))

    good = [(u, im, c) for u, im, c in res if im is not None]
    print(f"ok {len(good)} / {len(res)}")

    uids = [u for u, _, _ in good]
    X = np.stack([im for _, im, _ in good])
    np.save(os.path.join(args.out, "images.npy"), X)
    pd.DataFrame({"uid": uids, "cluster": [c for _, _, c in good]}).to_csv(
        os.path.join(args.out, "index.csv"), index=False)
    print(f"wrote {X.shape} to {args.out}")


if __name__ == "__main__":
    main()