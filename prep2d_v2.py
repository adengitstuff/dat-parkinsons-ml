#!/usr/bin/env python3
"""
Precompute 2.5D striatal images -- v2.

    python prep2d_v2.py --niftis ~/niftis/niftis_extracted --out ~/niftis/cache2d_v2

Changes from v1, each targeting one un-normalised nuisance factor:

  [POSE]   In-plane rotation is corrected by finding the angle that maximises
           left-right symmetry of the foreground MASK. Mask-based, so it cannot
           depend on how bright the striatum is -- the same discipline that fixed
           the centroid-crop bug.

  [SCALE]  v1 used three INDEPENDENT scale factors, so a 38-slice scan and a
           90-slice scan were stretched by different z-factors. v2 uses ONE
           isotropic factor and pads/crops the remainder. Anatomy keeps its
           shape; only field-of-view differs, and padding is honest about that.

  [SLAB]   Slab thickness is now a fixed physical fraction rather than a fixed
           voxel count, which only means the same thing once scaling is isotropic.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

HW = 224
NZ = 64
SLAB = 9
CLIP = 8.0
MAX_ROT = 20.0     # degrees; beyond this a "correction" is probably a bad fit


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


def symmetry_angle(mask, step=2.0):
    """
    Angle (deg) that best aligns the mask's mid-sagittal plane with the image
    axis. Uses the axial projection of the BINARY mask only.
    """
    proj = mask.sum(2).astype(np.float32)
    if proj.max() <= 0:
        return 0.0
    proj = proj / proj.max()
    # centre on the mask's own centre of mass so rotation is about the head
    com = ndimage.center_of_mass(proj)
    ctr = (np.array(proj.shape) - 1) / 2.0
    proj = ndimage.shift(proj, ctr - np.array(com), order=1, mode="constant")

    best_a, best_e = 0.0, np.inf
    for a in np.arange(-MAX_ROT, MAX_ROT + 1e-6, step):
        r = ndimage.rotate(proj, a, axes=(0, 1), reshape=False, order=1, mode="constant")
        e = float(np.abs(r - r[::-1]).mean())     # axis 0 is L/R after canonical
        if e < best_e:
            best_a, best_e = float(a), e
    return best_a


def volume_to_image(path, thr=0.30, correct_rotation=True):
    """Returns (3, HW, HW) float32, or None on failure."""
    img = nib.as_closest_canonical(nib.load(path))
    vol = np.asarray(img.dataobj, dtype=np.float32)

    mask = _mask(vol, thr)
    if mask is None:
        mask = _mask(vol, 0.15)
    if mask is None:
        return None

    # --- reference-band normalisation (before any geometry changes)
    vals = vol[mask]
    lo, hi = np.percentile(vals, [25.0, 60.0])
    band = vals[(vals >= lo) & (vals <= hi)]
    bg = float(band.mean()) if band.size else np.nan
    if not np.isfinite(bg) or bg <= 0:
        return None
    vol = np.clip(vol / bg, 0.0, CLIP)

    # --- [POSE] correct in-plane rotation, then re-derive the mask
    if correct_rotation:
        a = symmetry_angle(mask)
        if abs(a) > 1e-6:
            vol = ndimage.rotate(vol, a, axes=(0, 1), reshape=True,
                                 order=1, mode="constant", cval=0.0)
            m2 = _mask(vol, thr)
            mask = m2 if m2 is not None else _mask(vol, 0.15)
            if mask is None:
                return None

    # --- bounding box crop
    idx = np.where(mask)
    vol = vol[tuple(slice(i.min(), i.max() + 1) for i in idx)]

    # --- [SCALE] ONE isotropic factor, chosen so the in-plane extent fills HW
    d = np.array(vol.shape, dtype=np.float64)
    s = HW / max(d[0], d[1])
    vol = ndimage.zoom(vol, (s, s, s), order=1).astype(np.float32)

    # --- centre crop / pad to the fixed grid (padding = honest about missing FOV)
    vol = _fit(vol, (HW, HW, NZ))

    # --- peak striatal slab
    sums = vol.sum(axis=(0, 1))
    zc = int(np.argmax(sums))
    a0 = int(np.clip(zc - SLAB // 2, 0, NZ - SLAB))
    slab = vol[:, :, a0:a0 + SLAB]

    return np.stack([slab.mean(2), slab.max(2), vol[:, :, zc]]).astype(np.float32)


def _fit(v, target):
    """Centre crop or zero-pad each axis to `target`."""
    out = v
    for ax, t in enumerate(target):
        n = out.shape[ax]
        if n > t:
            a = (n - t) // 2
            out = np.take(out, range(a, a + t), axis=ax)
        elif n < t:
            before = (t - n) // 2
            pad = [(0, 0)] * out.ndim
            pad[ax] = (before, t - n - before)
            out = np.pad(out, pad, mode="constant")
    return out


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
    np.save(os.path.join(args.out, "images.npy"),
            np.stack([im for _, im, _ in good]))
    pd.DataFrame({"uid": [u for u, _, _ in good],
                  "cluster": [c for _, _, c in good]}).to_csv(
        os.path.join(args.out, "index.csv"), index=False)
    print("done")


if __name__ == "__main__":
    main()