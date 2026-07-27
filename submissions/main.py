"""
Submission entry point for the 2D striatal-slab fold ensemble.

Logging policy: static literals only. No f-strings, no counts, no indices,
no values derived from the test data.

Zip layout :
    main.py
    model2d.py
    prep2d.py
    calibration.json
    weights/fold0.pt ... fold4.pt
"""

import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

from model2d import Wrapped
from prep2d import volume_to_image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", "/code_execution/data")
NIFTI_DIR = os.path.join(DATA_DIR, "niftis")
FORMAT_CSV = os.path.join(DATA_DIR, "submission_format.csv")
OUT_CSV = os.environ.get("OUT_CSV", "submission.csv")

IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
CLIP = 8.0
BATCH = 64
PROB_CLIP = (0.02, 0.98)


def _prep(path):
    """Runs in a worker process. Returns (uid, image or None)."""
    uid = os.path.basename(path).split(".")[0]
    try:
        im = volume_to_image(path)
    except Exception:
        im = None
    return uid, (None if im is None else im.astype(np.float16))


def _normalize(batch):
    x = np.clip(batch.astype(np.float32) / CLIP, 0.0, 1.0)
    return (x - IMNET_MEAN) / IMNET_STD


def main():
    print("Loading configuration.")
    cal = json.load(open(os.path.join(HERE, "calibration.json")))
    arch = cal.get("arch", "resnet18")
    ckpts = sorted(glob.glob(os.path.join(HERE, "weights", "fold*.pt")))
    if not ckpts:
        raise RuntimeError("no checkpoints found")

    print("Loading submission format.")
    sub = pd.read_csv(FORMAT_CSV)
    uids = sub["uid"].tolist()

    print("Preprocessing volumes.")
    paths = [os.path.join(NIFTI_DIR, f"{u}.nii.gz") for u in uids]
    workers = min(16, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = dict(ex.map(_prep, paths, chunksize=4))

    order = [u for u in uids if results.get(u) is not None]
    failed = [u for u in uids if results.get(u) is None]
    X = np.stack([results[u] for u in order]) if order else np.zeros((0, 3, 224, 224))

    print("Loading models.")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    for c in ckpts:
        m = Wrapped(arch=arch, pretrained=False)
        m.load_state_dict(torch.load(c, map_location="cpu"))
        models.append(m.to(dev).eval())

    print("Running inference.")
    probs = np.zeros(len(order), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, len(order), BATCH):
            xb = torch.from_numpy(_normalize(X[s:s + BATCH])).to(dev)
            acc = torch.zeros(xb.shape[0], device=dev)
            for m in models:
                if dev == "cuda":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        a = torch.sigmoid(m(xb)) + torch.sigmoid(m(torch.flip(xb, [3])))
                else:
                    a = torch.sigmoid(m(xb)) + torch.sigmoid(m(torch.flip(xb, [3])))
                acc += a.float() / 2.0
            probs[s:s + xb.shape[0]] = (acc / len(models)).cpu().numpy()

    print("Calibrating.")
    p = np.clip(probs, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    p = 1.0 / (1.0 + np.exp(-(cal["coef"] * logit + cal["intercept"])))
    p = np.clip(p, PROB_CLIP[0], PROB_CLIP[1])

    print("Writing predictions.")
    out = sub.set_index("uid")
    out.loc[order, "is_pathologic"] = p
    if failed:
        out.loc[failed, "is_pathologic"] = 0.5   # base rate, never a guess
    out.reset_index().to_csv(OUT_CSV, index=False)
    print("Done.")


if __name__ == "__main__":
    main()