"""
Feature-based submission entry point.

Logging policy: only static, content-free strings. No counts, no indices,
no values derived from the test data. This is enforced by convention here --
every print() below is a literal with no interpolation.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.special import erfinv

from sbr_features import FEATURE_NAMES, cluster_key, features_from_path

DATA_DIR = os.environ.get("DATA_DIR", "/code_execution/data")
NIFTI_DIR = os.path.join(DATA_DIR, "niftis")
FORMAT_CSV = os.path.join(DATA_DIR, "submission_format.csv")
OUT_CSV = os.environ.get("OUT_CSV", "submission.csv")
MODEL_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature_model.json")


def main():
    print("Loading model.")
    with open(MODEL_JSON) as f:
        M = json.load(f)
    feats = M["features"]

    print("Loading submission format.")
    sub = pd.read_csv(FORMAT_CSV)

    print("Extracting features.")
    paths = [os.path.join(NIFTI_DIR, f"{u}.nii.gz") for u in sub["uid"]]
    workers = min(16, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = [r for _, r in ex.map(features_from_path, paths, chunksize=4)]
    F = pd.DataFrame(rows).set_index("uid")

    print("Standardising.")
    ok = F["ok"] == 1
    keys = F.apply(lambda r: cluster_key(r, M["cluster_key"]) if r["ok"] == 1 else "NA", axis=1)
    X = np.zeros((len(F), len(feats)), dtype=np.float64)

    for j, c in enumerate(feats):
        v = pd.to_numeric(F[c], errors="coerce")
        g = M["global_stats"][c]
        col = (v - g["mean"]) / (g["std"] + 1e-8)          # global fallback
        if M["mode"] != "none":
            for k, idx in F.index.groupby(keys).items():
                idx = [i for i in idx if ok.loc[i]]
                if k == "NA" or len(idx) < M["min_cluster"]:
                    continue
                s = v.loc[idx]
                if M["mode"] == "z":
                    col.loc[idx] = (s - s.mean()) / (s.std() + 1e-8)
                else:
                    r = s.rank(pct=True).clip(0.001, 0.999)
                    col.loc[idx] = np.sqrt(2) * erfinv(2 * r.values - 1)
        X[:, j] = np.nan_to_num(col.values, nan=0.0, posinf=0.0, neginf=0.0)

    X = (X - np.array(M["post_mean"])) / np.array(M["post_std"])
    z = X @ np.array(M["coef"]) + M["intercept"]
    p = 1.0 / (1.0 + np.exp(-z))
    p = np.clip(p, M["clip"][0], M["clip"][1])

    # any volume that failed feature extraction gets the base rate, not a guess
    p[~ok.values] = 0.5

    print("Writing predictions.")
    sub = sub.set_index("uid")
    sub.loc[F.index, "is_pathologic"] = p
    sub.reset_index().to_csv(OUT_CSV, index=False)
    print("Done.")


if __name__ == "__main__":
    main()