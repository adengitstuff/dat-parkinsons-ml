#!/usr/bin/env python3
"""
5-fold training of the 2D striatal-slab model.

    python train2d.py --cache ~/niftis/cache2d --labels ~/niftis/train_labels.csv \
                      --out ~/niftis/runs/r18_slab

Reports out-of-fold log loss, which is the number to compare against:
    18.9M-param 3D CNN   0.44   (leaderboard)
    9 scalar features    0.4412 (5-fold CV)
Both of those are global-intensity-statistic models. If preserved spatial
resolution is worth anything, this should be clearly below them.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

from model2d import Wrapped

IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
CLIP = 8.0


class SlabDS(Dataset):
    def __init__(self, X, y, train):
        self.X, self.y, self.train = X, y, train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        im = self.X[i].astype(np.float32)
        if self.train:
            if np.random.rand() < 0.5:                 # L/R flip: laterality has
                im = im[:, ::-1].copy()                # no consistent side in PD
            im = im * np.float32(np.random.uniform(0.9, 1.1))
        im = np.clip(im / CLIP, 0, 1)
        im = (im - IMNET_MEAN) / IMNET_STD
        t = torch.from_numpy(im)
        if self.train:
            ang = float(np.random.uniform(-10, 10))
            tr = [float(np.random.uniform(-0.06, 0.06) * im.shape[1]) for _ in range(2)]
            sc = float(np.random.uniform(0.92, 1.08))
            t = torch.from_numpy(
                np.ascontiguousarray(
                    _affine(t.numpy(), ang, tr, sc)))
        return t, torch.tensor(self.y[i], dtype=torch.float32)


def _affine(im, ang, tr, sc):
    from scipy import ndimage
    c = np.array(im.shape[1:]) / 2.0
    th = np.deg2rad(ang)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) / sc
    off = c - R @ (c + np.array(tr))
    return np.stack([ndimage.affine_transform(ch, R, off, order=1, mode="constant")
                     for ch in im])


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    out = []
    for x, _ in loader:
        x = x.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.sigmoid(model(x)) + torch.sigmoid(model(torch.flip(x, [3])))
        out.append((p / 2).float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    X = np.load(os.path.join(args.cache, "images.npy"))
    idx = pd.read_csv(os.path.join(args.cache, "index.csv"))
    lab = pd.read_csv(args.labels)
    idx = idx.merge(lab, on="uid")
    keep = idx.index.values
    y = idx["is_pathologic"].values.astype(np.float32)
    X = X[keep]
    print(f"data {X.shape}, positives {y.mean():.3f}")

    # stratify by label AND acquisition cluster, bucketing rare clusters
    cc = idx["cluster"].value_counts()
    cl = idx["cluster"].where(idx["cluster"].map(cc) >= 25, "other")
    skey = idx["is_pathologic"].astype(int).astype(str) + "_" + cl.astype(str)
    sc = skey.value_counts()
    skey = skey.where(skey.map(sc) >= args.folds, "rare")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    oof = np.zeros(len(y))
    kf = StratifiedKFold(args.folds, shuffle=True, random_state=42)

    for f, (tr, va) in enumerate(kf.split(X, skey)):
        model = Wrapped(arch=args.arch, pretrained=True).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr * 5, total_steps=args.epochs * max(1, len(tr) // args.bs + 1))
        lossf = nn.BCEWithLogitsLoss()

        dtr = DataLoader(SlabDS(X[tr], y[tr], True), batch_size=args.bs,
                         shuffle=True, num_workers=6, drop_last=True, pin_memory=True)
        dva = DataLoader(SlabDS(X[va], y[va], False), batch_size=64,
                         shuffle=False, num_workers=4, pin_memory=True)

        best, best_p = 9e9, None
        for ep in range(args.epochs):
            model.train()
            for xb, yb in dtr:
                xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = lossf(model(xb), yb)
                loss.backward()
                opt.step()
                sched.step()
            p = np.clip(predict(model, dva, dev), 1e-6, 1 - 1e-6)
            ll = log_loss(y[va], p, labels=[0, 1])
            if ll < best:
                best, best_p = ll, p
                torch.save(model.state_dict(), os.path.join(args.out, f"fold{f}.pt"))
            print(f"  fold {f} ep {ep:2d}  val logloss {ll:.4f}  best {best:.4f}")
        oof[va] = best_p
        print(f"fold {f} BEST {best:.4f}")

    oof = np.clip(oof, 1e-6, 1 - 1e-6)
    print(f"\nOOF logloss {log_loss(y, oof):.4f}   AUC {roc_auc_score(y, oof):.4f}")

    lr = LogisticRegression().fit(np.log(oof / (1 - oof)).reshape(-1, 1), y.astype(int))
    cal = {"coef": float(lr.coef_[0][0]), "intercept": float(lr.intercept_[0]),
           "arch": args.arch}
    json.dump(cal, open(os.path.join(args.out, "calibration.json"), "w"))
    pc = lr.predict_proba(np.log(oof / (1 - oof)).reshape(-1, 1))[:, 1]
    print(f"calibrated OOF logloss {log_loss(y, pc):.4f}")

    idx["oof"] = oof
    idx.to_csv(os.path.join(args.out, "oof.csv"), index=False)
    print("\n--- OOF logloss by cluster ---")
    for c, sub in idx.groupby(cl):
        if len(sub) >= 40 and sub["is_pathologic"].nunique() == 2:
            print(f"  n={len(sub):4d}  spacing {c:>8s}  "
                  f"logloss {log_loss(sub['is_pathologic'], sub['oof'], labels=[0,1]):.4f}")


if __name__ == "__main__":
    main()