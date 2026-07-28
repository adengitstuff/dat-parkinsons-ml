#!/usr/bin/env python3
"""
5-fold training -- v2.

    python train2d_v2.py --cache ~/niftis/cache2d_v2 --labels ~/niftis/train_labels.csv \
                         --out ~/niftis/runs/r18_v2

Changes from v1:

  [LR]   v1 used OneCycle with max_lr = 5e-4 on a pretrained backbone, which
         produced excursions like `ep 13 val logloss 1.3112` -- worse than
         predicting 0.5 for everything. v2 uses discriminative rates:
         backbone 1e-4, head 1e-3, cosine decay, no multiplier.

  [SWA]  v1 kept the best of 20 noisy epochs, so each checkpoint was the
         luckiest snapshot and the OOF was optimistic beyond the usual amount.
         v2 averages weights over the final epochs (SWA) and uses THAT.
         Averaged weights are not selected on validation noise, so the OOF
         number becomes trustworthy -- which matters more than the score,
         because every later decision is made against this metric.

Both models are evaluated so the change can be measured rather than assumed.
"""

import argparse
import copy
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.optim.swa_utils import AveragedModel, update_bn
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
            if np.random.rand() < 0.5:
                im = im[:, ::-1].copy()
            im = im * np.float32(np.random.uniform(0.9, 1.1))
            im = _affine(im,
                         float(np.random.uniform(-8, 8)),
                         [float(np.random.uniform(-0.05, 0.05) * im.shape[1])
                          for _ in range(2)],
                         float(np.random.uniform(0.94, 1.06)))
        im = np.clip(im / CLIP, 0, 1)
        im = (im - IMNET_MEAN) / IMNET_STD
        return torch.from_numpy(np.ascontiguousarray(im)), \
            torch.tensor(self.y[i], dtype=torch.float32)


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
        if dev == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = torch.sigmoid(model(x)) + torch.sigmoid(model(torch.flip(x, [3])))
        else:
            p = torch.sigmoid(model(x)) + torch.sigmoid(model(torch.flip(x, [3])))
        out.append((p / 2).float().cpu().numpy())
    return np.clip(np.concatenate(out), 1e-6, 1 - 1e-6)


def param_groups(model, lr_backbone, lr_head):
    head, back = [], []
    for n, p in model.named_parameters():
        (head if ".fc." in n else back).append(p)
    return [{"params": back, "lr": lr_backbone}, {"params": head, "lr": lr_head}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--swa-start", type=int, default=18)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    X = np.load(os.path.join(args.cache, "images.npy"))
    idx = pd.read_csv(os.path.join(args.cache, "index.csv")).merge(
        pd.read_csv(args.labels), on="uid")
    y = idx["is_pathologic"].values.astype(np.float32)
    print(f"data {X.shape}, positives {y.mean():.3f}")

    cc = idx["cluster"].astype(str).value_counts()
    cl = idx["cluster"].astype(str).where(
        idx["cluster"].astype(str).map(cc) >= 25, "other")
    skey = idx["is_pathologic"].astype(int).astype(str) + "_" + cl
    sc = skey.value_counts()
    skey = skey.where(skey.map(sc) >= args.folds, "rare")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    oof_swa, oof_best = np.zeros(len(y)), np.zeros(len(y))
    kf = StratifiedKFold(args.folds, shuffle=True, random_state=42)

    for f, (tr, va) in enumerate(kf.split(X, skey)):
        model = Wrapped(arch=args.arch, pretrained=True).to(dev)
        opt = torch.optim.AdamW(param_groups(model, args.lr, args.lr * 10),
                                weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        lossf = nn.BCEWithLogitsLoss()
        swa = AveragedModel(model)

        dtr = DataLoader(SlabDS(X[tr], y[tr], True), batch_size=args.bs, shuffle=True,
                         num_workers=6, drop_last=True, pin_memory=True)
        dva = DataLoader(SlabDS(X[va], y[va], False), batch_size=64,
                         num_workers=4, pin_memory=True)

        best_ll, best_p, best_sd = 9e9, None, None
        for ep in range(args.epochs):
            model.train()
            for xb, yb in dtr:
                xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                if dev == "cuda":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = lossf(model(xb), yb)
                else:
                    loss = lossf(model(xb), yb)
                loss.backward()
                opt.step()
            sched.step()
            if ep >= args.swa_start:
                swa.update_parameters(model)
            p = predict(model, dva, dev)
            ll = log_loss(y[va], p, labels=[0, 1])
            if ll < best_ll:
                best_ll, best_p = ll, p
                best_sd = copy.deepcopy(model.state_dict())
            print(f"  fold {f} ep {ep:2d}  val {ll:.4f}  best {best_ll:.4f}")

        # BN statistics are stale after weight averaging -> recompute on train data
        update_bn(dtr, swa, device=dev)
        p_swa = predict(swa.module, dva, dev)
        ll_swa = log_loss(y[va], p_swa, labels=[0, 1])
        print(f"fold {f}  best-epoch {best_ll:.4f}   SWA {ll_swa:.4f}")

        oof_best[va], oof_swa[va] = best_p, p_swa
        torch.save(swa.module.state_dict(), os.path.join(args.out, f"fold{f}.pt"))
        torch.save(best_sd, os.path.join(args.out, f"best_fold{f}.pt"))

    print(f"\nOOF best-epoch {log_loss(y, oof_best):.4f}  AUC {roc_auc_score(y, oof_best):.4f}")
    print(f"OOF SWA        {log_loss(y, oof_swa):.4f}  AUC {roc_auc_score(y, oof_swa):.4f}")

    lg = np.log(oof_swa / (1 - oof_swa)).reshape(-1, 1)
    lr = LogisticRegression().fit(lg, y.astype(int))
    json.dump({"coef": float(lr.coef_[0][0]), "intercept": float(lr.intercept_[0]),
               "arch": args.arch},
              open(os.path.join(args.out, "calibration.json"), "w"))
    print(f"calibrated SWA OOF {log_loss(y, lr.predict_proba(lg)[:, 1]):.4f}")

    idx["oof"] = oof_swa
    idx.to_csv(os.path.join(args.out, "oof.csv"), index=False)
    print("\n--- OOF logloss by cluster ---")
    for c, sub in idx.groupby(cl):
        if len(sub) >= 40 and sub["is_pathologic"].nunique() == 2:
            print(f"  n={len(sub):4d}  spacing {str(c):>8s}  "
                  f"ll {log_loss(sub['is_pathologic'], sub['oof'], labels=[0,1]):.4f}  "
                  f"auc {roc_auc_score(sub['is_pathologic'], sub['oof']):.4f}")


if __name__ == "__main__":
    main()