"""
This it the training run for the simple, very basic baseline.

In the really near-future, I'll test medicalNet checkpoints, MonAI extra stuff, etc.
This shoudl train just the simple basline, but also apply Platt scaling stuff for calibration.

This is the early, initial toes-in-the-water, meant to just give me a general, real feel and realistic
numbers to test out!

I tried to match trochvision, torch stuff to the runtime - let's see if this runs, lol!

This runs off the clusters in the audit.csv thing that I found
Usage: python train.py
"""
import os
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression

from dataset import DatScanDataset, DATA_DIR, NIFTI_DIR, LABELS_PATH, get_dataloader
from model import SimpleCNN3D

CACHE_DIR = os.path.expanduser("~/niftis/cache")
CHECKPOINT_PATH = os.path.expanduser("~/niftis/best_model.pt")
AUDIT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 30
LR = 2e-4


def build_stratified_split(labels_df, test_size=0.2, seed=42):
    """
    Stratify by label AND acquisition cluster where possible! This is the first try
    for cross-scanner generalization stuff. test_size = 0.2
    """
 
    df = labels_df.copy()
 
    if os.path.exists(AUDIT_PATH):
        audit = pd.read_csv(AUDIT_PATH)[["uid", "cluster"]]
        df = df.merge(audit, on="uid", how="left")
        df["cluster"] = df["cluster"].fillna("unknown")
        # rare clusters (n<10) get bucketed together so stratify doesn't
        # choke on classes with too few members to split
        counts = df["cluster"].value_counts()
        rare = counts[counts < 10].index
        df["cluster_bucket"] = df["cluster"].where(~df["cluster"].isin(rare), "rare")
        strat_key = df["is_pathologic"].astype(str) + "_" + df["cluster_bucket"]
    else:
        print("audit.csv not found - falling back to label-only stratification")
        strat_key = df["is_pathologic"].astype(str)
 
    train_df, val_df = train_test_split(
        df, test_size=test_size, stratify=strat_key, random_state=seed
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def run_epoch(model, loader, criterion, optimizer=None):
    """
    runs the epoch!
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
 
    all_logits, all_labels = [], []
    total_loss = 0.0
 
    with torch.set_grad_enabled(is_train):
        for tensors, labels, uids in loader:
            tensors, labels = tensors.to(DEVICE), labels.to(DEVICE)
 
            logits = model(tensors)
            loss = criterion(logits, labels)
 
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
 
            total_loss += loss.item() * len(labels)
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
 
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    probs = 1 / (1 + np.exp(-all_logits))  # sigmoid, for reporting real log loss
    real_logloss = log_loss(all_labels, probs, labels=[0, 1])
 
    return total_loss / len(all_labels), real_logloss, all_logits, all_labels



def calibrate(val_logits, val_labels):
    """the first try for calibration/pratt scaling stuff: fit a 1D logistic regression on raw logits -> calibrated probability."""
    calibrator = LogisticRegression()
    calibrator.fit(val_logits.reshape(-1, 1), val_labels)
    calibrated_probs = calibrator.predict_proba(val_logits.reshape(-1, 1))[:, 1]
    calibrated_logloss = log_loss(val_labels, calibrated_probs, labels=[0, 1])
    return calibrator, calibrated_logloss



def main():
    print(f"device: {DEVICE}")
 
    labels = pd.read_csv(LABELS_PATH)
    train_df, val_df = build_stratified_split(labels)
    print(f"train: {len(train_df)}  val: {len(val_df)}")
 
    train_loader = get_dataloader(train_df, cache_dir=CACHE_DIR, batch_size=BATCH_SIZE,
                                   shuffle=True, num_workers=4)
    val_loader = get_dataloader(val_df, cache_dir=CACHE_DIR, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=4)
 
    model = SimpleCNN3D().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}")
    optimizer = AdamW(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
 
    best_val_logloss = float("inf")
 
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_logloss, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_logloss, val_logits, val_labels = run_epoch(model, val_loader, criterion)
        elapsed = time.time() - t0
 
        flag = ""
        if val_logloss < best_val_logloss:
            best_val_logloss = val_logloss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            flag = "  <- saved best"
 
        print(f"epoch {epoch:3d}  train_logloss={train_logloss:.4f}  "
              f"val_logloss={val_logloss:.4f}  ({elapsed:.1f}s){flag}")
 
    print(f"\nbest val log loss (uncalibrated): {best_val_logloss:.4f}")
 
    # calibrate, even in v1 :) why not!
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    _, _, val_logits, val_labels = run_epoch(model, val_loader, criterion)
    calibrator, calibrated_logloss = calibrate(val_logits, val_labels)
    print(f"val log loss after calibration:    {calibrated_logloss:.4f}")
 
    baseline_p = labels["is_pathologic"].mean()
    baseline_logloss = -(baseline_p * np.log(baseline_p) + (1 - baseline_p) * np.log(1 - baseline_p))
    print(f"\n(reference: always-predict-base-rate log loss = {baseline_logloss:.4f})")
 
 
if __name__ == "__main__":
    main()