"""
Generate calibration.json from the EXISTING checkpoint - no retraining.
Rebuilds the same train/val split (same random_state) so the val set matches
what the model was actually validated on, then does one forward-only pass.
"""
import os
import json
import pandas as pd
import torch
import torch.nn as nn

from dataset import LABELS_PATH, get_dataloader
from model import SimpleCNN3D
from train import build_stratified_split, run_epoch, calibrate, CACHE_DIR, CHECKPOINT_PATH, DEVICE, BATCH_SIZE

CALIBRATION_PATH = os.path.expanduser("~/niftis/calibration.json")


def main():
    labels = pd.read_csv(LABELS_PATH)
    _, val_df = build_stratified_split(labels)  # same random_state=42 as training used
    val_loader = get_dataloader(val_df, cache_dir=CACHE_DIR, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=4)

    model = SimpleCNN3D().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    _, val_logloss, val_logits, val_labels = run_epoch(model, val_loader, criterion)
    print(f"reloaded checkpoint - val log loss (uncalibrated): {val_logloss:.4f}")

    calibrator, calibrated_logloss = calibrate(val_logits, val_labels)
    print(f"val log loss after calibration: {calibrated_logloss:.4f}")

    coef = float(calibrator.coef_[0][0])
    intercept = float(calibrator.intercept_[0])
    with open(CALIBRATION_PATH, "w") as f:
        json.dump({"coef": coef, "intercept": intercept}, f)
    print(f"saved calibration to {CALIBRATION_PATH}: coef={coef:.4f}, intercept={intercept:.4f}")


if __name__ == "__main__":
    main()