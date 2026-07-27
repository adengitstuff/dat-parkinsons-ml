"""Submission entry point for the DaT Parkinson's Challenge.
print N rows thing removed!
"""
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import torch

from model import SimpleCNN3D
from preprocess import resample_to_spacing, crop_or_pad_centered, normalize_percentile

# Defaults to the real container path. Override with the DATA_DIR env var for local testing
# (e.g. `DATA_DIR=~/niftis/smoke_test_extracted/smoke_test_data python main.py`)
DATA_ROOT = Path(os.environ.get("DATA_DIR", "/code_execution/data")).expanduser()
NIFTI_DIR = DATA_ROOT / "niftis"
SUBMISSION_FORMAT_PATH = DATA_ROOT / "submission_format.csv"
WRITE_SUBMISSION_PATH = Path("submission.csv")

SRC_ROOT = Path(__file__).parent.resolve()
WEIGHTS_PATH = SRC_ROOT / "model_weights.pt"
CALIBRATION_PATH = SRC_ROOT / "calibration.json"

TARGET_SPACING = (2.0, 2.0, 2.0)
TARGET_SHAPE = (96, 96, 96)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    model = SimpleCNN3D()
    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def load_calibration():
    with open(CALIBRATION_PATH) as f:
        cal = json.load(f)
    return cal["coef"], cal["intercept"]


def preprocess_volume(nifti_path):
    img = nib.load(nifti_path)
    data = img.get_fdata().astype(np.float32)
    spacing = img.header.get_zooms()[:3]
    data = resample_to_spacing(data, spacing, TARGET_SPACING)
    data = crop_or_pad_centered(data, TARGET_SHAPE)
    data = normalize_percentile(data)
    return data.astype(np.float32)


def predict_one(model, coef, intercept, nifti_path):
    data = preprocess_volume(nifti_path)
    tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,D,H,W)
    with torch.no_grad():
        logit = model(tensor).item()
    prob = 1.0 / (1.0 + np.exp(-(coef * logit + intercept)))
    return float(prob)


def main():
    submission_format = pd.read_csv(SUBMISSION_FORMAT_PATH)
    print("Loaded submission_format.csv.")

    model = load_model()
    coef, intercept = load_calibration()
    print("Model and calibration loaded.")

    submission_format = submission_format.set_index("uid")

    for uid in submission_format.index:
        img_path = NIFTI_DIR / f"{uid}.nii.gz"
        prob = predict_one(model, coef, intercept, img_path)
        submission_format.loc[uid, "is_pathologic"] = prob

    submission_format = submission_format.reset_index()
    submission_format.to_csv(WRITE_SUBMISSION_PATH, index=False)
    print("Wrote predictions to submission.csv")


if __name__ == "__main__":
    main()