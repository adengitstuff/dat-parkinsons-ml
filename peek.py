"""
Easy, fast look at the DaT Parkinson's Challenge data.

"""
import glob
import os
import nibabel as nib
import pandas as pd

DATA_DIR = os.path.expanduser("~/niftis")

labels_path = os.path.join(DATA_DIR, "train_labels.csv")
niftis_dir = os.path.join(DATA_DIR, "niftis_extracted")


labels = pd.read_csv(labels_path)
print(f"=== {labels_path} ===")
print(labels.shape)
print(labels.head(10))
print(labels["is_pathologic"].value_counts())
print()

files = sorted(glob.glob(os.path.join(niftis_dir, "*.nii.gz")))
print(f"=== found {len(files)} nifti files ===\n")

rows = []
for f in files[:20]:  # justt randomly grabbing 20
    img = nib.load(f)
    shape = img.shape
    zooms = img.header.get_zooms()
    data = img.get_fdata()
    uid = os.path.basename(f).replace(".nii.gz", "")
    label = labels.loc[labels["uid"] == uid, "is_pathologic"]
    rows.append({
        "uid": uid,
        "shape": shape,
        "spacing_mm": zooms,
        "physical_mm": tuple(round(s * z, 1) for s, z in zip(shape, zooms)),
        "dtype": img.get_data_dtype(),
        "min": data.min(),
        "max": data.max(),
        "mean": round(data.mean(), 1),
        "label": label.values[0] if len(label) else "MISSING",
    })

df = pd.DataFrame(rows)
print(df.to_string())