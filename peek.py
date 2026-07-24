"""
Quick look at the DaT Parkinson's Challenge data.

The data in the first peek had crazy values - shape and physical size scaling to 63cm, and the max
ranges are completely, truly off (7114 in 1 file vs 59760). Tryign to peek and make sure the data is sane
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
for f in files:  # full dataset now — 1362 files
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

# group into a rough "acquisition cluster" by shape + spacing signature
df["cluster"] = df["shape"].astype(str) + " @ " + df["spacing_mm"].astype(str)

print("\n=== cluster sizes ===")
print(df["cluster"].value_counts())

print("\n=== label rate by cluster (THE important check) ===")
print(df.groupby("cluster")["label"].agg(["count", "mean"]).sort_values("count", ascending=False))

print("\n=== intensity max distribution (flags the 8-bit-vs-raw-count split) ===")
print(df["max"].describe())
print("\nfiles maxing out suspiciously low (<300, likely rescaled to 0-255):")
print((df["max"] < 300).sum(), "out of", len(df))

df.to_csv("audit.csv", index=False)
print("\nsaved full audit to audit.csv")