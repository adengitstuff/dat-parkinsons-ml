"""
Dataset + DataLoader for the DaT Parkinson's Challenge. The two things I laerned
were the variability (<300 and <65k) in normalization, so I'll need per-scan normalization.
Then, to be robust I really need to handle multiple size of inputs.

I think I'm going to take the approach of 'roughly scale each input up to match 
its 'real, brain' scan'. Then, the network will effectively be training on different brain stuff.

From simple, early research online, I see the immense value of asymmetry analysis - the eventual real,
actual network that I build will need that. Then, the other part that's really, really cool is NATIVE to me :)
It's really something that I'm familiar with: An ROI-based network that can effectively 'zoom in' or analyze
a region deeply might relaly, really nail the probability scores...

Regardless: this pipeline is to just get started, and see an initial loss, so that I have real comparisions and real
numbers to measure if I'm going up or down!

Pipeline per sample:
  1. load .nii.gz (nibabel)
  2. resample to a common physical voxel spacing (so "one voxel" means
     the same real-world mm across every scan, regardless of original scanner)
  3. crop (if bigger than target) or pad (if smaller) to one fixed shape,
     centered on the volume's own intensity centroid rather than its
     geometric center - keeps the bright region in-frame even if the
     patient wasn't perfectly centered in the scanner
  4. per-scan percentile normalization - each volume is scaled using its
     OWN statistics, so it doesn't matter whether this particular file
     originally lived in the "raw count" regime or the "already rescaled
     to ~255" regime we found in the audit - both end up in the same
     range after this step
  5. cast to float32

Run this file directly to sanity-check it against real data:
    python dataset.py
"""
import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom, center_of_mass

# ---- config ----
DATA_DIR = os.path.expanduser("~/niftis")
NIFTI_DIR = os.path.join(DATA_DIR, "niftis_extracted")
LABELS_PATH = os.path.join(DATA_DIR, "train_labels.csv")




# I'm choosing a target spacing of (2.0, 2.0, 2.0) to start! Really I think I need to research
# the right or optimal thing. I think there's MedicalNet stuff too which has pre-trained checkpoints
# that I can use, but I think their normalization is like, batchnorm
TARGET_SPACING = (2.0, 2.0, 2.0)   # mm

TARGET_SHAPE = (96, 96, 96)         # voxels - every volume ends up exactly this shape
NORMALIZE_PERCENTILE = 99.5         # per-scan clip/scale percentile


# This should solve the shortuct-learning risk. This is the direct fix for the projection - one voxel always means 2mm of real space, after this
def resample_to_spacing(data: np.ndarray, orig_spacing, target_spacing) -> np.ndarray:
    """Resize so each voxel represents target_spacing mm, regardless of orig_spacing."""
    zoom_factors = tuple(o / t for o, t in zip(orig_spacing, target_spacing))
    return zoom(data, zoom_factors, order=1)  # order=1 = trilinear-ish


def crop_or_pad_centered(data: np.ndarray, target_shape) -> np.ndarray:
    """
    Make data exactly target_shape!
    - dims bigger than target: crop, centered on the volume's own intensity
      centroid (falls back to geometric center if the volume is ~all zero)
    - dims smaller than target: zero-pad, centered
    """
    total = data.sum()
    if total > 0:
        centroid = center_of_mass(data)
    else:
        centroid = tuple(s / 2 for s in data.shape)

    out = np.zeros(target_shape, dtype=data.dtype)

    src_starts, src_ends = [], []
    dst_starts, dst_ends = [], []

    for dim, (src_size, tgt_size, c) in enumerate(zip(data.shape, target_shape, centroid)):
        if src_size >= tgt_size:
            # crop!  centroid, clamped in-bounds
            start = int(round(c - tgt_size / 2))
            start = max(0, min(start, src_size - tgt_size))
            end = start + tgt_size
            src_starts.append(start)
            src_ends.append(end)
            dst_starts.append(0)
            dst_ends.append(tgt_size)
        else:
            # place the smaller source centered in the target
            src_starts.append(0)
            src_ends.append(src_size)
            offset = (tgt_size - src_size) // 2
            dst_starts.append(offset)
            dst_ends.append(offset + src_size)

    src_slices = tuple(slice(s, e) for s, e in zip(src_starts, src_ends))
    dst_slices = tuple(slice(s, e) for s, e in zip(dst_starts, dst_ends))
    out[dst_slices] = data[src_slices]
    return out

# The big technical gotcha from the early peeks into the data that I did:
def normalize_percentile(data: np.ndarray, percentile: float = NORMALIZE_PERCENTILE) -> np.ndarray:
    """Per-scan normalization: clip to this scan's own high percentile, scale to ~[0,1].
    Uses only nonzero voxels for the percentile so background doesn't skew it."""
    nonzero = data[data > 0]
    if nonzero.size == 0:
        return data  # degenerate empty volume, nothing to normalize
    hi = np.percentile(nonzero, percentile)
    if hi <= 0:
        return data
    out = np.clip(data, 0, hi) / hi
    return out


def preprocess_volume(nifti_path: str) -> np.ndarray:
    """Full preprocessing pipeline for one file. Returns a (D,H,W) float32 array."""
    img = nib.load(nifti_path)
    data = img.get_fdata().astype(np.float32)
    spacing = img.header.get_zooms()[:3]

    data = resample_to_spacing(data, spacing, TARGET_SPACING)
    data = crop_or_pad_centered(data, TARGET_SHAPE)
    data = normalize_percentile(data)

    return data.astype(np.float32)


class DatScanDataset(Dataset):
    """
    labels_df: DataFrame with columns ['uid', 'is_pathologic'] (label column
               optional - if absent, __getitem__ just won't return a label,
               useful for inference-only use later).
    nifti_dir: folder containing '{uid}.nii.gz' files.
    cache_dir: if given, preprocessed tensors are cached to disk as .pt files
               keyed by uid, so repeat epochs don't redo the resample/crop work.
    """

    def __init__(self, labels_df: pd.DataFrame, nifti_dir: str = NIFTI_DIR, cache_dir: str | None = None):
        self.df = labels_df.reset_index(drop=True)
        self.nifti_dir = nifti_dir
        self.cache_dir = cache_dir
        self.has_labels = "is_pathologic" in self.df.columns

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        # fail fast, at construction time, rather than mysteriously mid-training
        missing = [
            uid for uid in self.df["uid"]
            if not os.path.exists(os.path.join(self.nifti_dir, f"{uid}.nii.gz"))
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} uids in labels_df have no matching .nii.gz file, "
                f"e.g. {missing[:5]}"
            )

    def __len__(self):
        return len(self.df)

    def _load_volume(self, uid: str) -> np.ndarray:
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"{uid}.npy")
            if os.path.exists(cache_path):
                return np.load(cache_path)
            data = preprocess_volume(os.path.join(self.nifti_dir, f"{uid}.nii.gz"))
            np.save(cache_path, data)
            return data
        return preprocess_volume(os.path.join(self.nifti_dir, f"{uid}.nii.gz"))

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        uid = row["uid"]

        data = self._load_volume(uid)


        # just including these even in the real dataloader in case it's useful. can't hurt:

        print(f" ~~~ Sanity checking:")
        assert data.shape == TARGET_SHAPE, f"{uid}: bad shape {data.shape}"
        assert np.isfinite(data).all(), f"{uid}: NaN or Inf in volume"

        tensor = torch.from_numpy(data).unsqueeze(0)  # (1, D, H, W) - the channel dim

        if self.has_labels:
            label = float(row["is_pathologic"])
            assert label in (0.0, 1.0), f"{uid}: unexpected label value {label}"
            return tensor, torch.tensor(label, dtype=torch.float32), uid
        else:
            return tensor, uid


def get_dataloader(labels_df, nifti_dir=NIFTI_DIR, cache_dir=None,
                    batch_size=4, shuffle=True, num_workers=4):
    dataset = DatScanDataset(labels_df, nifti_dir=nifti_dir, cache_dir=cache_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                       num_workers=num_workers, pin_memory=True)


if __name__ == "__main__":
    labels = pd.read_csv(LABELS_PATH)
    print(f"labels: {labels.shape}")

    # small subset for a fast sanity check, not the full 1362
    sample_df = labels.sample(n=12, random_state=0)

    ds = DatScanDataset(sample_df, cache_dir=os.path.expanduser("~/niftis/cache"))
    print(f"dataset length: {len(ds)}")

    tensor, label, uid = ds[0]
    print(f"single sample -> uid={uid}, shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
          f"label={label.item()}, min={tensor.min().item():.3f}, max={tensor.max().item():.3f}")

    # check real batch through the DataLoader
    loader = get_dataloader(sample_df, cache_dir=os.path.expanduser("~/niftis/cache"),
                             batch_size=4, shuffle=True, num_workers=0)
    for batch_tensors, batch_labels, batch_uids in loader:
        print(f"batch -> tensor shape={tuple(batch_tensors.shape)}, "
              f"labels={batch_labels.tolist()}, uids={batch_uids}")
        break  # :)

    print("~~~~~~~~~~~~~~~~~~ \nAll sanity checks passed! ~~")
    