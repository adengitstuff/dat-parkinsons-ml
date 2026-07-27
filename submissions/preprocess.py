""" preprocessing, self-contained! For the runtime :)"""
import numpy as np
from scipy.ndimage import zoom, center_of_mass


def resample_to_spacing(data, orig_spacing, target_spacing):
    zoom_factors = tuple(o / t for o, t in zip(orig_spacing, target_spacing))
    return zoom(data, zoom_factors, order=1)


def crop_or_pad_centered(data, target_shape):
    total = data.sum()
    centroid = center_of_mass(data) if total > 0 else tuple(s / 2 for s in data.shape)

    out = np.zeros(target_shape, dtype=data.dtype)
    src_starts, src_ends, dst_starts, dst_ends = [], [], [], []

    for src_size, tgt_size, c in zip(data.shape, target_shape, centroid):
        if src_size >= tgt_size:
            start = int(round(c - tgt_size / 2))
            start = max(0, min(start, src_size - tgt_size))
            src_starts.append(start)
            src_ends.append(start + tgt_size)
            dst_starts.append(0)
            dst_ends.append(tgt_size)
        else:
            src_starts.append(0)
            src_ends.append(src_size)
            offset = (tgt_size - src_size) // 2
            dst_starts.append(offset)
            dst_ends.append(offset + src_size)

    src_slices = tuple(slice(s, e) for s, e in zip(src_starts, src_ends))
    dst_slices = tuple(slice(s, e) for s, e in zip(dst_starts, dst_ends))
    out[dst_slices] = data[src_slices]
    return out


def normalize_percentile(data, percentile=99.5):
    nonzero = data[data > 0]
    if nonzero.size == 0:
        return data
    hi = np.percentile(nonzero, percentile)
    if hi <= 0:
        return data
    return np.clip(data, 0, hi) / hi