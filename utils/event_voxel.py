"""Signed event voxels with temporal interpolation and percentile normalization."""

import numpy as np

NUM_BINS = 10


def load_event_npz(npz_path):
    """Load timestamp, x, y, and polarity columns as an [N, 4] float64 array."""
    d = np.load(npz_path)
    ts = d["timestamp"].astype(np.float64)
    x = d["x"].astype(np.float64)
    y = d["y"].astype(np.float64)
    pol = d["polarity"].astype(np.float64)
    return np.stack([ts, x, y, pol], axis=1)


def events_to_voxel_grid(events, num_bins, width, height):
    """Bin signed events with linear interpolation along the time axis."""
    assert events.shape[1] == 4
    assert num_bins > 0
    assert width > 0
    assert height > 0

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    if len(events) < 5:
        return np.reshape(voxel_grid, (num_bins, height, width))

    events = events[np.argsort(events[:, 0])]

    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp
    if deltaT == 0:
        deltaT = 1.0

    events = events.copy()
    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT

    ts = events[:, 0]
    xs = events[:, 1].astype(np.int64)
    ys = events[:, 2].astype(np.int64)
    pols = events[:, 3].copy()
    pols[pols == 0] = -1

    tis = ts.astype(np.int64)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    return np.reshape(voxel_grid, (num_bins, height, width))


def voxel_norm(voxel):
    """Normalize positive and negative voxel values using their 2nd and 98th percentiles."""
    voxel = voxel.copy()
    voxel_pos = voxel[voxel > 0]
    voxel_neg = voxel[voxel < 0]

    if len(voxel_pos) == 0 or len(voxel_neg) == 0:
        return voxel

    pos_2 = np.percentile(voxel_pos, 2)
    pos_98 = np.percentile(voxel_pos, 98)
    neg_2 = np.percentile(voxel_neg, 2)
    neg_98 = np.percentile(voxel_neg, 98)

    voxel[voxel > 0] = np.clip(voxel[voxel > 0], pos_2, pos_98)
    voxel[voxel < 0] = np.clip(voxel[voxel < 0], neg_2, neg_98)

    if pos_98 == pos_2:
        pos_98 = np.max(voxel_pos)
        pos_2 = np.min(voxel_pos)
    if neg_98 == neg_2:
        neg_98 = np.max(voxel_neg)
        neg_2 = np.min(voxel_neg)

    if pos_98 == pos_2:
        voxel[voxel > 0] = np.where(voxel[voxel > 0] > 0, 1, 0)
    else:
        voxel[voxel > 0] = voxel[voxel > 0] / pos_98

    if neg_98 == neg_2:
        voxel[voxel < 0] = np.where(voxel[voxel < 0] < 0, -1, 0)
    else:
        voxel[voxel < 0] = voxel[voxel < 0] / abs(neg_2)

    return voxel


def build_voxel(npz_path, width, height, num_bins=NUM_BINS):
    """Build a normalized [bins, height, width] signed event voxel."""
    events = load_event_npz(npz_path)
    voxel = events_to_voxel_grid(events, num_bins, width, height)
    return voxel_norm(voxel).astype(np.float32)
