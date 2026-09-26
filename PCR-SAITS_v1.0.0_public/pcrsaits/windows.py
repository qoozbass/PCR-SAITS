from __future__ import annotations

import numpy as np


def build_windows(values: np.ndarray, window: int, stride: int):
    """Exact extraction of the legacy v7.1 window builder."""
    n = len(values)
    if n < window:
        pad = np.full((window - n, values.shape[1]), np.nan)
        values = np.concatenate([values, pad], axis=0)
        n = len(values)

    starts = list(range(0, max(1, n - window + 1), stride))
    if starts[-1] != n - window:
        starts.append(n - window)

    windows = np.stack([values[s : s + window] for s in starts], axis=0)
    return windows, np.asarray(starts, dtype=int)


def reconstruct_from_windows(
    windows: np.ndarray,
    starts: np.ndarray,
    total_length: int,
):
    """Exact extraction of legacy overlap averaging."""
    _, win, n_feat = windows.shape
    acc = np.zeros((total_length, n_feat), dtype=float)
    cnt = np.zeros((total_length, n_feat), dtype=float)

    for i, s in enumerate(starts):
        e = min(total_length, s + win)
        part = windows[i][: e - s]
        valid = np.isfinite(part)
        acc[s:e][valid] += part[valid]
        cnt[s:e][valid] += 1.0

    out = np.full((total_length, n_feat), np.nan)
    valid = cnt > 0
    out[valid] = acc[valid] / cnt[valid]
    return out
