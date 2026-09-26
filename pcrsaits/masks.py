from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def apply_mask(values: np.ndarray, artificial_mask: np.ndarray):
    out = values.copy()
    out[artificial_mask] = np.nan
    return out


def build_correction_scope(
    masked_values: np.ndarray,
    evaluation_mask: Optional[np.ndarray] = None,
    evaluation_gap_len: Optional[np.ndarray] = None,
    gap_cap: int = 48,
):
    """Build deployment correction scope independently from evaluation scope.

    This is the reviewer-final R4.1 behavior now present in the latest v7.1
    source: every position missing in the supplied input is correctable.
    """
    masked_values = np.asarray(masked_values)
    correction_mask = ~np.isfinite(masked_values)
    n_t, n_f = correction_mask.shape

    correction_gap_len = np.ones((n_t, n_f), dtype=float)
    for f in range(n_f):
        t = 0
        while t < n_t:
            if not correction_mask[t, f]:
                t += 1
                continue
            s = t
            while t < n_t and correction_mask[t, f]:
                t += 1
            run_len = float(min(t - s, int(gap_cap)))
            correction_gap_len[s:t, f] = run_len

    if evaluation_mask is not None:
        evaluation_mask = np.asarray(evaluation_mask, dtype=bool)
        if evaluation_mask.shape != correction_mask.shape:
            raise ValueError("evaluation_mask shape must match masked_values")
        if np.any(evaluation_mask & ~correction_mask):
            raise ValueError(
                "evaluation targets must be missing in masked_values"
            )
        if evaluation_gap_len is not None:
            evaluation_gap_len = np.asarray(evaluation_gap_len, dtype=float)
            if evaluation_gap_len.shape != correction_mask.shape:
                raise ValueError(
                    "evaluation_gap_len shape must match masked_values"
                )
            correction_gap_len[evaluation_mask] = (
                evaluation_gap_len[evaluation_mask]
            )

    return correction_mask, correction_gap_len


def make_holdout_train_mask(
    values: np.ndarray,
    seed: int,
    holdout_ratio: float = 0.15,
    pointwise_fraction: float = 0.5,
    block_patterns: Tuple[int, ...] = (6, 12, 24, 48),
    block_buffer: int = 1,
):
    """Exact extraction of the legacy v7.1 PCR holdout generator."""
    rng = np.random.default_rng(seed)
    observed = np.isfinite(values)
    T, F = values.shape
    mask = np.zeros_like(observed, dtype=bool)
    gap_len = np.zeros_like(values, dtype=float)

    total_observed = int(observed.sum())
    target_missing = max(1, int(round(total_observed * holdout_ratio)))
    point_target = int(round(target_missing * pointwise_fraction))
    block_target = max(0, target_missing - point_target)

    # pointwise portion
    if point_target > 0:
        candidates = np.argwhere(observed)
        if len(candidates) > 0:
            chosen_idx = rng.choice(
                len(candidates),
                size=min(point_target, len(candidates)),
                replace=False,
            )
            chosen = candidates[chosen_idx]
            mask[chosen[:, 0], chosen[:, 1]] = True
            gap_len[chosen[:, 0], chosen[:, 1]] = 1.0

    # block portion
    if block_target > 0:
        occupied = mask.copy()
        masked_cells = int(mask.sum())
        candidates: List[Tuple[float, int, int, int]] = []
        for feat in range(F):
            valid = observed[:, feat]
            for block_len in block_patterns:
                if valid.sum() < block_len:
                    continue
                for start in range(0, T - block_len + 1):
                    seg = slice(start, start + block_len)
                    if not valid[seg].all():
                        continue
                    score = float(rng.random())
                    candidates.append((score, start, feat, block_len))

        candidates.sort(key=lambda x: x[0], reverse=True)
        for _, start, feat, block_len in candidates:
            if masked_cells >= target_missing:
                break
            seg = slice(start, start + block_len)
            left = max(0, start - block_buffer)
            right = min(T, start + block_len + block_buffer)
            if occupied[left:right, feat].any():
                continue
            occupied[left:right, feat] = True
            add = ~mask[seg, feat]
            if not np.any(add):
                continue
            mask[seg, feat] = True
            gap_len[seg, feat] = float(block_len)
            masked_cells = int(mask.sum())

    # fallback fill if blocks could not reach target
    remaining = max(0, target_missing - int(mask.sum()))
    if remaining > 0:
        candidates = np.argwhere(observed & ~mask)
        if len(candidates) > 0:
            chosen_idx = rng.choice(
                len(candidates),
                size=min(remaining, len(candidates)),
                replace=False,
            )
            chosen = candidates[chosen_idx]
            mask[chosen[:, 0], chosen[:, 1]] = True
            current = gap_len[chosen[:, 0], chosen[:, 1]]
            gap_len[chosen[:, 0], chosen[:, 1]] = np.where(
                current > 0,
                current,
                1.0,
            )

    return mask, gap_len
