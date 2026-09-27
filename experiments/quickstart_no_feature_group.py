"""
Research-only quickstart: PCR with and without feature-group tags.

This is NOT part of the stable PCR-SAITS v1.1.0 public API.

It demonstrates a controlled no-feature-group ablation using SAITS:
    SAITS backbone
    PCR-SAITS with explicit feature groups
    PCR-SAITS NoGroup (same PCR design, domain-tag inputs disabled)

Run from the repository root:

    python experiments/quickstart_no_feature_group.py

Requirements:
    pip install pcrsaits==1.1.0 pandas

The NoGroup helper is imported from:
    experiments/no_feature_group_ablation.py
"""

from __future__ import annotations

import random

import numpy as np
import torch

from pcrsaits import SAITSBackbone, PCRSAITS, build_windows
from no_feature_group_ablation import NoFeatureGroupPCR


SEED = 7
N_STEPS = 8
STRIDE = 8

FEATURE_NAMES = ["sensor_value", "temperature"]
FEATURE_GROUPS = ["sensor", "meteorological"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_series(n: int) -> np.ndarray:
    """Small structured two-feature synthetic series."""
    t = np.arange(n, dtype=np.float64)
    x1 = np.sin(2.0 * np.pi * t / 24.0) + 0.02 * t
    x2 = 0.6 * np.cos(2.0 * np.pi * t / 16.0) + 0.3 * x1
    return np.column_stack([x1, x2])


def standardize_from_train(train, val, test):
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-12] = 1.0
    return (
        (train - mean) / std,
        (val - mean) / std,
        (test - mean) / std,
    )


def rmse(truth: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    err = truth[mask] - pred[mask]
    return float(np.sqrt(np.mean(err ** 2)))


def mae(truth: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    err = truth[mask] - pred[mask]
    return float(np.mean(np.abs(err)))


def main() -> None:
    set_seed(SEED)

    full = make_series(192)
    train = full[:96]
    val = full[96:144]
    test = full[144:192]
    train, val, test = standardize_from_train(train, val, test)

    # Train one SAITS backbone. The SAME trained object is reused by both
    # grouped and no-group PCR variants.
    train_w, _ = build_windows(train, N_STEPS, stride=STRIDE)
    val_w, _ = build_windows(val, N_STEPS, stride=STRIDE)

    backbone = SAITSBackbone(
        n_steps=N_STEPS,
        n_features=train.shape[1],
        epochs=2,
        batch_size=8,
        patience=2,
        d_model=16,
        d_ffn=32,
        n_heads=2,
        n_layers=1,
        dropout=0.1,
        verbose=False,
    )
    backbone.fit(train_w, val_w, val_w)

    # Normal public PCR-SAITS with explicit feature-group metadata.
    grouped = PCRSAITS(
        backbone=backbone,
        feature_names=FEATURE_NAMES,
        feature_groups=FEATURE_GROUPS,
        metadata_mode="explicit",
        n_steps=N_STEPS,
        base_impute_stride=STRIDE,
        epochs=3,
        batch_size=32,
        patience=2,
        verbose=False,
    )
    grouped.fit(
        train,
        val,
        seed=SEED,
        holdout_ratio=0.15,
        pointwise_fraction=0.5,
        block_patterns=(1, 2, 4),
        block_buffer=1,
    )

    # Research-only NoGroup PCR.
    # Do NOT replace this with feature_groups=None in the public API.
    nogroup = NoFeatureGroupPCR(
        backbone=backbone,
        feature_names=FEATURE_NAMES,
        n_steps=N_STEPS,
        base_impute_stride=STRIDE,
        epochs=3,
        batch_size=32,
        patience=2,
        verbose=False,
    )
    nogroup.fit(
        train,
        val,
        seed=SEED,
        holdout_ratio=0.15,
        pointwise_fraction=0.5,
        block_patterns=(1, 2, 4),
        block_buffer=1,
    )

    # Artificially hide known test cells.
    mask = np.zeros_like(test, dtype=bool)
    mask[8:16, 0] = True
    mask[24:32, 1] = True

    masked = test.copy()
    masked[mask] = np.nan

    pred_group = grouped.impute(masked)
    pred_nogroup = nogroup.impute(masked)

    # Sanity: originally observed cells must remain unchanged.
    observed = ~mask
    assert np.allclose(pred_group[observed], masked[observed], rtol=0, atol=0)
    assert np.allclose(pred_nogroup[observed], masked[observed], rtol=0, atol=0)

    group_rmse = rmse(test, pred_group, mask)
    nogroup_rmse = rmse(test, pred_nogroup, mask)
    group_mae = mae(test, pred_group, mask)
    nogroup_mae = mae(test, pred_nogroup, mask)

    print("=" * 72)
    print("PCR-SAITS No-Feature-Group Research Quickstart")
    print("=" * 72)
    print(f"PCR-Group   RMSE: {group_rmse:.6f}  MAE: {group_mae:.6f}")
    print(f"PCR-NoGroup RMSE: {nogroup_rmse:.6f}  MAE: {nogroup_mae:.6f}")

    if group_rmse < nogroup_rmse:
        print("RMSE result: Group performed better in this tiny demonstration.")
    elif nogroup_rmse < group_rmse:
        print("RMSE result: NoGroup performed better in this tiny demonstration.")
    else:
        print("RMSE result: tie.")

    print()
    print("This is a functional example only, not a benchmark or scientific claim.")


if __name__ == "__main__":
    main()
