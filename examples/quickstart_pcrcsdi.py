"""Minimal real CSDIBackbone + PCRCSDI end-to-end smoke."""

from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pcrsaits import CSDIBackbone, PCRCSDI, build_windows

SEED = 7
np.random.seed(SEED)
torch.manual_seed(SEED)

rng = np.random.default_rng(SEED)
train = rng.normal(size=(48, 2))
val = rng.normal(size=(24, 2))
test = rng.normal(size=(16, 2))

train_w, _ = build_windows(train, 4, stride=4)
val_ori_w, _ = build_windows(val, 4, stride=4)
val_masked = val.copy()
val_masked[3:5, 0] = np.nan
val_w, _ = build_windows(val_masked, 4, stride=4)

backbone = CSDIBackbone(
    n_steps=4,
    n_features=2,
    epochs=1,
    batch_size=2,
    patience=1,
    n_layers=1,
    n_heads=1,
    n_channels=8,
    d_time_embedding=8,
    d_feature_embedding=4,
    d_diffusion_embedding=8,
    n_diffusion_steps=2,
    n_sampling_times=2,
    aggregation="median",
    sampling_seed=SEED,
    verbose=False,
)
backbone.fit(train_w, val_w, val_ori_w)

masked = test.copy()
masked[4:7, 0] = np.nan
masked_w, _ = build_windows(masked, 4, stride=4)

samples = backbone.sample(masked_w)
assert samples.shape == (4, 2, 4, 2)
print("CSDI sample shape:", samples.shape)

model = PCRCSDI(
    backbone=backbone,
    feature_names=["sensor_1", "sensor_2"],
    feature_groups=["sensor", "sensor"],
    n_steps=4,
    base_impute_stride=4,
    epochs=2,
    batch_size=16,
    patience=1,
    verbose=False,
)
model.fit(
    train,
    val,
    seed=SEED,
    holdout_ratio=0.15,
    pointwise_fraction=0.5,
    block_patterns=(1, 2, 3),
    block_buffer=1,
)
out = model.impute(masked)
observed = np.isfinite(masked)
assert np.isfinite(out[~observed]).all()
assert np.array_equal(out[observed], masked[observed])
print("PCRCSDI END-TO-END SMOKE PASS")
