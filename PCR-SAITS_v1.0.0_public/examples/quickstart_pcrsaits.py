"""Minimal PCR-SAITS public-API example.

The SAITS backbone must be trained first. This example focuses on the Phase-5
PCR public layer rather than dataset-specific preprocessing.
"""
import numpy as np

from pcrsaits import PCRSAITS, SAITSBackbone, build_windows

train = np.load("train.npy")  # shape: [time, features]
val = np.load("val.npy")
test_masked = np.load("test_masked.npy")

feature_names = ["PM2.5", "TEMP", "WSPM"]
feature_groups = ["pollutant", "meteorological", "meteorological"]

backbone = SAITSBackbone(
    n_steps=48,
    n_features=train.shape[1],
    epochs=100,
    batch_size=128,
    patience=10,
    d_model=64,
    d_ffn=128,
    n_heads=4,
    n_layers=2,
    dropout=0.1,
    verbose=True,
)

train_windows, _ = build_windows(train, 48, stride=48)
val_windows, _ = build_windows(val, 48, stride=48)
backbone.fit(train_windows, val_windows)

model = PCRSAITS(
    backbone=backbone,
    feature_names=feature_names,
    feature_groups=feature_groups,
    n_steps=48,
    base_impute_stride=24,
)
model.fit(train, val, seed=7)

imputed = model.impute(test_masked)
np.save("test_imputed.npy", imputed)

# PCR metadata/weights are separate from backbone weights.
backbone.save("saits_backbone.pypots")
model.save("pcrsaits.pt")
