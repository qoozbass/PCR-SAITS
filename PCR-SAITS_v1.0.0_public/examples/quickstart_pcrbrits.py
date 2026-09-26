"""Minimal PCR-BRITS public-API example."""
import numpy as np

from pcrsaits import BRITSBackbone, PCRBRITS, build_windows

train = np.load("train.npy")
val = np.load("val.npy")
test_masked = np.load("test_masked.npy")

feature_names = ["PM2.5", "TEMP", "WSPM"]
feature_groups = ["pollutant", "meteorological", "meteorological"]

backbone = BRITSBackbone(
    n_steps=48,
    n_features=train.shape[1],
    epochs=100,
    batch_size=128,
    patience=10,
    rnn_hidden_size=64,
    verbose=True,
)

train_windows, _ = build_windows(train, 48, stride=48)
val_windows, _ = build_windows(val, 48, stride=48)
backbone.fit(train_windows, val_windows, val_windows)

model = PCRBRITS(
    backbone=backbone,
    feature_names=feature_names,
    feature_groups=feature_groups,
    n_steps=48,
    base_impute_stride=24,
)
model.fit(train, val, seed=7)

imputed = model.impute(test_masked)
np.save("test_imputed.npy", imputed)

backbone.save("brits_backbone.pypots")
model.save("pcrbrits.pt")
