"""
PCR-SAITS v1.1.0 — No-Feature-Group Ablation

Purpose
-------
Test the effect of removing feature-group/domain-tag information from PCR while
keeping the rest of the public PCR numerical path the same.

IMPORTANT:
- Simply passing feature_groups=None to the public API is NOT a no-group
  ablation. In metadata_mode="explicit" it raises ValueError.
- metadata_mode="paper_legacy_inference" also is NOT group-free; it infers
  pollutant/sensor/meteorological groups from feature names.
- This script therefore builds an experimental PCR corrector with exactly the
  public v1.1.0 PCR variant, then disables ONLY the 2 domain-tag inputs.
- It does not modify the installed pcrsaits package.

Comparisons per backbone:
    Backbone
    PCR-Group      = public PCR wrapper with explicit feature groups
    PCR-NoGroup    = same PCR core, but no domain-tag inputs

Backbones:
    SAITS, BRITS, CSDI

Dataset:
    UCI Air Quality

Outputs:
    raw_results.csv
    group_ablation.csv
"""

from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import json
import random
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import pcrsaits
from pcrsaits import (
    SAITSBackbone,
    BRITSBackbone,
    CSDIBackbone,
    PCRSAITS,
    PCRBRITS,
    PCRCSDI,
    build_windows,
    reconstruct_from_windows,
)

# Internal audited components used only for the experimental ablation.
from pcrsaits.corrector import PCRCorrector
from pcrsaits.model import PCRResidualNet
from pcrsaits.masks import (
    make_holdout_train_mask,
    build_correction_scope,
)

EXPECTED_VERSION = "1.1.0"
PUBLIC_VARIANT = "pcrsaitsv14_no_seasonal_branch"

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"

FEATURES = [
    "CO(GT)",
    "PT08.S1(CO)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "NOx(GT)",
    "PT08.S3(NOx)",
    "NO2(GT)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
]

FEATURE_GROUPS = [
    "pollutant",
    "sensor",
    "pollutant",
    "sensor",
    "pollutant",
    "sensor",
    "pollutant",
    "sensor",
    "sensor",
    "meteorological",
    "meteorological",
    "meteorological",
]

PATTERNS = ("pointwise", "block_12", "block_48")
N_STEPS = 48
STRIDE = 24
TEST_MISSING_RATE = 0.20
VAL_HOLDOUT_RATE = 0.10

# Same stronger settings used for the intermediate fair pilot.
SAITS_EPOCHS = 30
BRITS_EPOCHS = 30
CSDI_EPOCHS = 10
PCR_EPOCHS = 30

SAITS_PATIENCE = 5
BRITS_PATIENCE = 5
CSDI_PATIENCE = 3
PCR_PATIENCE = 5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--rows", type=int, default=3840)
    p.add_argument("--seeds", type=int, nargs="+", default=[7])
    p.add_argument(
        "--output-dir",
        type=str,
        default="pcr_no_feature_group_results",
    )
    p.add_argument("--reset", action="store_true")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def obtain_csv(local_path):
    if local_path:
        p = Path(local_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    # Reuse common locations first.
    candidates = [
        Path("AirQualityUCI.csv"),
        Path("uci_air_quality_data/AirQualityUCI.csv"),
        Path("../uci_air_quality_data/AirQualityUCI.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()

    data_dir = Path("uci_air_quality_data")
    data_dir.mkdir(exist_ok=True)
    zip_path = data_dir / "air_quality.zip"

    print("Downloading UCI Air Quality dataset...")
    urllib.request.urlretrieve(UCI_ZIP_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [
            n for n in zf.namelist()
            if n.replace("\\", "/").endswith("AirQualityUCI.csv")
        ]
        if not members:
            raise RuntimeError("AirQualityUCI.csv not found in archive.")
        zf.extract(members[0], data_dir)
        return (data_dir / members[0]).resolve()


def load_data(path):
    df = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        encoding="latin1",
        engine="python",
    )
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    x = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    x = x.replace(-200, np.nan).dropna(how="all")
    return x.to_numpy(dtype=np.float64)


def split_and_standardize(values, max_rows):
    values = values[: min(max_rows, len(values))]
    usable = (len(values) // N_STEPS) * N_STEPS
    values = values[:usable]

    n_blocks = len(values) // N_STEPS
    n_train_blocks = int(n_blocks * 0.60)
    n_val_blocks = int(n_blocks * 0.20)

    n_train = n_train_blocks * N_STEPS
    n_val = n_val_blocks * N_STEPS

    train = values[:n_train]
    val = values[n_train:n_train+n_val]
    test = values[n_train+n_val:]

    mean = np.nanmean(train, axis=0)
    std = np.nanstd(train, axis=0)
    std[~np.isfinite(std) | (std < 1e-12)] = 1.0

    return (
        (train - mean) / std,
        (val - mean) / std,
        (test - mean) / std,
    )


def pointwise_mask(values, rate, seed):
    rng = np.random.default_rng(seed)
    obs = np.flatnonzero(np.isfinite(values).ravel())
    n = max(1, int(round(rate * len(obs))))
    chosen = rng.choice(obs, size=min(n, len(obs)), replace=False)
    mask = np.zeros(values.size, dtype=bool)
    mask[chosen] = True
    return mask.reshape(values.shape)


def block_mask(values, rate, block_size, seed):
    rng = np.random.default_rng(seed)
    observed = np.isfinite(values)
    target = max(1, int(round(rate * observed.sum())))
    mask = np.zeros_like(observed, dtype=bool)
    T, F = values.shape

    attempts = 0
    while mask.sum() < target and attempts < max(10000, target * 50):
        attempts += 1
        f = int(rng.integers(0, F))
        start = int(rng.integers(0, max(1, T - block_size + 1)))
        stop = min(T, start + block_size)
        idx = np.flatnonzero(observed[start:stop, f] & ~mask[start:stop, f]) + start
        if not len(idx):
            continue
        remaining = target - int(mask.sum())
        mask[idx[:remaining], f] = True

    if mask.sum() < target:
        candidates = np.flatnonzero((observed & ~mask).ravel())
        need = min(target - int(mask.sum()), len(candidates))
        chosen = rng.choice(candidates, size=need, replace=False)
        mask.ravel()[chosen] = True

    return mask


def make_test_mask(values, pattern, seed):
    if pattern == "pointwise":
        return pointwise_mask(values, TEST_MISSING_RATE, seed)
    return block_mask(
        values,
        TEST_MISSING_RATE,
        int(pattern.split("_")[1]),
        seed,
    )


def apply_mask(values, mask):
    out = values.copy()
    out[mask] = np.nan
    return out


def val_windows(val, seed):
    mask = pointwise_mask(val, VAL_HOLDOUT_RATE, seed)
    masked = apply_mask(val, mask)
    vw, _ = build_windows(masked, N_STEPS, stride=STRIDE)
    vo, _ = build_windows(val, N_STEPS, stride=STRIDE)
    return vw, vo


def base_impute(backbone, masked):
    windows, starts = build_windows(masked, N_STEPS, stride=STRIDE)
    pred_w = backbone.impute(windows.astype(np.float32))
    pred = reconstruct_from_windows(pred_w, starts, len(masked))
    observed = np.isfinite(masked)
    pred[observed] = masked[observed]
    return np.asarray(pred, dtype=np.float64)


class NoFeatureGroupPCR:
    """
    Experimental wrapper that removes only the two PCR domain-tag inputs.

    Public PCR v1.1.0:
        variant = pcrsaitsv14_no_seasonal_branch
        local branch = ON
        seasonal branch = OFF
        domain tags = ON
        direct residual = ON
        input dim = 10

    This ablation:
        same settings, except
        domain tags = OFF
        input dim = 8
    """

    def __init__(
        self,
        *,
        backbone,
        feature_names,
        n_steps=N_STEPS,
        base_impute_stride=STRIDE,
        learning_rate=1e-3,
        weight_decay=1e-4,
        epochs=PCR_EPOCHS,
        batch_size=128,
        patience=PCR_PATIENCE,
        verbose=False,
    ):
        self.feature_names = list(feature_names)

        self.corrector = PCRCorrector(
            variant=PUBLIC_VARIANT,
            n_steps=n_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            preserve_loss_weight=0.0,
            rel_loss_weight=0.0,
            sparse_loss_weight=0.0,
            base_model=backbone,
            feature_names=self.feature_names,
            base_impute_stride=base_impute_stride,
            verbose=verbose,
        )

        # Surgical ablation: remove ONLY group/domain-tag inputs.
        self.corrector.use_domain_tags = False
        self.corrector.input_dim = 8
        self.corrector.model = PCRResidualNet(8).to(self.corrector.device)

        # Safety assertions: all other public-path flags remain unchanged.
        assert self.corrector.use_local_branch is True
        assert self.corrector.use_seasonal_branch is False
        assert self.corrector.direct_residual is True
        assert self.corrector.use_domain_tags is False
        assert self.corrector.input_dim == 8

    def fit(
        self,
        train_values,
        val_values,
        *,
        seed=7,
        holdout_ratio=0.15,
        pointwise_fraction=0.5,
        block_patterns=(6, 12, 24, 48),
        block_buffer=1,
    ):
        train_mask, train_gap = make_holdout_train_mask(
            train_values,
            seed=seed + 100,
            holdout_ratio=holdout_ratio,
            pointwise_fraction=pointwise_fraction,
            block_patterns=tuple(block_patterns),
            block_buffer=block_buffer,
        )
        val_mask, val_gap = make_holdout_train_mask(
            val_values,
            seed=seed + 200,
            holdout_ratio=holdout_ratio,
            pointwise_fraction=pointwise_fraction,
            block_patterns=tuple(block_patterns),
            block_buffer=block_buffer,
        )

        self.corrector.fit(
            train_values,
            train_mask,
            train_gap,
            val_values,
            val_mask,
            val_gap,
        )
        return self

    def impute(self, masked_values):
        base = self.corrector._impute_full_series_with_base(masked_values)
        correction_mask, gap_len = build_correction_scope(masked_values)
        return self.corrector.correct(
            masked_values,
            base,
            correction_mask,
            gap_len,
        )


def train_saits(train, val, seed):
    set_seed(seed)
    tw, _ = build_windows(train, N_STEPS, stride=STRIDE)
    vw, vo = val_windows(val, seed + 10001)
    m = SAITSBackbone(
        n_steps=N_STEPS,
        n_features=train.shape[1],
        epochs=SAITS_EPOCHS,
        batch_size=32,
        patience=SAITS_PATIENCE,
        d_model=64,
        d_ffn=128,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        verbose=False,
    )
    t0 = time.perf_counter()
    m.fit(tw, vw, vo)
    return m, time.perf_counter() - t0


def train_brits(train, val, seed):
    set_seed(seed)
    tw, _ = build_windows(train, N_STEPS, stride=STRIDE)
    vw, vo = val_windows(val, seed + 20001)
    m = BRITSBackbone(
        n_steps=N_STEPS,
        n_features=train.shape[1],
        epochs=BRITS_EPOCHS,
        batch_size=32,
        patience=BRITS_PATIENCE,
        rnn_hidden_size=64,
        verbose=False,
    )
    t0 = time.perf_counter()
    m.fit(tw, vw, vo)
    return m, time.perf_counter() - t0


def train_csdi(train, seed):
    set_seed(seed)
    tw, _ = build_windows(train, N_STEPS, stride=STRIDE)
    m = CSDIBackbone(
        n_steps=N_STEPS,
        n_features=train.shape[1],
        epochs=CSDI_EPOCHS,
        batch_size=16,
        patience=CSDI_PATIENCE,
        n_layers=2,
        n_heads=2,
        n_channels=16,
        d_time_embedding=16,
        d_feature_embedding=8,
        d_diffusion_embedding=32,
        n_diffusion_steps=20,
        target_strategy="random",
        is_unconditional=False,
        schedule="quad",
        beta_start=0.0001,
        beta_end=0.5,
        n_sampling_times=10,
        aggregation="median",
        sampling_seed=seed,
        verbose=False,
    )
    t0 = time.perf_counter()
    m.fit(tw)
    return m, time.perf_counter() - t0


def fit_public_pcr(cls, backbone, train, val, seed):
    set_seed(seed)
    m = cls(
        backbone=backbone,
        feature_names=FEATURES,
        feature_groups=FEATURE_GROUPS,
        metadata_mode="explicit",
        n_steps=N_STEPS,
        base_impute_stride=STRIDE,
        learning_rate=1e-3,
        weight_decay=1e-4,
        epochs=PCR_EPOCHS,
        batch_size=128,
        patience=PCR_PATIENCE,
        verbose=False,
    )
    t0 = time.perf_counter()
    m.fit(
        train,
        val,
        seed=seed,
        holdout_ratio=0.15,
        pointwise_fraction=0.5,
        block_patterns=(6, 12, 24, 48),
        block_buffer=1,
    )
    return m, time.perf_counter() - t0


def fit_no_group_pcr(backbone, train, val, seed):
    set_seed(seed)
    m = NoFeatureGroupPCR(
        backbone=backbone,
        feature_names=FEATURES,
        n_steps=N_STEPS,
        base_impute_stride=STRIDE,
        learning_rate=1e-3,
        weight_decay=1e-4,
        epochs=PCR_EPOCHS,
        batch_size=128,
        patience=PCR_PATIENCE,
        verbose=False,
    )
    t0 = time.perf_counter()
    m.fit(
        train,
        val,
        seed=seed,
        holdout_ratio=0.15,
        pointwise_fraction=0.5,
        block_patterns=(6, 12, 24, 48),
        block_buffer=1,
    )
    return m, time.perf_counter() - t0


def score(truth, pred, mask):
    err = truth[mask] - pred[mask]
    return float(np.sqrt(np.mean(err**2))), float(np.mean(np.abs(err)))


def evaluate(method, fn, truth, masked, mask, train_sec):
    t0 = time.perf_counter()
    pred = fn(masked)
    infer_sec = time.perf_counter() - t0

    if not np.isfinite(pred[mask]).all():
        raise RuntimeError(f"{method}: non-finite predictions.")

    observed = np.isfinite(masked)
    if not np.allclose(pred[observed], masked[observed], rtol=0, atol=0):
        raise RuntimeError(f"{method}: observed values changed.")

    rmse, mae = score(truth, pred, mask)
    return {
        "method": method,
        "rmse": rmse,
        "mae": mae,
        "train_seconds": train_sec,
        "inference_seconds": infer_sec,
    }


def make_ablation_table(raw):
    rows = []
    for (seed, pattern, backbone), g in raw.groupby(
        ["seed", "pattern", "backbone"]
    ):
        idx = g.set_index("condition")
        base_rmse = float(idx.loc["Backbone", "rmse"])
        group_rmse = float(idx.loc["PCR-Group", "rmse"])
        nogroup_rmse = float(idx.loc["PCR-NoGroup", "rmse"])

        base_mae = float(idx.loc["Backbone", "mae"])
        group_mae = float(idx.loc["PCR-Group", "mae"])
        nogroup_mae = float(idx.loc["PCR-NoGroup", "mae"])

        rows.append({
            "seed": int(seed),
            "pattern": pattern,
            "backbone": backbone,

            "base_rmse": base_rmse,
            "group_rmse": group_rmse,
            "nogroup_rmse": nogroup_rmse,

            "pcr_group_vs_base_rmse_pct":
                100 * (base_rmse - group_rmse) / base_rmse,
            "pcr_nogroup_vs_base_rmse_pct":
                100 * (base_rmse - nogroup_rmse) / base_rmse,

            # Positive means GROUPED PCR is better than NO-GROUP PCR.
            "group_benefit_rmse_pct":
                100 * (nogroup_rmse - group_rmse) / nogroup_rmse,

            "base_mae": base_mae,
            "group_mae": group_mae,
            "nogroup_mae": nogroup_mae,

            "pcr_group_vs_base_mae_pct":
                100 * (base_mae - group_mae) / base_mae,
            "pcr_nogroup_vs_base_mae_pct":
                100 * (base_mae - nogroup_mae) / base_mae,

            # Positive means GROUPED PCR is better than NO-GROUP PCR.
            "group_benefit_mae_pct":
                100 * (nogroup_mae - group_mae) / nogroup_mae,
        })

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_results.csv"

    version = importlib_metadata.version("pcrsaits")
    print("=" * 96)
    print("PCR-SAITS v1.1.0 — NO-FEATURE-GROUP ABLATION")
    print("=" * 96)
    print("pcrsaits version :", version)
    print("pcrsaits path    :", pcrsaits.__file__)
    print("seeds            :", args.seeds)
    print("patterns         :", PATTERNS)
    print()

    if version != EXPECTED_VERSION:
        raise RuntimeError(f"Expected pcrsaits==1.1.0, found {version}")

    if "site-packages" not in str(pcrsaits.__file__).lower():
        raise RuntimeError("Run from the pip-installed environment, not repo source.")

    # Demonstrate that public explicit mode truly rejects missing groups.
    try:
        PCRSAITS(
            backbone=object(),
            feature_names=["x"],
            feature_groups=None,
            metadata_mode="explicit",
        )
    except Exception:
        pass

    csv_path = obtain_csv(args.data)
    values = load_data(csv_path)
    train, val, test = split_and_standardize(values, args.rows)

    print("dataset          :", csv_path)
    print("train/val/test   :", train.shape, val.shape, test.shape)
    print()

    if args.reset and raw_path.exists():
        raw_path.unlink()

    if raw_path.exists():
        existing = pd.read_csv(raw_path)
        rows = existing.to_dict("records")
        completed = set(existing["seed"].astype(int).unique())
    else:
        rows = []
        completed = set()

    todo = [int(s) for s in args.seeds if int(s) not in completed]

    for seed in todo:
        print("\n" + "#" * 96)
        print(f"SEED {seed}")
        print("#" * 96)

        masks = {}
        masked = {}
        for i, pattern in enumerate(PATTERNS):
            m = make_test_mask(test, pattern, seed * 1000 + i)
            masks[pattern] = m
            masked[pattern] = apply_mask(test, m)

        # Train each backbone ONCE.
        print("[1/9] Train SAITS")
        saits, saits_sec = train_saits(train, val, seed)

        print("[2/9] Fit PCR-SAITS Group")
        saits_group, sg_sec = fit_public_pcr(PCRSAITS, saits, train, val, seed)

        print("[3/9] Fit PCR-SAITS NoGroup")
        saits_nogroup, sn_sec = fit_no_group_pcr(saits, train, val, seed)

        print("[4/9] Train BRITS")
        brits, brits_sec = train_brits(train, val, seed)

        print("[5/9] Fit PCR-BRITS Group")
        brits_group, bg_sec = fit_public_pcr(PCRBRITS, brits, train, val, seed)

        print("[6/9] Fit PCR-BRITS NoGroup")
        brits_nogroup, bn_sec = fit_no_group_pcr(brits, train, val, seed)

        print("[7/9] Train CSDI")
        csdi, csdi_sec = train_csdi(train, seed)

        print("[8/9] Fit PCR-CSDI Group")
        csdi_group, cg_sec = fit_public_pcr(PCRCSDI, csdi, train, val, seed)

        print("[9/9] Fit PCR-CSDI NoGroup")
        csdi_nogroup, cn_sec = fit_no_group_pcr(csdi, train, val, seed)

        specs = [
            ("SAITS", "Backbone",
             lambda x: base_impute(saits, x), saits_sec),
            ("SAITS", "PCR-Group",
             saits_group.impute, saits_sec + sg_sec),
            ("SAITS", "PCR-NoGroup",
             saits_nogroup.impute, saits_sec + sn_sec),

            ("BRITS", "Backbone",
             lambda x: base_impute(brits, x), brits_sec),
            ("BRITS", "PCR-Group",
             brits_group.impute, brits_sec + bg_sec),
            ("BRITS", "PCR-NoGroup",
             brits_nogroup.impute, brits_sec + bn_sec),

            ("CSDI", "Backbone",
             lambda x: base_impute(csdi, x), csdi_sec),
            ("CSDI", "PCR-Group",
             csdi_group.impute, csdi_sec + cg_sec),
            ("CSDI", "PCR-NoGroup",
             csdi_nogroup.impute, csdi_sec + cn_sec),
        ]

        for pattern in PATTERNS:
            for backbone_name, condition, fn, train_sec in specs:
                r = evaluate(
                    f"{backbone_name}-{condition}",
                    fn,
                    test,
                    masked[pattern],
                    masks[pattern],
                    train_sec,
                )
                r.update({
                    "seed": seed,
                    "pattern": pattern,
                    "backbone": backbone_name,
                    "condition": condition,
                })
                rows.append(r)

        raw = pd.DataFrame(rows)
        raw.to_csv(raw_path, index=False)

        abl = make_ablation_table(raw)
        abl.to_csv(out_dir / "group_ablation.csv", index=False)

        this = abl[abl["seed"] == seed]
        print("\nGROUP ABLATION — seed", seed)
        print(
            this[
                [
                    "pattern",
                    "backbone",
                    "pcr_group_vs_base_rmse_pct",
                    "pcr_nogroup_vs_base_rmse_pct",
                    "group_benefit_rmse_pct",
                ]
            ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
        )

    raw = pd.DataFrame(rows)
    abl = make_ablation_table(raw)

    raw.to_csv(raw_path, index=False)
    abl.to_csv(out_dir / "group_ablation.csv", index=False)

    summary = (
        abl.groupby(["pattern", "backbone"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            group_vs_base_rmse_mean=("pcr_group_vs_base_rmse_pct", "mean"),
            nogroup_vs_base_rmse_mean=("pcr_nogroup_vs_base_rmse_pct", "mean"),
            group_benefit_rmse_mean=("group_benefit_rmse_pct", "mean"),
            group_benefit_rmse_std=("group_benefit_rmse_pct", "std"),
            group_vs_base_mae_mean=("pcr_group_vs_base_mae_pct", "mean"),
            nogroup_vs_base_mae_mean=("pcr_nogroup_vs_base_mae_pct", "mean"),
            group_benefit_mae_mean=("group_benefit_mae_pct", "mean"),
            group_benefit_mae_std=("group_benefit_mae_pct", "std"),
        )
    )
    summary.to_csv(out_dir / "group_ablation_summary.csv", index=False)

    config = {
        "pcrsaits_version": version,
        "study": "no_feature_group_ablation",
        "public_variant": PUBLIC_VARIANT,
        "group_model_input_dim": 10,
        "nogroup_model_input_dim": 8,
        "only_changed_factor": "use_domain_tags",
        "seeds": [int(x) for x in args.seeds],
        "patterns": list(PATTERNS),
        "missing_rate": TEST_MISSING_RATE,
    }
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 110)
    print("SUMMARY")
    print("group_benefit > 0 means explicit feature groups improved PCR.")
    print("group_benefit < 0 means NoGroup PCR performed better.")
    print("=" * 110)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nSaved:")
    print(" ", raw_path)
    print(" ", out_dir / "group_ablation.csv")
    print(" ", out_dir / "group_ablation_summary.csv")
    print(" ", out_dir / "config.json")


if __name__ == "__main__":
    main()
