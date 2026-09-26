#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Focused reviewer-experiments runner for PCR-SAITS.

This script addresses four reviewer-facing questions in a separate experiment file:
1) Linear/Ridge residual-corrector baselines
2) XGBoost residual-corrector baseline
3) Ablation without domain tags
4) MCAR-only vs mixed-holdout training ablation

It reuses core utilities from the PCR-SAITS full/paper suite script in the same folder,
then runs a focused comparison on top of a frozen SAITS backbone.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception as e:
    raise RuntimeError("This script requires torch.") from e

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

try:
    from sklearn.linear_model import LinearRegression, Ridge
except Exception as e:
    raise RuntimeError("This script requires scikit-learn.") from e

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

DEFAULT_DATASETS = ["airquality", "beijingpm25", "synthetic"]
DEFAULT_METHODS = [
    "saits",
    "pcr_mlp_mixed",
    "pcr_mlp_mcar_only",
    "pcr_mlp_no_domain_tags",
    "pcr_linear",
    "pcr_ridge",
    "pcr_xgboost",
]
DEFAULT_MECHANISMS = ["MCAR", "MAR", "MNAR"]
DEFAULT_PATTERNS = ["pointwise", "block_6", "block_12", "block_24", "block_48"]
DEFAULT_RATES = [0.1, 0.2, 0.3, 0.4, 0.5]
DEFAULT_SEEDS = [7, 21, 42]
POLLUTANT_NAMES = {
    "CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)",
    "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
}
METEO_NAMES = {"T", "RH", "AH", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"}


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_mean(values) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else float("nan")


def safe_std(values) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0


def choose_core_script(script_dir: Path, explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = script_dir / explicit
        if not p.exists():
            raise FileNotFoundError(f"Core script not found: {p}")
        return p
    candidates = [
        "pcr_saits_v1_4_paper_suite_patched6.py",
        "pcr_saits_v1_4_paper_suite_patched5.py",
        "pcr_saits_v1_4_paper_suite_patched4.py",
        "pcr_saits_v1_4_paper_suite_patched3.py",
        "pcr_saits_v1_4_paper_suite_patched2.py",
        "pcr_saits_v1_4_paper_suite_patched.py",
        "pcr_saits_v1_4_paper_suite.py",
        "pcr_saits_v1_4_full_protocol_final_patched.py",
        "pcr_saits_v1_4_full_protocol_final.py",
    ]
    for name in candidates:
        p = script_dir / name
        if p.exists():
            return p
    # fallback: any compatible suite script in the same folder
    for p in sorted(script_dir.glob('pcr_saits_v1_4_*.py')):
        if p.name != Path(__file__).name:
            return p
    raise FileNotFoundError(
        "Could not find a compatible PCR-SAITS core script in the same folder. "
        "Place this file next to one of the pcr_saits_v1_4_* suite scripts or pass --core-script explicitly."
    )


def load_core_module(path: Path):
    spec = importlib.util.spec_from_file_location("pcr_core_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import core script from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dataset_group_id(feature_name: str) -> int:
    if feature_name in POLLUTANT_NAMES:
        return 0
    if feature_name.startswith("PT08."):
        return 1
    if feature_name in METEO_NAMES:
        return 2
    if feature_name.lower().startswith("synthetic"):
        return 3
    return 1


def dataset_group_name(feature_name: str) -> str:
    gid = dataset_group_id(feature_name)
    if gid == 0:
        return "pollutant"
    if gid == 1:
        return "sensor"
    if gid == 2:
        return "meteorological"
    return "synthetic"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred) & mask.astype(bool)
    if not np.any(valid):
        return {"rmse": float("nan"), "mae": float("nan")}
    err = y_pred[valid] - y_true[valid]
    return {
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "mae": float(np.mean(np.abs(err))),
    }


def compute_grouped_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray, features: Sequence[str]) -> Dict[str, float]:
    # Synthetic features do not map meaningfully to pollutant/sensor/meteorological groups.
    if features and all(dataset_group_name(f) == "synthetic" for f in features):
        return {
            "pollutant_rmse": float("nan"),
            "pollutant_mae": float("nan"),
            "sensor_rmse": float("nan"),
            "sensor_mae": float("nan"),
            "meteorological_rmse": float("nan"),
            "meteorological_mae": float("nan"),
        }
    out = {}
    for g in ["pollutant", "sensor", "meteorological"]:
        idx = [i for i, f in enumerate(features) if dataset_group_name(f) == g]
        if idx:
            metrics = compute_metrics(y_true[:, idx], y_pred[:, idx], mask[:, idx])
            out[f"{g}_rmse"] = metrics["rmse"]
            out[f"{g}_mae"] = metrics["mae"]
        else:
            out[f"{g}_rmse"] = float("nan")
            out[f"{g}_mae"] = float("nan")
    return out


def pointwise_holdout_mask(values: np.ndarray, seed: int, holdout_ratio: float = 0.15):
    rng = np.random.default_rng(seed)
    observed = np.isfinite(values)
    candidates = np.argwhere(observed)
    mask = np.zeros_like(observed, dtype=bool)
    gap_len = np.zeros_like(values, dtype=float)
    target = max(1, int(round(observed.sum() * holdout_ratio)))
    if len(candidates) > 0:
        chosen_idx = rng.choice(len(candidates), size=min(target, len(candidates)), replace=False)
        chosen = candidates[chosen_idx]
        mask[chosen[:, 0], chosen[:, 1]] = True
        gap_len[chosen[:, 0], chosen[:, 1]] = 1.0
    return mask, gap_len


class ResidualExampleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.7)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class ResidualCorrectorBase:
    name: str = "base"

    def __init__(self, feature_names: Sequence[str], base_imputer, core, train_mode: str = "mixed", use_domain_tags: bool = True):
        self.feature_names = list(feature_names)
        self.base_imputer = base_imputer
        self.core = core
        self.train_mode = train_mode
        self.use_domain_tags = use_domain_tags
        self.input_dim = 10 if use_domain_tags else 8
        self.window = int(
            getattr(base_imputer, "n_steps", 0)
            or getattr(getattr(base_imputer, "model", None), "n_steps", 0)
            or 48
        )

    def _build_training_holdout(self, values: np.ndarray, seed: int, holdout_ratio: float = 0.15):
        if self.train_mode == "mcar_only":
            return pointwise_holdout_mask(values, seed=seed, holdout_ratio=holdout_ratio)
        return self.core.make_holdout_train_mask(values, seed=seed, holdout_ratio=holdout_ratio)

    def _impute_full_series_with_base(self, masked_values: np.ndarray, stride: int):
        windows, starts = self.core.build_windows(masked_values, self.window, stride=max(1, int(stride)))
        imputed_windows = self.base_imputer.impute(windows.astype(np.float32))
        return self.core.reconstruct_from_windows(imputed_windows, starts, len(masked_values))

    def _make_feature_row(self, base: float, local_val: float, seasonal_val: float, local_ok: float, seasonal_ok: float, gl: float, feat_group: int):
        row = [
            base,
            local_val,
            seasonal_val,
            base - local_val,
            base - seasonal_val,
            local_ok,
            seasonal_ok,
            min(float(gl) / 48.0, 1.0),
        ]
        if self.use_domain_tags:
            row.extend([float(feat_group == 0), float(feat_group == 1)])
        return np.asarray(row, dtype=np.float32)

    def _build_examples(self, original_values, masked_values, base_imputed_values, target_mask, gap_len):
        feats, tgts = [], []
        local_vals_all, local_ok_all = self.core._precompute_local_estimates(masked_values)
        n_t, n_f = original_values.shape
        for t in range(n_t):
            for f in range(n_f):
                if not target_mask[t, f]:
                    continue
                base = float(base_imputed_values[t, f])
                true = float(original_values[t, f])
                local_val = float(local_vals_all[t, f])
                local_ok = float(local_ok_all[t, f])
                seasonal_val, seasonal_ok = self.core.seasonal_estimate(masked_values, t, f)
                feat_group = dataset_group_id(self.feature_names[f])
                feats.append(self._make_feature_row(base, local_val, seasonal_val, local_ok, seasonal_ok, gap_len[t, f], feat_group))
                tgts.append(np.float32(true - base))
        if not feats:
            return np.zeros((0, self.input_dim), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        return np.stack(feats), np.asarray(tgts, dtype=np.float32)

    def fit(self, train_values, val_values, eval_stride: int, seed: int):
        raise NotImplementedError

    def correct(self, original_masked_values, base_imputed_values, artificial_mask, gap_len):
        raise NotImplementedError


class MLPResidualCorrector(ResidualCorrectorBase):
    def __init__(self, feature_names, base_imputer, core, train_mode="mixed", use_domain_tags=True, learning_rate=1e-3, weight_decay=1e-4, epochs=30, batch_size=256, patience=8, verbose=True):
        super().__init__(feature_names, base_imputer, core, train_mode=train_mode, use_domain_tags=use_domain_tags)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.verbose = verbose
        self.device = core.choose_device()
        self.model = ResidualMLP(self.input_dim).to(self.device)

    def fit(self, train_values, val_values, eval_stride: int, seed: int):
        train_mask, train_gap = self._build_training_holdout(train_values, seed=seed * 100 + 1)
        val_mask, val_gap = self._build_training_holdout(val_values, seed=seed * 100 + 2)
        train_masked = self.core.apply_mask(train_values, train_mask)
        val_masked = self.core.apply_mask(val_values, val_mask)
        train_base = self._impute_full_series_with_base(train_masked, stride=eval_stride)
        val_base = self._impute_full_series_with_base(val_masked, stride=eval_stride)
        train_X, train_y = self._build_examples(train_values, train_masked, train_base, train_mask, train_gap)
        val_X, val_y = self._build_examples(val_values, val_masked, val_base, val_mask, val_gap)
        if len(train_X) == 0 or len(val_X) == 0:
            raise RuntimeError("Empty residual-corrector training data.")
        train_loader = DataLoader(ResidualExampleDataset(train_X, train_y), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(ResidualExampleDataset(val_X, val_y), batch_size=self.batch_size, shuffle=False)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        best_state, best_loss, bad_epochs = None, float("inf"), 0
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_sum, train_count = 0.0, 0
            for X, y in train_loader:
                X = X.to(self.device)
                y = y.to(self.device).unsqueeze(-1)
                opt.zero_grad(set_to_none=True)
                pred = self.model(X)
                loss = F.smooth_l1_loss(pred, y)
                if not torch.isfinite(loss):
                    if self.verbose:
                        print(f"[WARN] Non-finite MLP residual loss; skipping batch.")
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                opt.step()
                train_sum += float(loss.detach().cpu()) * len(X)
                train_count += len(X)
            self.model.eval()
            val_sum, val_count = 0.0, 0
            with torch.no_grad():
                for X, y in val_loader:
                    X = X.to(self.device)
                    y = y.to(self.device).unsqueeze(-1)
                    pred = self.model(X)
                    loss = F.smooth_l1_loss(pred, y)
                    if not torch.isfinite(loss):
                        continue
                    val_sum += float(loss.detach().cpu()) * len(X)
                    val_count += len(X)
            train_loss = train_sum / max(1, train_count)
            val_loss = val_sum / max(1, val_count)
            if self.verbose:
                print(f"[INFO] {self.__class__.__name__}({self.train_mode},{'tags' if self.use_domain_tags else 'no_tags'}) epoch={epoch}/{self.epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
            if val_loss + 1e-8 < best_loss:
                best_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break
        if best_state is None:
            if self.verbose:
                print("[WARN] No valid MLP corrector state found; keeping latest model state.")
        else:
            self.model.load_state_dict(best_state)
        self.model.eval()

    def correct(self, original_masked_values, base_imputed_values, artificial_mask, gap_len):
        corrected = base_imputed_values.copy()
        local_vals_all, local_ok_all = self.core._precompute_local_estimates(original_masked_values)
        rows, coords = [], []
        n_t, n_f = corrected.shape
        for t in range(n_t):
            for f in range(n_f):
                if not artificial_mask[t, f]:
                    continue
                base = float(base_imputed_values[t, f])
                local_val = float(local_vals_all[t, f])
                local_ok = float(local_ok_all[t, f])
                seasonal_val, seasonal_ok = self.core.seasonal_estimate(original_masked_values, t, f)
                feat_group = dataset_group_id(self.feature_names[f])
                rows.append(self._make_feature_row(base, local_val, seasonal_val, local_ok, seasonal_ok, gap_len[t, f], feat_group))
                coords.append((t, f))
        if rows:
            X = torch.tensor(np.stack(rows), dtype=torch.float32, device=self.device)
            with torch.no_grad():
                residual = self.model(X).squeeze(-1).cpu().numpy()
            for (t, f), r in zip(coords, residual):
                corrected[t, f] = corrected[t, f] + float(r)
        observed = np.isfinite(original_masked_values)
        corrected[observed] = original_masked_values[observed]
        return corrected


class LinearResidualCorrector(ResidualCorrectorBase):
    def __init__(self, feature_names, base_imputer, core, train_mode="mixed", use_domain_tags=True):
        super().__init__(feature_names, base_imputer, core, train_mode=train_mode, use_domain_tags=use_domain_tags)
        self.model = LinearRegression()
        self.residual_scale = 1.0

    def fit(self, train_values, val_values, eval_stride: int, seed: int):
        train_mask, train_gap = self._build_training_holdout(train_values, seed=seed * 100 + 1)
        train_masked = self.core.apply_mask(train_values, train_mask)
        train_base = self._impute_full_series_with_base(train_masked, stride=eval_stride)
        train_X, train_y = self._build_examples(train_values, train_masked, train_base, train_mask, train_gap)
        if len(train_X) == 0:
            raise RuntimeError("Empty training data for linear residual corrector.")
        self.model.fit(train_X, train_y)
        self.residual_scale = 1.0

    def correct(self, original_masked_values, base_imputed_values, artificial_mask, gap_len):
        corrected = base_imputed_values.copy()
        local_vals_all, local_ok_all = self.core._precompute_local_estimates(original_masked_values)
        rows, coords = [], []
        n_t, n_f = corrected.shape
        for t in range(n_t):
            for f in range(n_f):
                if not artificial_mask[t, f]:
                    continue
                base = float(base_imputed_values[t, f])
                local_val = float(local_vals_all[t, f])
                local_ok = float(local_ok_all[t, f])
                seasonal_val, seasonal_ok = self.core.seasonal_estimate(original_masked_values, t, f)
                feat_group = dataset_group_id(self.feature_names[f])
                rows.append(self._make_feature_row(base, local_val, seasonal_val, local_ok, seasonal_ok, gap_len[t, f], feat_group))
                coords.append((t, f))
        if rows:
            residual = self.residual_scale * np.asarray(self.model.predict(np.stack(rows)), dtype=float)
            for (t, f), r in zip(coords, residual):
                corrected[t, f] = corrected[t, f] + float(r)
        observed = np.isfinite(original_masked_values)
        corrected[observed] = original_masked_values[observed]
        return corrected


class RidgeResidualCorrector(LinearResidualCorrector):
    def __init__(self, feature_names, base_imputer, core, train_mode="mixed", use_domain_tags=True, alpha: float = 1.0):
        super().__init__(feature_names, base_imputer, core, train_mode=train_mode, use_domain_tags=use_domain_tags)
        self.model = Ridge(alpha=alpha, random_state=0)


class XGBResidualCorrector(ResidualCorrectorBase):
    def __init__(self, feature_names, base_imputer, core, train_mode="mixed", use_domain_tags=True):
        if XGBRegressor is None:
            raise RuntimeError("xgboost is not installed.")
        super().__init__(feature_names, base_imputer, core, train_mode=train_mode, use_domain_tags=use_domain_tags)
        self.model = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            reg_lambda=1.0,
            random_state=0,
            n_jobs=min(4, os.cpu_count() or 1),
        )

    def fit(self, train_values, val_values, eval_stride: int, seed: int):
        train_mask, train_gap = self._build_training_holdout(train_values, seed=seed * 100 + 1)
        val_mask, val_gap = self._build_training_holdout(val_values, seed=seed * 100 + 2)
        train_masked = self.core.apply_mask(train_values, train_mask)
        val_masked = self.core.apply_mask(val_values, val_mask)
        train_base = self._impute_full_series_with_base(train_masked, stride=eval_stride)
        val_base = self._impute_full_series_with_base(val_masked, stride=eval_stride)
        X, y = self._build_examples(train_values, train_masked, train_base, train_mask, train_gap)
        val_X, val_y = self._build_examples(val_values, val_masked, val_base, val_mask, val_gap)
        if len(X) == 0:
            raise RuntimeError("Empty training data for XGBoost residual corrector.")
        fit_kwargs = {"eval_set": [(val_X, val_y)], "verbose": False} if len(val_X) else {"verbose": False}
        try:
            self.model.fit(X, y, early_stopping_rounds=20, **fit_kwargs)
        except TypeError:
            try:
                from xgboost.callback import EarlyStopping
                self.model.fit(X, y, callbacks=[EarlyStopping(rounds=20, save_best=True)], **fit_kwargs)
            except Exception:
                self.model.fit(X, y, **fit_kwargs)

    def correct(self, original_masked_values, base_imputed_values, artificial_mask, gap_len):
        corrected = base_imputed_values.copy()
        local_vals_all, local_ok_all = self.core._precompute_local_estimates(original_masked_values)
        rows, coords = [], []
        n_t, n_f = corrected.shape
        for t in range(n_t):
            for f in range(n_f):
                if not artificial_mask[t, f]:
                    continue
                base = float(base_imputed_values[t, f])
                local_val = float(local_vals_all[t, f])
                local_ok = float(local_ok_all[t, f])
                seasonal_val, seasonal_ok = self.core.seasonal_estimate(original_masked_values, t, f)
                feat_group = dataset_group_id(self.feature_names[f])
                rows.append(self._make_feature_row(base, local_val, seasonal_val, local_ok, seasonal_ok, gap_len[t, f], feat_group))
                coords.append((t, f))
        if rows:
            residual = self.model.predict(np.stack(rows))
            for (t, f), r in zip(coords, residual):
                corrected[t, f] = corrected[t, f] + float(r)
        observed = np.isfinite(original_masked_values)
        corrected[observed] = original_masked_values[observed]
        return corrected


def resolve_dataset_path(script_dir: Path, dataset: str) -> Path:
    ds = dataset.lower()
    if ds in {"airquality", "airqualityuci", "uci"}:
        for name in ["AirQualityUCI.xlsx", "AirQualityUCI.xls", "AirQualityUCI.csv", "AirQualityUCI.data"]:
            p = script_dir / name
            if p.exists():
                return p
        raise FileNotFoundError("Could not find AirQualityUCI dataset in the same folder.")
    if ds in {"beijingpm25", "beijing", "pm25"}:
        for name in ["BeijingPM25.csv", "BeijingPM25.xlsx", "beijingpm25.csv", "beijingpm25.xlsx"]:
            p = script_dir / name
            if p.exists():
                return p
        raise FileNotFoundError("Could not find BeijingPM25.csv in the same folder.")
    if ds in {"synthetic", "synth"}:
        return script_dir / "__synthetic__.synthetic"
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_dataset(core, path: Path):
    name = path.name.lower()
    if name.endswith(".synthetic") or "__synthetic__" in name:
        return core.load_synthetic_dataset(path)
    if "beijing" in name:
        return core.load_beijing_pm25_dataset(path)
    return core.load_air_quality_dataset(path)


def build_val_windows(core, val_std: np.ndarray, window: int, stride: int, seed: int):
    if hasattr(core, "build_val_saits_windows"):
        return core.build_val_saits_windows(val_std, window=window, stride=stride, seed=seed)
    val_windows, _ = core.build_windows(val_std, window, stride=stride)
    x_ori = val_windows.copy().astype(np.float32)
    masked = val_windows.copy()
    rng = np.random.default_rng(seed)
    for i in range(len(masked)):
        finite = np.isfinite(masked[i])
        drop = rng.random(masked[i].shape) < 0.1
        masked[i][finite & drop] = np.nan
    return masked.astype(np.float32), x_ori


def build_mask_seed(seed: int, mechanism: str, pattern: str, rate: float) -> int:
    payload = f"{mechanism}_{pattern}_{rate:.3f}"
    h = int(__import__("hashlib").md5(payload.encode()).hexdigest(), 16)
    return seed * 1000 + int(rate * 1000) + h % 997


def create_corrector(method: str, feature_names: Sequence[str], base_imputer, core, args):
    if method == "pcr_mlp_mixed":
        return MLPResidualCorrector(feature_names, base_imputer, core, train_mode="mixed", use_domain_tags=True,
                                    learning_rate=args.pcr_lr, weight_decay=args.pcr_weight_decay, epochs=args.pcr_epochs,
                                    batch_size=args.pcr_batch_size, patience=args.pcr_patience, verbose=not args.quiet)
    if method == "pcr_mlp_mcar_only":
        return MLPResidualCorrector(feature_names, base_imputer, core, train_mode="mcar_only", use_domain_tags=True,
                                    learning_rate=args.pcr_lr, weight_decay=args.pcr_weight_decay, epochs=args.pcr_epochs,
                                    batch_size=args.pcr_batch_size, patience=args.pcr_patience, verbose=not args.quiet)
    if method == "pcr_mlp_no_domain_tags":
        return MLPResidualCorrector(feature_names, base_imputer, core, train_mode="mixed", use_domain_tags=False,
                                    learning_rate=args.pcr_lr, weight_decay=args.pcr_weight_decay, epochs=args.pcr_epochs,
                                    batch_size=args.pcr_batch_size, patience=args.pcr_patience, verbose=not args.quiet)
    if method == "pcr_linear":
        return LinearResidualCorrector(feature_names, base_imputer, core, train_mode="mixed", use_domain_tags=True)
    if method == "pcr_ridge":
        return RidgeResidualCorrector(feature_names, base_imputer, core, train_mode="mixed", use_domain_tags=True, alpha=args.ridge_alpha)
    if method == "pcr_xgboost":
        return XGBResidualCorrector(feature_names, base_imputer, core, train_mode="mixed", use_domain_tags=True)
    raise ValueError(f"Unknown corrector method: {method}")


def summarize_results(df: pd.DataFrame, output_dir: Path):
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (dataset, method), sub in df.groupby(["dataset", "method"]):
        rows.append({
            "dataset": dataset,
            "method": method,
            "overall_rmse_mean": safe_mean(sub["rmse"]),
            "overall_rmse_std": safe_std(sub["rmse"]),
            "overall_mae_mean": safe_mean(sub["mae"]),
            "overall_mae_std": safe_std(sub["mae"]),
            "pointwise_rmse_mean": safe_mean(sub["pointwise_rmse"]),
            "long_gap_rmse_mean": safe_mean(sub["long_gap_rmse"]),
            "pollutant_rmse_mean": safe_mean(sub["pollutant_rmse"]),
            "sensor_rmse_mean": safe_mean(sub["sensor_rmse"]),
            "meteorological_rmse_mean": safe_mean(sub["meteorological_rmse"]),
            "train_time_sec_mean": safe_mean(sub["train_time_sec"]),
            "infer_time_sec_mean": safe_mean(sub["infer_time_sec"]),
        })
    summary = pd.DataFrame(rows)
    ref = summary[summary["method"] == "saits"][ ["dataset", "overall_rmse_mean", "overall_mae_mean"] ].rename(columns={"overall_rmse_mean": "ref_rmse", "overall_mae_mean": "ref_mae"})
    summary = summary.merge(ref, on="dataset", how="left")
    summary["rmse_improvement_vs_saits_pct"] = (summary["ref_rmse"] - summary["overall_rmse_mean"]) / summary["ref_rmse"] * 100.0
    summary["mae_improvement_vs_saits_pct"] = (summary["ref_mae"] - summary["overall_mae_mean"]) / summary["ref_mae"] * 100.0
    summary.to_csv(output_dir / "summary_results.csv", index=False)
    return summary


def compute_pairwise_wilcoxon(df: pd.DataFrame) -> pd.DataFrame:
    if wilcoxon is None or df.empty:
        return pd.DataFrame()
    rows = []
    keys = ["dataset", "seed", "mechanism", "pattern", "rate"]
    methods = sorted(df["method"].unique().tolist())
    for dataset, ds_sub in df.groupby("dataset"):
        for i, m1 in enumerate(methods):
            for m2 in methods[i+1:]:
                a = ds_sub[ds_sub["method"] == m1][keys + ["rmse", "mae"]].rename(columns={"rmse": "rmse_1", "mae": "mae_1"})
                b = ds_sub[ds_sub["method"] == m2][keys + ["rmse", "mae"]].rename(columns={"rmse": "rmse_2", "mae": "mae_2"})
                merged = a.merge(b, on=keys, how="inner")
                if len(merged) < 5:
                    continue
                try:
                    p_rmse = float(wilcoxon(merged["rmse_1"], merged["rmse_2"]).pvalue)
                    p_mae = float(wilcoxon(merged["mae_1"], merged["mae_2"]).pvalue)
                    p_rmse_less = float(wilcoxon(merged["rmse_1"], merged["rmse_2"], alternative="less").pvalue)
                    p_mae_less = float(wilcoxon(merged["mae_1"], merged["mae_2"], alternative="less").pvalue)
                except Exception:
                    p_rmse, p_mae, p_rmse_less, p_mae_less = np.nan, np.nan, np.nan, np.nan
                rows.append({"dataset": dataset, "method_1": m1, "method_2": m2, "n_pairs": len(merged), "p_rmse_two_sided": p_rmse, "p_mae_two_sided": p_mae, "p_rmse_less": p_rmse_less, "p_mae_less": p_mae_less})
    return pd.DataFrame(rows)


def runtime_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (dataset, method), sub in df.groupby(["dataset", "method"]):
        uniq = sub[["dataset", "method", "seed", "train_time_sec"]].drop_duplicates()
        rows.append({
            "dataset": dataset,
            "method": method,
            "train_time_sec_mean": safe_mean(uniq["train_time_sec"]),
            "train_time_sec_std": safe_std(uniq["train_time_sec"]),
            "infer_time_sec_mean": safe_mean(sub["infer_time_sec"]),
            "infer_time_sec_std": safe_std(sub["infer_time_sec"]),
            "train_n_unique_models": len(uniq),
            "n_eval_runs": len(sub),
        })
    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser(description="Focused reviewer experiments for PCR-SAITS")
    p.add_argument("--core-script", type=str, default=None, help="Path to PCR-SAITS core suite script. Defaults to auto-detect in same folder.")
    p.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    p.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    p.add_argument("--mechanisms", type=str, default=",".join(DEFAULT_MECHANISMS))
    p.add_argument("--patterns", type=str, default=",".join(DEFAULT_PATTERNS))
    p.add_argument("--rates", type=str, default=",".join(str(x) for x in DEFAULT_RATES))
    p.add_argument("--seeds", type=str, default=",".join(str(x) for x in DEFAULT_SEEDS))
    p.add_argument("--window", type=int, default=48)
    p.add_argument("--eval-stride", type=int, default=24)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--saits-epochs", type=int, default=50)
    p.add_argument("--saits-batch-size", type=int, default=64)
    p.add_argument("--saits-patience", type=int, default=10)
    p.add_argument("--saits-d-model", type=int, default=64)
    p.add_argument("--saits-d-ffn", type=int, default=128)
    p.add_argument("--saits-heads", type=int, default=4)
    p.add_argument("--saits-layers", type=int, default=2)
    p.add_argument("--saits-dropout", type=float, default=0.1)
    p.add_argument("--pcr-epochs", type=int, default=30)
    p.add_argument("--pcr-batch-size", type=int, default=256)
    p.add_argument("--pcr-patience", type=int, default=8)
    p.add_argument("--pcr-lr", type=float, default=1e-3)
    p.add_argument("--pcr-weight-decay", type=float, default=1e-4)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    args.datasets = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    args.methods = [x.strip() for x in str(args.methods).split(",") if x.strip()]
    args.mechanisms = [str(x).upper().strip() for x in str(args.mechanisms).split(",") if x.strip()]
    args.patterns = [x.strip() for x in str(args.patterns).split(",") if x.strip()]
    args.rates = [float(x) for x in str(args.rates).split(",") if str(x).strip()]
    args.seeds = [int(x) for x in str(args.seeds).split(",") if str(x).strip()]
    return args


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    core_path = choose_core_script(script_dir, args.core_script)
    core = load_core_module(core_path)
    output_dir = Path(args.output_dir) if args.output_dir else (script_dir / f"pcr_reviewer_experiments_outputs_{now_stamp()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Using core script: {core_path}")
    print(f"[INFO] Output directory: {output_dir}")

    run_rows: List[Dict] = []

    for dataset_name in args.datasets:
        ds_path = resolve_dataset_path(script_dir, dataset_name)
        df, features = load_dataset(core, ds_path)
        values = df[features].to_numpy(dtype=float)
        split = core.chronological_split(values, ratios=(args.train_ratio, args.val_ratio, 1.0 - args.train_ratio - args.val_ratio))
        train_raw = split["train"]
        val_raw = split["val"]
        test_raw = split["test"]
        mean, std = core.fit_standardizer(train_raw)
        train_std = core.transform_values(train_raw, mean, std)
        val_std = core.transform_values(val_raw, mean, std)
        test_std = core.transform_values(test_raw, mean, std)
        fill_values = np.nanmean(train_std, axis=0)
        fill_values = np.where(np.isfinite(fill_values), fill_values, 0.0)
        ref_map = core.compute_ref_feature_map(train_std)
        print(f"[INFO] Dataset={dataset_name} rows={len(values)} features={len(features)}")

        for seed in args.seeds:
            core.set_seed(seed)
            train_windows, _ = core.build_windows(train_std, args.window, stride=1)
            val_windows, val_windows_ori = build_val_windows(core, val_std, args.window, args.eval_stride, seed * 1000 + 7)
            t0 = time.time()
            saits = core.SAITSWrapper(
                n_steps=args.window,
                n_features=len(features),
                epochs=args.saits_epochs,
                batch_size=args.saits_batch_size,
                patience=args.saits_patience,
                d_model=args.saits_d_model,
                d_ffn=args.saits_d_ffn,
                n_heads=args.saits_heads,
                n_layers=args.saits_layers,
                dropout=args.saits_dropout,
                verbose=not args.quiet,
            )
            print(f"[INFO] Training SAITS for dataset={dataset_name} seed={seed}...")
            saits.fit(train_windows, val_windows, val_windows_ori)
            saits_train_time = time.time() - t0

            trained_correctors: Dict[str, object] = {}
            train_times: Dict[str, float] = {"saits": saits_train_time}
            for method in args.methods:
                if method == "saits":
                    continue
                if method == "pcr_xgboost" and XGBRegressor is None:
                    print("[WARN] xgboost is not installed; skipping pcr_xgboost.")
                    continue
                corrector = create_corrector(method, features, saits, core, args)
                t1 = time.time()
                print(f"[INFO] Training {method} for dataset={dataset_name} seed={seed}...")
                corrector.fit(train_std, val_std, eval_stride=args.eval_stride, seed=seed)
                train_times[method] = time.time() - t1
                trained_correctors[method] = corrector

            for mechanism in args.mechanisms:
                for pattern in args.patterns:
                    for rate in args.rates:
                        mask_seed = build_mask_seed(seed, mechanism, pattern, rate)
                        artificial_mask, gap_len = core.generate_artificial_mask(test_std, mechanism, pattern, rate, mask_seed, ref_map)
                        masked_test = core.apply_mask(test_std, artificial_mask)
                        test_windows, test_starts = core.build_windows(masked_test, args.window, stride=args.eval_stride)
                        infer0 = time.time()
                        saits_windows = saits.impute(test_windows.astype(np.float32))
                        saits_full = core.reconstruct_from_windows(saits_windows, test_starts, len(test_std))
                        observed = np.isfinite(masked_test)
                        saits_full[observed] = masked_test[observed]
                        saits_infer_time = time.time() - infer0

                        scenario_methods = [m for m in args.methods if m == "saits" or m in trained_correctors]
                        for method in scenario_methods:
                            infer_start = time.time()
                            if method == "saits":
                                pred_std = saits_full
                                infer_time = saits_infer_time
                            else:
                                pred_std = trained_correctors[method].correct(masked_test, saits_full.copy(), artificial_mask, gap_len)
                                infer_time = time.time() - infer_start
                            pred_raw = core.inverse_transform_values(pred_std, mean, std)
                            truth_raw = test_raw
                            metrics = core.compute_metrics(truth_raw, pred_raw, artificial_mask)
                            point_mask = artificial_mask & (gap_len <= 1.0)
                            long_mask = artificial_mask & (gap_len >= 6.0)
                            point_rmse = core.compute_metrics(truth_raw, pred_raw, point_mask)["rmse"] if np.any(point_mask) else np.nan
                            long_rmse = core.compute_metrics(truth_raw, pred_raw, long_mask)["rmse"] if np.any(long_mask) else np.nan
                            grouped = compute_grouped_metrics(truth_raw, pred_raw, artificial_mask, features)
                            run_rows.append({
                                "dataset": dataset_name,
                                "seed": seed,
                                "mechanism": mechanism,
                                "pattern": pattern,
                                "rate": rate,
                                "method": method,
                                "rmse": metrics["rmse"],
                                "mae": metrics["mae"],
                                "pointwise_rmse": point_rmse,
                                "long_gap_rmse": long_rmse,
                                **{k: v for k, v in grouped.items() if k.endswith("_rmse")},
                                "train_time_sec": train_times[method],
                                "infer_time_sec": infer_time,
                                "train_mode": (getattr(trained_correctors.get(method), "train_mode", np.nan) if method != "saits" else np.nan),
                                "use_domain_tags": (getattr(trained_correctors.get(method), "use_domain_tags", np.nan) if method != "saits" else np.nan),
                            })
                            print(f"[DONE] dataset={dataset_name} seed={seed} method={method} scenario={mechanism}/{pattern}/rate={rate} | RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f}")

    run_df = pd.DataFrame(run_rows)
    run_df.to_csv(output_dir / "run_results.csv", index=False)
    summary_df = summarize_results(run_df, output_dir)
    pairwise_df = compute_pairwise_wilcoxon(run_df)
    if not pairwise_df.empty:
        pairwise_df.to_csv(output_dir / "pairwise_wilcoxon.csv", index=False)
    runtime_df = runtime_summary(run_df)
    if not runtime_df.empty:
        runtime_df.to_csv(output_dir / "runtime_summary.csv", index=False)
    with pd.ExcelWriter(output_dir / "results.xlsx") as writer:
        run_df.to_excel(writer, sheet_name="RunResults", index=False)
        if not summary_df.empty:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
        if not pairwise_df.empty:
            pairwise_df.to_excel(writer, sheet_name="PairwiseWilcoxon", index=False)
        if not runtime_df.empty:
            runtime_df.to_excel(writer, sheet_name="RuntimeSummary", index=False)
    metadata = {
        "created_at": now_stamp(),
        "core_script": str(core_path),
        "datasets": args.datasets,
        "methods": args.methods,
        "mechanisms": args.mechanisms,
        "patterns": args.patterns,
        "rates": args.rates,
        "seeds": args.seeds,
        "notes": {
            "pcr_mlp_mixed": "MLP residual corrector with mixed pointwise+block training holdout and domain tags",
            "pcr_mlp_mcar_only": "MLP residual corrector with MCAR-only training holdout and domain tags",
            "pcr_mlp_no_domain_tags": "MLP residual corrector with mixed holdout but no domain tag features",
            "pcr_linear": "Linear regression residual corrector with validation-tuned residual scaling",
            "pcr_ridge": "Ridge regression residual corrector",
            "pcr_xgboost": "XGBoost residual corrector with validation-based early stopping (if xgboost installed)",
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[DONE] Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
