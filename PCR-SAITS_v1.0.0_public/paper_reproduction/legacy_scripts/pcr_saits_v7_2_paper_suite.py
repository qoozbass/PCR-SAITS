#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCR-SAITS-v1.4 full protocol runner
----------------------------------
Clean restart implementation:
- SAITS is the only backbone/base imputer.
- PCR model is trained AFTER SAITS, on top of frozen SAITS outputs.
- PCR only learns a residual correction from stable hand-crafted features.
- Intended for full protocol evaluation with SAITS as reference and PCR-SAITS-v1.4 base as proposed method.

Default scope:
- Dataset: UCI Air Quality in same folder as this script
- Mechanisms: MCAR, MAR, MNAR
- Patterns: pointwise, block_6, block_12, block_24, block_48
- Rates: 0.1, 0.2, 0.3, 0.4, 0.5
- Seeds: 7, 21, 42, 123, 456
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception as e:
    raise RuntimeError("This script requires torch. Please install torch first.") from e

try:
    from pypots.imputation import SAITS, BRITS
except Exception:
    SAITS = None
    BRITS = None

# v7.2: CSDI (diffusion baseline, R4.3) — imported lazily/guarded like SAITS.
try:
    from pypots.imputation import CSDI
except Exception:
    CSDI = None

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

DEFAULT_METHODS = [
    "linear",
    "seasonal_naive24",
    "brits",
    "saits",
    "pcr_mlp_no_domain_tags",
]
POINTWISE = "pointwise"
BLOCK_PATTERNS = {"block_6": 6, "block_12": 12, "block_24": 24, "block_48": 48}
SUPPORTED_MECHANISMS = {"MCAR", "MAR", "MNAR"}
POLLUTANT_VARS = {"CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"}
METEO_VARS = {"T", "RH", "AH"}
EPS = 1e-8
_UNKNOWN_GROUP_WARNED = set()


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def safe_mean(values) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else float("nan")


def safe_std(values) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0


def robust_time_to_timedelta(series: pd.Series) -> pd.Series:
    if pd.api.types.is_timedelta64_dtype(series):
        return series
    if pd.api.types.is_datetime64_any_dtype(series):
        dt = pd.to_datetime(series, errors="coerce")
        return (
            pd.to_timedelta(dt.dt.hour.fillna(0), unit="h")
            + pd.to_timedelta(dt.dt.minute.fillna(0), unit="m")
            + pd.to_timedelta(dt.dt.second.fillna(0), unit="s")
        )
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_timedelta(pd.to_numeric(series, errors="coerce"), unit="D")

    s = series.astype(str).str.strip().str.replace(".", ":", regex=False)
    parsed = pd.to_datetime(s, errors="coerce")
    td = (
        pd.to_timedelta(parsed.dt.hour.fillna(0), unit="h")
        + pd.to_timedelta(parsed.dt.minute.fillna(0), unit="m")
        + pd.to_timedelta(parsed.dt.second.fillna(0), unit="s")
    )
    bad = parsed.isna()
    if bad.any():
        parts = s[bad].str.extract(r"(?P<h>\d{1,2}):(?P<m>\d{1,2})(?::(?P<sec>\d{1,2}))?")
        h = pd.to_numeric(parts["h"], errors="coerce").fillna(0)
        m = pd.to_numeric(parts["m"], errors="coerce").fillna(0)
        sec = pd.to_numeric(parts["sec"], errors="coerce").fillna(0)
        td.loc[bad] = (
            pd.to_timedelta(h, unit="h")
            + pd.to_timedelta(m, unit="m")
            + pd.to_timedelta(sec, unit="s")
        ).values
    return td


def find_dataset_file(script_dir: Path) -> Path:
    for name in ["AirQualityUCI.xlsx", "AirQualityUCI.xls", "AirQualityUCI.csv", "AirQualityUCI.data"]:
        p = script_dir / name
        if p.exists():
            return p
    files = list(script_dir.glob("AirQualityUCI*"))
    if files:
        return files[0]
    raise FileNotFoundError("Could not find AirQualityUCI dataset in the same folder as the script.")


def load_air_quality_dataset(path: Path):
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        try:
            df = pd.read_csv(path, sep=";", decimal=",")
        except Exception:
            df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    if not {"Date", "Time"}.issubset(df.columns):
        raise ValueError(f"Dataset must contain Date and Time. Found: {df.columns.tolist()}")
    df = df.dropna(axis=1, how="all").copy()
    for c in [c for c in df.columns if c not in {"Date", "Time"}]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] == -200, c] = np.nan
    date_part = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    time_part = robust_time_to_timedelta(df["Time"])
    df["timestamp"] = date_part.dt.normalize() + time_part
    raw_rows = len(df)
    valid_rows = df["timestamp"].notna().sum()
    unique_rows = df["timestamp"].nunique(dropna=True)
    print(f"[INFO] Timestamp diagnostics | raw={raw_rows} valid={valid_rows} unique={unique_rows}")
    df = df.dropna(subset=["timestamp"]).copy()
    dup = int(df["timestamp"].duplicated().sum())
    if dup > 0:
        print(f"[WARN] Duplicate timestamps found and removed: {dup}")
    df = df.drop_duplicates(subset=["timestamp"], keep="first").sort_values("timestamp").reset_index(drop=True)
    features = [c for c in df.columns if c not in {"Date", "Time", "timestamp"}]
    return df, features


def infer_feature_group(feature_name: str) -> str:
    if feature_name in METEO_VARS:
        return "meteorological"
    if feature_name.startswith("PT08."):
        return "sensor"
    if feature_name in POLLUTANT_VARS:
        return "pollutant"
    if feature_name not in _UNKNOWN_GROUP_WARNED:
        print(f"[WARN] Unknown feature group for '{feature_name}'. Falling back to 'sensor'.")
        _UNKNOWN_GROUP_WARNED.add(feature_name)
    return "sensor"


def group_id(feature_name: str) -> int:
    g = infer_feature_group(feature_name)
    return 0 if g == "pollutant" else 1 if g == "sensor" else 2


def chronological_split(values: np.ndarray, ratios=(0.70, 0.15, 0.15)):
    n = len(values)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    n_test = n - n_train - n_val
    return {
        "train": values[:n_train],
        "val": values[n_train:n_train+n_val],
        "test": values[n_train+n_val:n_train+n_val+n_test],
    }


def fit_standardizer(train_values: np.ndarray):
    mean = np.nanmean(train_values, axis=0)
    std = np.nanstd(train_values, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def transform_values(values: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (values - mean) / std


def inverse_transform_values(values: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return values * std + mean


def build_windows(values: np.ndarray, window: int, stride: int):
    n = len(values)
    if n < window:
        pad = np.full((window-n, values.shape[1]), np.nan)
        values = np.concatenate([values, pad], axis=0)
        n = len(values)
    starts = list(range(0, max(1, n - window + 1), stride))
    if starts[-1] != n - window:
        starts.append(n - window)
    windows = np.stack([values[s:s+window] for s in starts], axis=0)
    return windows, np.asarray(starts, dtype=int)


def reconstruct_from_windows(windows: np.ndarray, starts: np.ndarray, total_length: int):
    _, win, n_feat = windows.shape
    acc = np.zeros((total_length, n_feat), dtype=float)
    cnt = np.zeros((total_length, n_feat), dtype=float)
    for i, s in enumerate(starts):
        e = min(total_length, s + win)
        part = windows[i][:e-s]
        valid = np.isfinite(part)
        acc[s:e][valid] += part[valid]
        cnt[s:e][valid] += 1.0
    out = np.full((total_length, n_feat), np.nan)
    valid = cnt > 0
    out[valid] = acc[valid] / cnt[valid]
    return out


def create_output_dir(script_dir: Path, prefix: str, explicit_dir: Optional[str] = None):
    out = (script_dir / explicit_dir) if explicit_dir else (script_dir / f"{prefix}_{now_stamp()}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def checkpoint_key(seed: int, method: str, mechanism: str, pattern: str, rate: float) -> Tuple[int, str, str, str, float]:
    return (int(seed), str(method), str(mechanism), str(pattern), float(rate))



def is_case_scenario(args, seed, mechanism, pattern, rate):
    return (
        int(seed) == int(args.case_seed)
        and str(mechanism).upper() == str(args.case_mechanism).upper()
        and str(pattern) == str(args.case_pattern)
        and abs(float(rate) - float(args.case_rate)) < 1e-12
    )
def deterministic_scenario_seed(seed: int, mechanism: str, pattern: str, rate: float) -> int:
    key = f"{mechanism}|{pattern}|{rate:.6f}"
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
    return int(seed) * 1000 + int(round(rate * 100)) + h % 997


def append_csv_row(path: Path, row: Dict):
    df = pd.DataFrame([row])
    header = (not path.exists()) or path.stat().st_size == 0
    df.to_csv(path, mode="a", header=header, index=False)


def append_csv_rows(path: Path, rows: List[Dict]):
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = (not path.exists()) or path.stat().st_size == 0
    df.to_csv(path, mode="a", header=header, index=False)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray):
    valid = mask & np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        return {"rmse": float("nan"), "mae": float("nan")}
    err = y_pred[valid] - y_true[valid]
    return {"rmse": float(np.sqrt(np.mean(err**2))), "mae": float(np.mean(np.abs(err)))}


def compute_grouped_metrics(y_true, y_pred, mask, features):
    out = {}
    for g in ["pollutant", "sensor", "meteorological"]:
        idx = [i for i, f in enumerate(features) if infer_feature_group(f) == g]
        out[f"{g}_rmse"] = compute_metrics(y_true[:, idx], y_pred[:, idx], mask[:, idx])["rmse"] if idx else float("nan")
    return out


def _block_positions(length: int, gap: int, rate: float, rng: np.random.Generator, allowed: np.ndarray):
    target = int(round(np.sum(allowed) * rate))
    out = np.zeros(length, dtype=bool)
    if target <= 0:
        return out
    current, tries = 0, 0
    while current < target and tries < 100000:
        tries += 1
        s = int(rng.integers(0, max(1, length-gap+1)))
        e = min(length, s+gap)
        if not np.all(allowed[s:e]) or np.any(out[s:e]):
            continue
        out[s:e] = True
        current = int(out.sum())
    return out


def compute_ref_feature_map(train_std_values: np.ndarray) -> Dict[int, int]:
    df = pd.DataFrame(train_std_values)
    corr = df.corr(method="pearson", min_periods=50).abs().to_numpy(copy=True)
    n_features = corr.shape[0]
    ref_map: Dict[int, int] = {}
    for f in range(n_features):
        corr[f, f] = np.nan
        if np.all(np.isnan(corr[f])):
            ref_map[f] = (f + 1) % n_features
        else:
            ref_map[f] = int(np.nanargmax(corr[f]))
    return ref_map


def _compute_point_scores(X_std: np.ndarray, candidates: np.ndarray, mechanism: str, ref_map: Dict[int, int], rng: np.random.Generator) -> np.ndarray:
    scores = rng.random(len(candidates)) * 1e-6
    if mechanism == "MCAR":
        scores += rng.random(len(candidates))
        return scores
    n_features = X_std.shape[1]
    directions = rng.choice([-1.0, 1.0], size=n_features)
    times = candidates[:, 0]
    feats = candidates[:, 1]
    if mechanism == "MAR":
        ref_feats = np.array([ref_map[int(f)] for f in feats], dtype=int)
        base = X_std[times, ref_feats]
    else:
        base = X_std[times, feats]
    base = np.nan_to_num(base, nan=0.0)
    scores += directions[feats] * base
    scores += directions[feats] * base
    return scores


def _compute_block_score(X_std: np.ndarray, start: int, block_len: int, feat: int, mechanism: str, ref_map: Dict[int, int], direction: float, rng: np.random.Generator) -> float:
    if mechanism == "MCAR":
        return float(rng.random())
    if mechanism == "MAR":
        ref_feat = ref_map[feat]
        block_vals = X_std[start:start + block_len, ref_feat]
    else:
        block_vals = X_std[start:start + block_len, feat]
    base = np.nanmean(np.nan_to_num(block_vals, nan=0.0))
    return float(direction * base + rng.random() * 1e-6)


def generate_artificial_mask(X_std: np.ndarray, mechanism: str, pattern: str, rate: float, seed: int, ref_map: Dict[int, int], block_buffer: int = 1):
    if mechanism not in SUPPORTED_MECHANISMS:
        raise ValueError(f"Unsupported mechanism: {mechanism}")
    rng = np.random.default_rng(seed)
    observed = ~np.isnan(X_std)
    artificial_mask = np.zeros_like(observed, dtype=bool)
    gap_len = np.ones_like(X_std, dtype=float)
    total_observed = int(observed.sum())
    if total_observed == 0 or rate <= 0:
        return artificial_mask, gap_len
    target_missing = max(1, int(round(total_observed * rate)))
    block_len = 1 if pattern == POINTWISE else int(pattern.split("_")[1])
    if block_len == 1:
        candidates = np.argwhere(observed)
        if len(candidates) == 0:
            return artificial_mask, gap_len
        scores = _compute_point_scores(X_std, candidates, mechanism, ref_map, rng)
        order = np.argsort(scores)[::-1]
        selected = candidates[order[:min(target_missing, len(candidates))]]
        artificial_mask[selected[:, 0], selected[:, 1]] = True
        gap_len[selected[:, 0], selected[:, 1]] = 1.0
        return artificial_mask, gap_len

    T, F = X_std.shape
    directions = rng.choice([-1.0, 1.0], size=F)
    candidates: List[Tuple[float, int, int]] = []
    for feat in range(F):
        valid = observed[:, feat]
        if valid.sum() < block_len:
            continue
        for start in range(0, T - block_len + 1):
            if not valid[start:start + block_len].all():
                continue
            score = _compute_block_score(X_std, start, block_len, feat, mechanism, ref_map, float(directions[feat]), rng)
            candidates.append((score, start, feat))
    if not candidates:
        return artificial_mask, gap_len
    candidates.sort(key=lambda x: x[0], reverse=True)
    occupied = np.zeros_like(observed, dtype=bool)
    masked_cells = 0
    for _, start, feat in candidates:
        if masked_cells >= target_missing:
            break
        left = max(0, start - block_buffer)
        right = min(T, start + block_len + block_buffer)
        if occupied[left:right, feat].any():
            continue
        artificial_mask[start:start + block_len, feat] = True
        occupied[left:right, feat] = True
        gap_len[start:start + block_len, feat] = block_len
        masked_cells += block_len
    return artificial_mask, gap_len


def apply_mask(values: np.ndarray, artificial_mask: np.ndarray):
    out = values.copy()
    out[artificial_mask] = np.nan
    return out


def make_holdout_train_mask(
    values: np.ndarray,
    seed: int,
    holdout_ratio: float = 0.15,
    pointwise_fraction: float = 0.5,
    block_patterns: Tuple[int, ...] = (6, 12, 24, 48),
    block_buffer: int = 1,
):
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
            chosen_idx = rng.choice(len(candidates), size=min(point_target, len(candidates)), replace=False)
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
            chosen_idx = rng.choice(len(candidates), size=min(remaining, len(candidates)), replace=False)
            chosen = candidates[chosen_idx]
            mask[chosen[:, 0], chosen[:, 1]] = True
            gap_len[chosen[:, 0], chosen[:, 1]] = np.where(gap_len[chosen[:, 0], chosen[:, 1]] > 0, gap_len[chosen[:, 0], chosen[:, 1]], 1.0)

    return mask, gap_len


def _precompute_local_estimates(full_series: np.ndarray):
    arr = np.asarray(full_series, dtype=float)
    n_t, n_f = arr.shape
    local_vals = np.zeros((n_t, n_f), dtype=np.float32)
    local_ok = np.zeros((n_t, n_f), dtype=np.float32)
    for f in range(n_f):
        col = arr[:, f]
        finite = np.isfinite(col)
        prev_idx = np.full(n_t, -1, dtype=int)
        next_idx = np.full(n_t, -1, dtype=int)
        last = -1
        for t in range(n_t):
            prev_idx[t] = last
            if finite[t]:
                last = t
        last = -1
        for t in range(n_t - 1, -1, -1):
            next_idx[t] = last
            if finite[t]:
                last = t
        for t in range(n_t):
            li = prev_idx[t]
            ri = next_idx[t]
            if li >= 0 and ri >= 0 and ri != li:
                w = (t - li) / float(ri - li)
                local_vals[t, f] = float((1.0 - w) * col[li] + w * col[ri])
                local_ok[t, f] = 1.0
            elif li >= 0:
                local_vals[t, f] = float(col[li])
                local_ok[t, f] = 1.0
            elif ri >= 0:
                local_vals[t, f] = float(col[ri])
                local_ok[t, f] = 1.0
    return local_vals, local_ok


def local_estimate(full_series: np.ndarray, t: int, f: int):
    local_vals, local_ok = _precompute_local_estimates(full_series)
    return float(local_vals[t, f]), float(local_ok[t, f])


def seasonal_estimate(full_series: np.ndarray, t: int, f: int, lags=(24, 48, 168)):
    vals = []
    n = len(full_series)
    for lag in lags:
        left = t - lag
        right = t + lag
        if left >= 0 and np.isfinite(full_series[left, f]):
            vals.append(float(full_series[left, f]))
        if right < n and np.isfinite(full_series[right, f]):
            vals.append(float(full_series[right, f]))
    return (float(np.mean(vals)), 1.0) if vals else (0.0, 0.0)


class SAITSWrapper:
    checkpoint_ext = ".pypots"

    def __init__(self, n_steps, n_features, epochs, batch_size, patience, d_model, d_ffn, n_heads, n_layers, dropout, verbose=True):
        if SAITS is None:
            raise RuntimeError("PyPOTS is not installed. Please install pypots.")
        self.model = SAITS(
            n_steps=n_steps,
            n_features=n_features,
            n_layers=n_layers,
            d_model=d_model,
            d_ffn=d_ffn,
            n_heads=n_heads,
            d_k=d_model // max(1, n_heads),
            d_v=d_model // max(1, n_heads),
            dropout=dropout,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            num_workers=0,
            device=choose_device(),
        )
        self.verbose = verbose

    def fit(self, train_windows, val_windows, val_windows_ori=None):
        train_set = {"X": train_windows.astype(np.float32)}
        if val_windows_ori is None:
            val_windows_ori = val_windows
        val_set = {
            "X": val_windows.astype(np.float32),
            "X_ori": val_windows_ori.astype(np.float32),
        }
        self.model.fit(train_set, val_set)

    def impute(self, windows):
        out = self.model.impute({"X": windows.astype(np.float32)})
        return np.asarray(out, dtype=float)

    def save(self, path: Path):
        self.model.save(str(path))

    @classmethod
    def load_from_checkpoint(cls, path: Path, **kwargs):
        obj = cls(**kwargs)
        obj.model.load(str(path))
        return obj


def _linear_fill_1d(values: np.ndarray, default_fill: float) -> np.ndarray:
    s = pd.Series(values.copy())
    filled = s.interpolate(method="linear", limit_direction="both")
    filled = filled.fillna(default_fill)
    return filled.to_numpy(dtype=float)


def impute_linear_windows(X_windows: np.ndarray, fill_values: np.ndarray) -> np.ndarray:
    X_imp = X_windows.copy()
    n_samples, _, n_features = X_imp.shape
    for i in range(n_samples):
        for f in range(n_features):
            X_imp[i, :, f] = _linear_fill_1d(X_imp[i, :, f], float(fill_values[f]))
    return X_imp


def _seasonal_fill_1d(values: np.ndarray, lag: int, default_fill: float) -> np.ndarray:
    arr = values.copy()
    n = len(arr)
    missing = np.isnan(arr)
    if not missing.any():
        return arr
    for t in np.where(missing)[0]:
        candidates = []
        step = 1
        while True:
            found_any = False
            left = t - step * lag
            right = t + step * lag
            if left >= 0 and not np.isnan(arr[left]):
                candidates.append(arr[left]); found_any = True
            if right < n and not np.isnan(arr[right]):
                candidates.append(arr[right]); found_any = True
            if found_any or (left < 0 and right >= n):
                break
            step += 1
        if candidates:
            arr[t] = float(np.mean(candidates))
    if np.isnan(arr).any():
        arr = _linear_fill_1d(arr, default_fill)
    return arr


def impute_seasonal_naive24_windows(X_windows: np.ndarray, fill_values: np.ndarray, lag: int = 24) -> np.ndarray:
    X_imp = X_windows.copy()
    n_samples, _, n_features = X_imp.shape
    for i in range(n_samples):
        for f in range(n_features):
            X_imp[i, :, f] = _seasonal_fill_1d(X_imp[i, :, f], lag=lag, default_fill=float(fill_values[f]))
    return X_imp


class BRITSWrapper:
    checkpoint_ext = ".pypots"

    def __init__(self, n_steps, n_features, epochs, batch_size, patience, rnn_hidden_size, verbose=True):
        if BRITS is None:
            raise RuntimeError("PyPOTS BRITS is not installed. Please install pypots.")
        self.model = BRITS(
            n_steps=n_steps,
            n_features=n_features,
            rnn_hidden_size=rnn_hidden_size,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            num_workers=0,
            device=choose_device(),
            verbose=verbose,
            saving_path=None,
        )
        self.verbose = verbose

    def fit(self, train_windows, val_windows=None, val_windows_ori=None):
        train_set = {"X": train_windows.astype(np.float32)}
        if val_windows is not None and val_windows_ori is not None:
            val_set = {"X": val_windows.astype(np.float32), "X_ori": val_windows_ori.astype(np.float32)}
            try:
                self.model.fit(train_set, val_set)
                return
            except Exception as exc:
                print(f"[WARN] BRITS fit with val_set failed: {exc}. Falling back to train only.", flush=True)
        self.model.fit(train_set)

    def impute(self, windows):
        out = self.model.impute({"X": windows.astype(np.float32)})
        return np.asarray(out, dtype=float)

    def save(self, path: Path):
        self.model.save(str(path))

    @classmethod
    def load_from_checkpoint(cls, path: Path, **kwargs):
        obj = cls(**kwargs)
        obj.model.load(str(path))
        return obj


class CSDIWrapper:
    """v7.2: adapter around PyPOTS CSDI (diffusion imputer, R4.3) with the same
    fit/impute/save/load interface as SAITSWrapper and BRITSWrapper, so it drops
    into the existing method dispatch. CSDI is probabilistic; impute() averages
    over n_sampling_times diffusion samples to yield a deterministic point
    estimate, consistent with how the other imputers are scored."""

    checkpoint_ext = ".pypots"

    def __init__(self, n_steps, n_features, epochs, batch_size, patience,
                 n_layers=4, n_heads=8, n_channels=64,
                 d_time_embedding=128, d_feature_embedding=16,
                 d_diffusion_embedding=128, n_diffusion_steps=50,
                 n_sampling_times=10, verbose=True):
        if CSDI is None:
            raise RuntimeError("PyPOTS CSDI is not installed. Please install "
                               "pypots (pip install pypots).")
        self.n_sampling_times = int(n_sampling_times)
        self.verbose = verbose
        self.model = CSDI(
            n_steps=n_steps,
            n_features=n_features,
            n_layers=n_layers,
            n_heads=n_heads,
            n_channels=n_channels,
            d_time_embedding=d_time_embedding,
            d_feature_embedding=d_feature_embedding,
            d_diffusion_embedding=d_diffusion_embedding,
            n_diffusion_steps=n_diffusion_steps,
            target_strategy="random",
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            num_workers=0,
            device=choose_device(),
            saving_path=None,
            verbose=verbose,
        )

    def fit(self, train_windows, val_windows=None, val_windows_ori=None):
        train_set = {"X": train_windows.astype(np.float32)}
        if val_windows is not None and val_windows_ori is not None:
            val_set = {"X": val_windows.astype(np.float32),
                       "X_ori": val_windows_ori.astype(np.float32)}
            try:
                self.model.fit(train_set, val_set)
                return
            except Exception as exc:
                print(f"[WARN] CSDI fit with val_set failed: {exc}. Falling "
                      f"back to train only.", flush=True)
        self.model.fit(train_set)

    def impute(self, windows):
        out = self.model.impute({"X": windows.astype(np.float32)},
                                n_sampling_times=self.n_sampling_times)
        arr = np.asarray(out, dtype=float)
        # CSDI returns (n_samples, n_sampling_times, n_steps, n_features) when
        # multiple samples are requested; average over the sampling axis.
        if arr.ndim == 4:
            arr = arr.mean(axis=1)
        return arr

    def save(self, path: Path):
        self.model.save(str(path))

    @classmethod
    def load_from_checkpoint(cls, path: Path, **kwargs):
        obj = cls(**kwargs)
        obj.model.load(str(path))
        return obj


class PCRExampleDataset(Dataset):
    def __init__(self, features, targets, base_errors, easy_mask):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.base_errors = torch.tensor(base_errors, dtype=torch.float32)
        self.easy_mask = torch.tensor(easy_mask.astype(np.float32), dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.base_errors[idx], self.easy_mask[idx]


class PCRResidualNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.delta_head = nn.Linear(hidden_dim, 1)
        self.mask_head = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.backbone[0].weight, gain=0.5)
        nn.init.zeros_(self.backbone[0].bias)
        nn.init.xavier_uniform_(self.backbone[2].weight, gain=0.5)
        nn.init.zeros_(self.backbone[2].bias)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.mask_head.weight)
        nn.init.constant_(self.mask_head.bias, -1.0)

    def forward(self, x, direct_residual=False):
        h = self.backbone(x)
        delta = torch.tanh(self.delta_head(h)) * 4.0
        corr_mask = torch.ones_like(delta) if direct_residual else torch.sigmoid(self.mask_head(h))
        return delta, corr_mask


class PCRSAITSV1CleanWrapper:
    checkpoint_ext = ".pt"

    def __init__(
        self,
        variant,
        n_steps,
        learning_rate,
        weight_decay,
        epochs,
        batch_size,
        patience,
        preserve_loss_weight,
        rel_loss_weight,
        sparse_loss_weight,
        base_model,
        feature_names,
        base_impute_stride=None,
        verbose=True,
    ):
        self.variant = variant
        self.n_steps = int(n_steps)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.preserve_loss_weight = preserve_loss_weight
        self.rel_loss_weight = rel_loss_weight
        self.sparse_loss_weight = sparse_loss_weight
        self.base_model = base_model
        self.feature_names = list(feature_names)
        self.base_impute_stride = int(base_impute_stride) if base_impute_stride is not None else int(n_steps)
        self.verbose = verbose
        self.device = choose_device()
        self.use_rel_loss = variant == "pcrsaitsv14_with_rel_loss"
        self.use_preserve_loss = variant == "pcrsaitsv14_with_preserve_loss"
        self.use_sparse_loss = variant == "pcrsaitsv14_with_sparse_loss"
        # v7.2: match by suffix so the BRITS backbone variant
        # 'pcr_brits_no_seasonal_branch' gets the SAME no-seasonal config as the
        # proposed SAITS 'pcrsaitsv14_no_seasonal_branch'. This makes E6 a clean
        # backbone swap (SAITS->BRITS) with everything else identical.
        self.use_seasonal_branch = not str(variant).endswith("no_seasonal_branch")
        self.use_local_branch = variant != "pcrsaitsv14_no_local_branch"
        # v7.2: a variant whose name ends with '_no_domain_tags' omits the two
        # domain-tag features (8-dim) — this is the SAITS ablation
        # 'pcr_mlp_no_domain_tags'. The proposed model and the E6 BRITS variant
        # ('pcrsaitsv14_brits_no_seasonal_branch') do NOT end with that suffix,
        # so both keep domain tags (10-dim). E6 is therefore a clean backbone
        # swap of the proposed config (no-seasonal, WITH domain tags), differing
        # from the SAITS proposed model only in the backbone (SAITS -> BRITS).
        self.use_domain_tags = not str(variant).endswith("_no_domain_tags")
        self.direct_residual = variant != "pcrsaitsv14_masked_residual"
        self.input_dim = 10 if self.use_domain_tags else 8
        self.model = PCRResidualNet(self.input_dim).to(self.device)

    def _build_examples(self, original_values, masked_values, base_imputed_values, target_mask, gap_len):
        feats, tgts, base_errs, easy_mask = [], [], [], []
        n_t, n_f = original_values.shape
        local_vals_all, local_ok_all = _precompute_local_estimates(masked_values)
        for t in range(n_t):
            for f in range(n_f):
                if not target_mask[t, f]:
                    continue
                base = float(base_imputed_values[t, f])
                true = float(original_values[t, f])
                local_val = float(local_vals_all[t, f])
                local_ok = float(local_ok_all[t, f])
                seasonal_val, seasonal_ok = seasonal_estimate(masked_values, t, f)
                if not self.use_local_branch:
                    local_val, local_ok = 0.0, 0.0
                if not self.use_seasonal_branch:
                    seasonal_val, seasonal_ok = 0.0, 0.0
                gl = float(gap_len[t, f])
                gln = min(gl / 48.0, 1.0)
                feat_group = group_id(self.feature_names[f]) if self.use_domain_tags else -1
                row = [
                    base,
                    local_val,
                    seasonal_val,
                    base-local_val,
                    base-seasonal_val,
                    local_ok,
                    seasonal_ok,
                    gln,
                ]
                if self.use_domain_tags:
                    row.extend([
                        float(feat_group == 0),
                        float(feat_group == 1),
                    ])
                feats.append(np.array(row, dtype=np.float32))
                tgts.append(np.float32(true - base))
                base_errs.append(np.float32(abs(true - base)))
                easy_mask.append(np.float32(1.0 if gl <= 1.0 else 0.0))
        if not feats:
            return np.zeros((0, self.input_dim), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        return np.stack(feats), np.asarray(tgts, dtype=np.float32), np.asarray(base_errs, dtype=np.float32), np.asarray(easy_mask, dtype=np.float32)

    def _loss(self, delta, corr_mask, target, base_err, easy_mask):
        pred_residual = corr_mask * delta
        corrected_err = torch.abs(pred_residual - target)
        rec = F.smooth_l1_loss(pred_residual, target)
        preserve = (easy_mask * torch.abs(pred_residual)).mean()
        rel = torch.relu(corrected_err - base_err).mean()
        sparse = torch.abs(corr_mask).mean()
        total = rec
        if self.use_preserve_loss:
            total = total + self.preserve_loss_weight * preserve
        if self.use_rel_loss:
            total = total + self.rel_loss_weight * rel
        if self.use_sparse_loss:
            total = total + self.sparse_loss_weight * sparse
        return total

    def _impute_full_series_with_base(self, masked_values: np.ndarray):
        stride = max(1, int(self.base_impute_stride))
        windows, starts = build_windows(masked_values, self.n_steps, stride=stride)
        imputed_windows = self.base_model.impute(windows.astype(np.float32))
        return reconstruct_from_windows(imputed_windows, starts, len(masked_values))

    def fit(self, train_values, train_holdout_mask, train_gap_len, val_values, val_holdout_mask, val_gap_len):
        train_masked = apply_mask(train_values, train_holdout_mask)
        val_masked = apply_mask(val_values, val_holdout_mask)
        train_base = self._impute_full_series_with_base(train_masked)
        val_base = self._impute_full_series_with_base(val_masked)
        train_X, train_y, train_base_err, train_easy = self._build_examples(train_values, train_masked, train_base, train_holdout_mask, train_gap_len)
        val_X, val_y, val_base_err, val_easy = self._build_examples(val_values, val_masked, val_base, val_holdout_mask, val_gap_len)
        if len(train_X) == 0 or len(val_X) == 0:
            raise RuntimeError("PCR training data is empty. Check mask generation.")
        train_loader = DataLoader(PCRExampleDataset(train_X, train_y, train_base_err, train_easy), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(PCRExampleDataset(val_X, val_y, val_base_err, val_easy), batch_size=self.batch_size, shuffle=False)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        best_state, best_loss, bad_epochs = None, float("inf"), 0
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_sum, train_count = 0.0, 0
            for X, y, base_err, easy_mask in train_loader:
                X = X.to(self.device)
                y = y.to(self.device).unsqueeze(-1)
                base_err = base_err.to(self.device).unsqueeze(-1)
                easy_mask = easy_mask.to(self.device).unsqueeze(-1)
                opt.zero_grad(set_to_none=True)
                delta, corr_mask = self.model(X, direct_residual=self.direct_residual)
                loss = self._loss(delta, corr_mask, y, base_err, easy_mask)
                if not torch.isfinite(loss):
                    if self.verbose:
                        print(f"[WARN] Non-finite PCR loss in {self.variant}; skipping batch.")
                    opt.zero_grad(set_to_none=True)
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                opt.step()
                train_sum += float(loss.detach().cpu()) * len(X)
                train_count += len(X)
            self.model.eval()
            val_sum, val_count = 0.0, 0
            with torch.no_grad():
                for X, y, base_err, easy_mask in val_loader:
                    X = X.to(self.device)
                    y = y.to(self.device).unsqueeze(-1)
                    base_err = base_err.to(self.device).unsqueeze(-1)
                    easy_mask = easy_mask.to(self.device).unsqueeze(-1)
                    delta, corr_mask = self.model(X, direct_residual=self.direct_residual)
                    loss = self._loss(delta, corr_mask, y, base_err, easy_mask)
                    if not torch.isfinite(loss):
                        if self.verbose:
                            print(f"[WARN] Non-finite PCR validation loss in {self.variant}; skipping batch.")
                        continue
                    val_sum += float(loss.detach().cpu()) * len(X)
                    val_count += len(X)
            train_epoch_loss = train_sum / train_count if train_count > 0 else float('inf')
            val_epoch_loss = val_sum / val_count if val_count > 0 else float('inf')
            # debug metrics to verify the correction path is actually active
            self.model.eval()
            with torch.no_grad():
                sample_X = torch.tensor(val_X[: min(4096, len(val_X))], dtype=torch.float32, device=self.device)
                d_dbg, m_dbg = self.model(sample_X, direct_residual=self.direct_residual)
                delta_abs_mean = float(torch.abs(d_dbg).mean().cpu())
                mask_mean = float(m_dbg.mean().cpu())
                applied_delta_mean = float(torch.abs(d_dbg * m_dbg).mean().cpu())
            if self.verbose:
                print(f"[INFO] {self.variant} epoch={epoch}/{self.epochs} train_loss={train_epoch_loss:.6f} val_loss={val_epoch_loss:.6f} delta_abs_mean={delta_abs_mean:.6f} mask_mean={mask_mean:.6f} applied_delta_mean={applied_delta_mean:.6f}")
            if val_epoch_loss + 1e-8 < best_loss:
                best_loss = val_epoch_loss
                best_state = copy.deepcopy(self.model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    if self.verbose:
                        print(f"[INFO] {self.variant} early stopping at epoch {epoch}")
                    break
        if best_state is None:
            raise RuntimeError(f"No valid state found for {self.variant}.")
        self.model.load_state_dict(best_state)

    def correct(self, original_masked_values, base_imputed_values, artificial_mask, gap_len):
        corrected = base_imputed_values.copy()
        rows, coords = [], []
        n_t, n_f = corrected.shape
        local_vals_all, local_ok_all = _precompute_local_estimates(original_masked_values)
        for t in range(n_t):
            for f in range(n_f):
                if not artificial_mask[t, f]:
                    continue
                base = float(base_imputed_values[t, f])
                local_val = float(local_vals_all[t, f])
                local_ok = float(local_ok_all[t, f])
                seasonal_val, seasonal_ok = seasonal_estimate(original_masked_values, t, f)
                if not self.use_local_branch:
                    local_val, local_ok = 0.0, 0.0
                if not self.use_seasonal_branch:
                    seasonal_val, seasonal_ok = 0.0, 0.0
                gl = float(gap_len[t, f])
                gln = min(gl / 48.0, 1.0)
                feat_group = group_id(self.feature_names[f]) if self.use_domain_tags else -1
                row = [
                    base,
                    local_val,
                    seasonal_val,
                    base-local_val,
                    base-seasonal_val,
                    local_ok,
                    seasonal_ok,
                    gln,
                ]
                if self.use_domain_tags:
                    row.extend([
                        float(feat_group == 0),
                        float(feat_group == 1),
                    ])
                rows.append(np.array(row, dtype=np.float32))
                coords.append((t, f))
        if rows:
            X = torch.tensor(np.stack(rows), dtype=torch.float32, device=self.device)
            self.model.eval()
            with torch.no_grad():
                delta, corr_mask = self.model(X, direct_residual=self.direct_residual)
                residual = (delta.squeeze(-1) * corr_mask.squeeze(-1)).cpu().numpy()
            for (t, f), r in zip(coords, residual):
                corrected[t, f] = corrected[t, f] + float(r)
        observed = np.isfinite(original_masked_values)
        corrected[observed] = original_masked_values[observed]
        return corrected

    def save(self, path: Path):
        torch.save({"state_dict": self.model.state_dict(), "variant": self.variant}, path)

    @classmethod
    def load_from_checkpoint(cls, path: Path, **kwargs):
        obj = cls(**kwargs)
        state = torch.load(path, map_location=obj.device)
        obj.model.load_state_dict(state["state_dict"])
        obj.model.eval()
        return obj


def save_plot_rate(df: pd.DataFrame, output_dir: Path, metric: str = "rmse"):
    if plt is None or df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, sub in df.groupby("method"):
        sub = sub.sort_values("rate")
        ax.plot(sub["rate"], sub[metric], marker="o", label=method)
    ax.set_xlabel("Missing rate")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} vs Missing Rate")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"{metric}_vs_rate.png", dpi=200)
    plt.close(fig)


def save_feature_heatmap(feature_df: pd.DataFrame, output_dir: Path):
    if plt is None or feature_df.empty:
        return
    pivot = feature_df.pivot_table(index="feature", columns="method", values="rmse", aggfunc="mean")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Per-feature RMSE")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "per_feature_rmse_heatmap.png", dpi=200)
    plt.close(fig)


def save_grouped_plot(results_df: pd.DataFrame, output_dir: Path):
    if plt is None or results_df.empty:
        return
    grp = results_df.groupby("method")[["pollutant_rmse", "sensor_rmse", "meteorological_rmse"]].mean().sort_index()
    if grp.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(grp.index))
    width = 0.25
    ax.bar(x - width, grp["pollutant_rmse"], width, label="Pollutant")
    ax.bar(x, grp["sensor_rmse"], width, label="Sensor")
    ax.bar(x + width, grp["meteorological_rmse"], width, label="Meteorological")
    ax.set_xticks(x)
    ax.set_xticklabels(grp.index, rotation=45, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("Grouped RMSE by Method")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "grouped_rmse_by_method.png", dpi=200)
    plt.close(fig)


def save_pattern_improvement_plot(results_df: pd.DataFrame, output_dir: Path, reference_method: str):
    if plt is None or results_df.empty or reference_method not in set(results_df["method"]):
        return
    ref = results_df[results_df["method"] == reference_method]
    rows = []
    for method in sorted(set(results_df["method"]) - {reference_method}):
        cur = results_df[results_df["method"] == method]
        merged = cur[["seed", "mechanism", "pattern", "rate", "rmse"]].merge(
            ref[["seed", "mechanism", "pattern", "rate", "rmse"]].rename(columns={"rmse": "ref_rmse"}),
            on=["seed", "mechanism", "pattern", "rate"], how="inner")
        if merged.empty:
            continue
        merged["improve"] = (merged["ref_rmse"] - merged["rmse"]) / (merged["ref_rmse"] + EPS) * 100.0
        pat = merged.groupby("pattern")["improve"].mean().reset_index()
        pat["method"] = method
        rows.append(pat)
    if not rows:
        return
    df = pd.concat(rows, ignore_index=True)
    pivot = df.pivot(index="pattern", columns="method", values="improve").fillna(0.0)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(pivot.index))
    width = max(0.1, 0.8 / max(1, len(pivot.columns)))
    for i, method in enumerate(pivot.columns):
        ax.bar(x + (i - (len(pivot.columns)-1)/2)*width, pivot[method].values, width, label=method)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel("RMSE improvement vs reference (%)")
    ax.set_title("Pattern-wise Improvement vs Reference")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "pattern_improvement_vs_reference.png", dpi=200)
    plt.close(fig)


def save_case_study_plot(case_payload: Dict, output_dir: Path):
    if plt is None:
        return
    if not case_payload:
        print("[WARN] Case study plot was requested but no matching case payload was found.")
        return
    truth = case_payload.get("truth")
    masked = case_payload.get("masked")
    pred_ref = case_payload.get("pred_ref")
    pred_prop = case_payload.get("pred_prop")
    feature_name = case_payload.get("feature_name", "feature")
    title = case_payload.get("title", "Case Study")
    if truth is None or masked is None or pred_ref is None or pred_prop is None:
        print("[WARN] Case study payload is incomplete. Skipping case study plot.")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(truth))
    ax.plot(x, truth, label="Ground Truth")
    ax.plot(x, masked, label="Masked Input")
    ax.plot(x, pred_ref, label=case_payload.get("reference_method", "reference"))
    ax.plot(x, pred_prop, label=case_payload.get("proposed_method", "proposed"))
    ax.set_title(f"{title} | {feature_name}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "case_study_plot.png", dpi=200)
    plt.close(fig)


def compute_wilcoxon(results_df: pd.DataFrame, reference_method: str):
    if wilcoxon is None or results_df.empty or reference_method not in set(results_df["method"]):
        return pd.DataFrame()
    rows = []
    ref = results_df[results_df["method"] == reference_method][["seed", "mechanism", "pattern", "rate", "rmse", "mae"]].rename(columns={"rmse": "ref_rmse", "mae": "ref_mae"})
    for method in sorted(set(results_df["method"]) - {reference_method}):
        cur = results_df[results_df["method"] == method][["seed", "mechanism", "pattern", "rate", "rmse", "mae"]]
        merged = cur.merge(ref, on=["seed", "mechanism", "pattern", "rate"], how="inner")
        if len(merged) < 2:
            continue
        try:
            p_rmse_two = wilcoxon(merged["rmse"], merged["ref_rmse"], alternative="two-sided").pvalue
            p_mae_two = wilcoxon(merged["mae"], merged["ref_mae"], alternative="two-sided").pvalue
            p_rmse = wilcoxon(merged["rmse"], merged["ref_rmse"], alternative="less").pvalue
            p_mae = wilcoxon(merged["mae"], merged["ref_mae"], alternative="less").pvalue
        except Exception:
            p_rmse = np.nan
            p_mae = np.nan
            p_rmse_two = np.nan
            p_mae_two = np.nan
        rows.append({"method": method, "reference_method": reference_method, "wilcoxon_p_rmse": p_rmse, "wilcoxon_p_mae": p_mae, "wilcoxon_p_rmse_two_sided": p_rmse_two, "wilcoxon_p_mae_two_sided": p_mae_two})
    return pd.DataFrame(rows)



def save_excel(output_dir: Path, dfs: Dict[str, pd.DataFrame]):
    with pd.ExcelWriter(output_dir / "results.xlsx", engine="openpyxl") as writer:
        for name, df in dfs.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)


def run_experiment(args):
    script_dir = Path(args.script_dir).resolve() if args.script_dir else Path(__file__).resolve().parent
    dataset_path = find_dataset_file(script_dir)
    df, feature_names = load_air_quality_dataset(dataset_path)
    print(f"[INFO] Dataset: {dataset_path}")
    output_dir = create_output_dir(script_dir, args.output_prefix, args.output_dir)
    print(f"[INFO] Output directory: {output_dir}")

    values = df[feature_names].to_numpy(dtype=float)
    print(f"[INFO] Rows: {len(values):,} | Features: {len(feature_names)}")
    print(f"[INFO] Features: {feature_names}")
    splits = chronological_split(values, (args.train_ratio, args.val_ratio, args.test_ratio))
    train_raw, val_raw, test_raw = splits["train"], splits["val"], splits["test"]
    print(f"[INFO] Split sizes | train={len(train_raw):,} val={len(val_raw):,} test={len(test_raw):,}")

    mean, std = fit_standardizer(train_raw)
    train_std = transform_values(train_raw, mean, std)
    val_std = transform_values(val_raw, mean, std)
    test_std = transform_values(test_raw, mean, std)
    fill_values_std = np.nanmean(train_std, axis=0)
    fill_values_std = np.where(np.isfinite(fill_values_std), fill_values_std, 0.0)
    ref_map = compute_ref_feature_map(train_std)

    model_dir = output_dir / "model_checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    run_ckpt_path = output_dir / "checkpoint_results.csv"
    feat_ckpt_path = output_dir / "checkpoint_per_feature_results.csv"
    if args.reset_checkpoints:
        for p in [run_ckpt_path, feat_ckpt_path]:
            if p.exists():
                p.unlink()

    results_rows, feature_rows = [], []
    done_keys = set()
    if run_ckpt_path.exists():
        existing_runs = pd.read_csv(run_ckpt_path)
        if not existing_runs.empty:
            results_rows = existing_runs.to_dict("records")
            for _, r in existing_runs.iterrows():
                done_keys.add(checkpoint_key(r["seed"], r["method"], r["mechanism"], r["pattern"], r["rate"]))
    if feat_ckpt_path.exists():
        existing_feat = pd.read_csv(feat_ckpt_path)
        if not existing_feat.empty:
            feature_rows = existing_feat.to_dict("records")

    test_windows, test_starts = build_windows(test_std, window=args.window, stride=args.eval_stride)
    total_runs = len(args.seeds) * len(args.mechanisms) * len(args.patterns) * len(args.rates) * len(args.methods)
    completed = len(done_keys)
    case_payload = {}

    def pending_for(seed, method):
        for mech in args.mechanisms:
            for pat in args.patterns:
                for rate in args.rates:
                    if checkpoint_key(seed, method, mech, pat, rate) not in done_keys:
                        return True
        return False

    for seed in args.seeds:
        print("=" * 100)
        print(f"[INFO] Seed={seed}")
        set_seed(seed)
        train_windows, _ = build_windows(train_std, window=args.window, stride=args.train_stride)
        val_art_mask, _ = generate_artificial_mask(val_std, mechanism=args.val_mechanism, pattern=args.val_pattern, rate=args.val_rate, seed=seed + 991, ref_map=ref_map)
        val_masked = apply_mask(val_std, val_art_mask)
        val_windows, _ = build_windows(val_masked, window=args.window, stride=args.eval_stride)
        val_windows_ori, _ = build_windows(val_std, window=args.window, stride=args.eval_stride)

        trained_models = {}
        train_holdout_mask, train_holdout_gap = make_holdout_train_mask(train_std, seed=seed + 100)
        val_holdout_mask, val_holdout_gap = make_holdout_train_mask(val_std, seed=seed + 200)

        for method in args.methods:
            if method in {"linear", "seasonal_naive24"}:
                continue
            if not pending_for(seed, method):
                print(f"[SKIP] seed={seed} method={method} | all runs completed from checkpoint")
                continue
            if method == "saits":
                ckpt = model_dir / f"saits_seed{seed}{SAITSWrapper.checkpoint_ext}"
                kwargs = dict(n_steps=args.window, n_features=len(feature_names), epochs=args.saits_epochs, batch_size=args.saits_batch_size,
                              patience=args.saits_patience, d_model=args.saits_d_model, d_ffn=args.saits_d_ffn, n_heads=args.saits_heads,
                              n_layers=args.saits_layers, dropout=args.saits_dropout, verbose=not args.quiet)
                if ckpt.exists() and not args.retrain_models:
                    try:
                        wrapper = SAITSWrapper.load_from_checkpoint(ckpt, **kwargs)
                        train_time = 0.0
                        print(f"[INFO] Loaded SAITS checkpoint: {ckpt}")
                    except Exception as exc:
                        print(f"[WARN] Failed to load SAITS checkpoint: {exc}. Retraining.")
                        wrapper = SAITSWrapper(**kwargs)
                        print(f"[INFO] Training SAITS for seed={seed}...", flush=True)
                        t0=time.time(); wrapper.fit(train_windows, val_windows, val_windows_ori); train_time=time.time()-t0
                        wrapper.save(ckpt); print(f"[INFO] Saved saits checkpoint: {ckpt}")
                else:
                    wrapper = SAITSWrapper(**kwargs)
                    print(f"[INFO] Training SAITS for seed={seed}...", flush=True)
                    t0=time.time(); wrapper.fit(train_windows, val_windows, val_windows_ori); train_time=time.time()-t0
                    wrapper.save(ckpt); print(f"[INFO] Saved saits checkpoint: {ckpt}")
                trained_models[method]=(wrapper, train_time)
            elif method == "brits":
                ckpt = model_dir / f"brits_seed{seed}{BRITSWrapper.checkpoint_ext}"
                kwargs = dict(n_steps=args.window, n_features=len(feature_names), epochs=args.brits_epochs, batch_size=args.brits_batch_size,
                              patience=args.brits_patience, rnn_hidden_size=args.brits_hidden_size, verbose=not args.quiet)
                if ckpt.exists() and not args.retrain_models:
                    try:
                        wrapper = BRITSWrapper.load_from_checkpoint(ckpt, **kwargs)
                        train_time=0.0
                        print(f"[INFO] Loaded BRITS checkpoint: {ckpt}")
                    except Exception as exc:
                        print(f"[WARN] Failed to load BRITS checkpoint: {exc}. Retraining.")
                        wrapper = BRITSWrapper(**kwargs)
                        print(f"[INFO] Training BRITS for seed={seed}...", flush=True)
                        t0=time.time(); wrapper.fit(train_windows, val_windows, val_windows_ori); train_time=time.time()-t0
                        wrapper.save(ckpt); print(f"[INFO] Saved brits checkpoint: {ckpt}")
                else:
                    wrapper = BRITSWrapper(**kwargs)
                    print(f"[INFO] Training BRITS for seed={seed}...", flush=True)
                    t0=time.time(); wrapper.fit(train_windows, val_windows, val_windows_ori); train_time=time.time()-t0
                    wrapper.save(ckpt); print(f"[INFO] Saved brits checkpoint: {ckpt}")
                trained_models[method]=(wrapper, train_time)
            elif method == "csdi":
                # v7.2 (E7): CSDI diffusion baseline, same fit/impute interface.
                ckpt = model_dir / f"csdi_seed{seed}{CSDIWrapper.checkpoint_ext}"
                kwargs = dict(n_steps=args.window, n_features=len(feature_names),
                              epochs=args.csdi_epochs, batch_size=args.csdi_batch_size,
                              patience=args.csdi_patience,
                              n_layers=args.csdi_layers, n_heads=args.csdi_heads,
                              n_channels=args.csdi_channels,
                              n_diffusion_steps=args.csdi_diffusion_steps,
                              n_sampling_times=args.csdi_sampling_times,
                              verbose=not args.quiet)
                if ckpt.exists() and not args.retrain_models:
                    try:
                        wrapper = CSDIWrapper.load_from_checkpoint(ckpt, **kwargs)
                        train_time = 0.0
                        print(f"[INFO] Loaded CSDI checkpoint: {ckpt}")
                    except Exception as exc:
                        print(f"[WARN] Failed to load CSDI checkpoint: {exc}. Retraining.")
                        wrapper = CSDIWrapper(**kwargs)
                        print(f"[INFO] Training CSDI for seed={seed}...", flush=True)
                        t0=time.time(); wrapper.fit(train_windows, val_windows, val_windows_ori); train_time=time.time()-t0
                        wrapper.save(ckpt); print(f"[INFO] Saved csdi checkpoint: {ckpt}")
                else:
                    wrapper = CSDIWrapper(**kwargs)
                    print(f"[INFO] Training CSDI for seed={seed}...", flush=True)
                    t0=time.time(); wrapper.fit(train_windows, val_windows, val_windows_ori); train_time=time.time()-t0
                    wrapper.save(ckpt); print(f"[INFO] Saved csdi checkpoint: {ckpt}")
                trained_models[method]=(wrapper, train_time)
            elif method == "pcrsaitsv14_brits_no_seasonal_branch":
                # v7.2 (E6): PCR corrector on a BRITS backbone. Train (or load)
                # BRITS first, then build the corrector with base_model=BRITS.
                if "brits" not in trained_models:
                    brits_ckpt = model_dir / f"brits_seed{seed}{BRITSWrapper.checkpoint_ext}"
                    kwargs_b = dict(n_steps=args.window, n_features=len(feature_names), epochs=args.brits_epochs,
                                    batch_size=args.brits_batch_size, patience=args.brits_patience,
                                    rnn_hidden_size=args.brits_hidden_size, verbose=not args.quiet)
                    if brits_ckpt.exists():
                        brits_wrapper = BRITSWrapper.load_from_checkpoint(brits_ckpt, **kwargs_b)
                        trained_models["brits"] = (brits_wrapper, 0.0)
                    else:
                        raise RuntimeError(f"{method} requires a trained BRITS checkpoint "
                                           f"(include 'brits' in --methods before it)")
                brits_wrapper, _ = trained_models["brits"]
                ckpt = model_dir / f"{method}_seed{seed}{PCRSAITSV1CleanWrapper.checkpoint_ext}"
                # E6 config: the corrector uses the SAME settings as the proposed
                # SAITS model 'pcrsaitsv14_no_seasonal_branch' (no-seasonal branch,
                # WITH domain tags -> 10-dim), differing ONLY in the backbone,
                # which here is BRITS (base_model=brits_wrapper). This makes E6 a
                # clean backbone-swap ablation answering R4.2.
                kwargs = dict(variant=method, n_steps=args.window, learning_rate=args.pcr_lr, weight_decay=args.pcr_weight_decay,
                              epochs=args.pcr_epochs, batch_size=args.pcr_batch_size, patience=args.pcr_patience,
                              preserve_loss_weight=args.pcr_preserve_loss_weight, rel_loss_weight=args.pcr_rel_loss_weight,
                              sparse_loss_weight=args.pcr_sparse_loss_weight, base_model=brits_wrapper, feature_names=feature_names,
                              base_impute_stride=args.eval_stride, verbose=not args.quiet)
                if ckpt.exists() and not args.retrain_models:
                    try:
                        wrapper = PCRSAITSV1CleanWrapper.load_from_checkpoint(ckpt, **kwargs)
                        train_time = 0.0
                        print(f"[INFO] Loaded {method} checkpoint: {ckpt}")
                    except Exception as exc:
                        print(f"[WARN] Failed to load PCR checkpoint for {method}: {exc}. Retraining.")
                        wrapper = PCRSAITSV1CleanWrapper(**kwargs)
                        print(f"[INFO] Training {method.upper()} (BRITS backbone) for seed={seed}...", flush=True)
                        t0=time.time(); wrapper.fit(train_std, train_holdout_mask, train_holdout_gap, val_std, val_holdout_mask, val_holdout_gap); train_time=time.time()-t0
                        wrapper.save(ckpt); print(f"[INFO] Saved {method} checkpoint: {ckpt}")
                else:
                    wrapper = PCRSAITSV1CleanWrapper(**kwargs)
                    print(f"[INFO] Training {method.upper()} (BRITS backbone) for seed={seed}...", flush=True)
                    t0=time.time(); wrapper.fit(train_std, train_holdout_mask, train_holdout_gap, val_std, val_holdout_mask, val_holdout_gap); train_time=time.time()-t0
                    wrapper.save(ckpt); print(f"[INFO] Saved {method} checkpoint: {ckpt}")
                trained_models[method]=(wrapper, train_time)
            elif method.startswith("pcrsaitsv14_") or method == "pcr_mlp_no_domain_tags":
                if "saits" not in trained_models:
                    saits_ckpt = model_dir / f"saits_seed{seed}{SAITSWrapper.checkpoint_ext}"
                    kwargs_s = dict(n_steps=args.window, n_features=len(feature_names), epochs=args.saits_epochs, batch_size=args.saits_batch_size,
                                    patience=args.saits_patience, d_model=args.saits_d_model, d_ffn=args.saits_d_ffn, n_heads=args.saits_heads,
                                    n_layers=args.saits_layers, dropout=args.saits_dropout, verbose=not args.quiet)
                    if saits_ckpt.exists():
                        saits_wrapper = SAITSWrapper.load_from_checkpoint(saits_ckpt, **kwargs_s)
                        trained_models["saits"] = (saits_wrapper, 0.0)
                    else:
                        raise RuntimeError(f"{method} requires a trained SAITS checkpoint")
                saits_wrapper, _ = trained_models["saits"]
                ckpt = model_dir / f"{method}_seed{seed}{PCRSAITSV1CleanWrapper.checkpoint_ext}"
                kwargs = dict(variant=method, n_steps=args.window, learning_rate=args.pcr_lr, weight_decay=args.pcr_weight_decay,
                              epochs=args.pcr_epochs, batch_size=args.pcr_batch_size, patience=args.pcr_patience,
                              preserve_loss_weight=args.pcr_preserve_loss_weight, rel_loss_weight=args.pcr_rel_loss_weight,
                              sparse_loss_weight=args.pcr_sparse_loss_weight, base_model=saits_wrapper, feature_names=feature_names,
                              base_impute_stride=args.eval_stride, verbose=not args.quiet)
                if ckpt.exists() and not args.retrain_models:
                    try:
                        wrapper = PCRSAITSV1CleanWrapper.load_from_checkpoint(ckpt, **kwargs)
                        train_time=0.0
                        print(f"[INFO] Loaded {method} checkpoint: {ckpt}")
                    except Exception as exc:
                        print(f"[WARN] Failed to load PCR checkpoint for {method}: {exc}. Retraining.")
                        wrapper = PCRSAITSV1CleanWrapper(**kwargs)
                        print(f"[INFO] Training {method.upper()} for seed={seed}...", flush=True)
                        t0=time.time(); wrapper.fit(train_std, train_holdout_mask, train_holdout_gap, val_std, val_holdout_mask, val_holdout_gap); train_time=time.time()-t0
                        wrapper.save(ckpt); print(f"[INFO] Saved {method} checkpoint: {ckpt}")
                else:
                    wrapper = PCRSAITSV1CleanWrapper(**kwargs)
                    print(f"[INFO] Training {method.upper()} for seed={seed}...", flush=True)
                    t0=time.time(); wrapper.fit(train_std, train_holdout_mask, train_holdout_gap, val_std, val_holdout_mask, val_holdout_gap); train_time=time.time()-t0
                    wrapper.save(ckpt); print(f"[INFO] Saved {method} checkpoint: {ckpt}")
                trained_models[method]=(wrapper, train_time)
            else:
                raise ValueError(f"Unknown method: {method}")

        for mechanism in args.mechanisms:
            for pattern in args.patterns:
                for rate in args.rates:
                    mask_seed = deterministic_scenario_seed(seed, mechanism, pattern, rate)
                    artificial_mask_full, gap_len_full = generate_artificial_mask(test_std, mechanism=mechanism, pattern=pattern, rate=rate, seed=mask_seed, ref_map=ref_map)
                    masked_test = apply_mask(test_std, artificial_mask_full)
                    masked_test_windows, _ = build_windows(masked_test, window=args.window, stride=args.eval_stride)
                    scenario_preds = {}
                    for method in args.methods:
                        key = checkpoint_key(seed, method, mechanism, pattern, rate)
                        case_scn = is_case_scenario(args, seed, mechanism, pattern, rate)
                        if key in done_keys and not case_scn:
                            print(f"[SKIP {completed}/{total_runs}] seed={seed} method={method} scenario={mechanism}/{pattern}/rate={rate} | checkpoint hit")
                            continue
                        infer_t0 = time.time()
                        if method == "linear":
                            imputed_windows = impute_linear_windows(masked_test_windows, fill_values=fill_values_std)
                            pred_std = reconstruct_from_windows(imputed_windows, test_starts, total_length=len(test_std))
                            train_time = 0.0
                        elif method == "seasonal_naive24":
                            imputed_windows = impute_seasonal_naive24_windows(masked_test_windows, fill_values=fill_values_std, lag=args.seasonal_lag)
                            pred_std = reconstruct_from_windows(imputed_windows, test_starts, total_length=len(test_std))
                            train_time = 0.0
                        elif method == "saits":
                            wrapper, train_time = trained_models.get(method, (None, 0.0))
                            if wrapper is None:
                                saits_ckpt = model_dir / f"saits_seed{seed}{SAITSWrapper.checkpoint_ext}"
                                kwargs_s = dict(n_steps=args.window, n_features=len(feature_names), epochs=args.saits_epochs, batch_size=args.saits_batch_size,
                                                patience=args.saits_patience, d_model=args.saits_d_model, d_ffn=args.saits_d_ffn, n_heads=args.saits_heads,
                                                n_layers=args.saits_layers, dropout=args.saits_dropout, verbose=not args.quiet)
                                wrapper = SAITSWrapper.load_from_checkpoint(saits_ckpt, **kwargs_s)
                            base_imputed_windows = wrapper.impute(masked_test_windows)
                            pred_std = reconstruct_from_windows(base_imputed_windows, test_starts, total_length=len(test_std))
                        elif method == "brits":
                            wrapper, train_time = trained_models.get(method, (None, 0.0))
                            if wrapper is None:
                                brits_ckpt = model_dir / f"brits_seed{seed}{BRITSWrapper.checkpoint_ext}"
                                kwargs_b = dict(n_steps=args.window, n_features=len(feature_names), epochs=args.brits_epochs, batch_size=args.brits_batch_size,
                                                patience=args.brits_patience, rnn_hidden_size=args.brits_hidden_size, verbose=not args.quiet)
                                wrapper = BRITSWrapper.load_from_checkpoint(brits_ckpt, **kwargs_b)
                            imp_windows = wrapper.impute(masked_test_windows)
                            pred_std = reconstruct_from_windows(imp_windows, test_starts, total_length=len(test_std))
                        elif method == "csdi":
                            # v7.2 (E7): CSDI eval, same shape as BRITS.
                            wrapper, train_time = trained_models.get(method, (None, 0.0))
                            if wrapper is None:
                                csdi_ckpt = model_dir / f"csdi_seed{seed}{CSDIWrapper.checkpoint_ext}"
                                kwargs_c = dict(n_steps=args.window, n_features=len(feature_names), epochs=args.csdi_epochs,
                                                batch_size=args.csdi_batch_size, patience=args.csdi_patience,
                                                n_layers=args.csdi_layers, n_heads=args.csdi_heads, n_channels=args.csdi_channels,
                                                n_diffusion_steps=args.csdi_diffusion_steps,
                                                n_sampling_times=args.csdi_sampling_times, verbose=not args.quiet)
                                wrapper = CSDIWrapper.load_from_checkpoint(csdi_ckpt, **kwargs_c)
                            imp_windows = wrapper.impute(masked_test_windows)
                            pred_std = reconstruct_from_windows(imp_windows, test_starts, total_length=len(test_std))
                        elif method == "pcrsaitsv14_brits_no_seasonal_branch":
                            # v7.2 (E6): PCR corrector on a BRITS backbone.
                            wrapper, train_time = trained_models.get(method, (None, 0.0))
                            brits_base = None
                            if wrapper is None:
                                brits_ckpt = model_dir / f"brits_seed{seed}{BRITSWrapper.checkpoint_ext}"
                                kwargs_b = dict(n_steps=args.window, n_features=len(feature_names), epochs=args.brits_epochs, batch_size=args.brits_batch_size,
                                                patience=args.brits_patience, rnn_hidden_size=args.brits_hidden_size, verbose=not args.quiet)
                                brits_base = BRITSWrapper.load_from_checkpoint(brits_ckpt, **kwargs_b)
                                pcr_ckpt = model_dir / f"{method}_seed{seed}{PCRSAITSV1CleanWrapper.checkpoint_ext}"
                                kwargs_p = dict(variant=method, n_steps=args.window, learning_rate=args.pcr_lr, weight_decay=args.pcr_weight_decay,
                                                epochs=args.pcr_epochs, batch_size=args.pcr_batch_size, patience=args.pcr_patience,
                                                preserve_loss_weight=args.pcr_preserve_loss_weight, rel_loss_weight=args.pcr_rel_loss_weight,
                                                sparse_loss_weight=args.pcr_sparse_loss_weight, base_model=brits_base, feature_names=feature_names, verbose=not args.quiet)
                                wrapper = PCRSAITSV1CleanWrapper.load_from_checkpoint(pcr_ckpt, **kwargs_p)
                            else:
                                brits_base, _ = trained_models.get("brits", (None, 0.0))
                                if brits_base is None:
                                    brits_ckpt = model_dir / f"brits_seed{seed}{BRITSWrapper.checkpoint_ext}"
                                    kwargs_b = dict(n_steps=args.window, n_features=len(feature_names), epochs=args.brits_epochs, batch_size=args.brits_batch_size,
                                                    patience=args.brits_patience, rnn_hidden_size=args.brits_hidden_size, verbose=not args.quiet)
                                    brits_base = BRITSWrapper.load_from_checkpoint(brits_ckpt, **kwargs_b)
                            base_imputed_windows = brits_base.impute(masked_test_windows)
                            base_imputed_full = reconstruct_from_windows(base_imputed_windows, test_starts, total_length=len(test_std))
                            base_imputed_full[np.isfinite(masked_test)] = masked_test[np.isfinite(masked_test)]
                            pred_std = wrapper.correct(masked_test, base_imputed_full, artificial_mask_full, gap_len_full)
                        elif method.startswith("pcrsaitsv14_") or method == "pcr_mlp_no_domain_tags":
                            wrapper, train_time = trained_models.get(method, (None, 0.0))
                            if wrapper is None:
                                saits_ckpt = model_dir / f"saits_seed{seed}{SAITSWrapper.checkpoint_ext}"
                                kwargs_s = dict(n_steps=args.window, n_features=len(feature_names), epochs=args.saits_epochs, batch_size=args.saits_batch_size,
                                                patience=args.saits_patience, d_model=args.saits_d_model, d_ffn=args.saits_d_ffn, n_heads=args.saits_heads,
                                                n_layers=args.saits_layers, dropout=args.saits_dropout, verbose=not args.quiet)
                                saits_wrapper = SAITSWrapper.load_from_checkpoint(saits_ckpt, **kwargs_s)
                                pcr_ckpt = model_dir / f"{method}_seed{seed}{PCRSAITSV1CleanWrapper.checkpoint_ext}"
                                kwargs_p = dict(variant=method, n_steps=args.window, learning_rate=args.pcr_lr, weight_decay=args.pcr_weight_decay,
                                                epochs=args.pcr_epochs, batch_size=args.pcr_batch_size, patience=args.pcr_patience,
                                                preserve_loss_weight=args.pcr_preserve_loss_weight, rel_loss_weight=args.pcr_rel_loss_weight,
                                                sparse_loss_weight=args.pcr_sparse_loss_weight, base_model=saits_wrapper, feature_names=feature_names, verbose=not args.quiet)
                                wrapper = PCRSAITSV1CleanWrapper.load_from_checkpoint(pcr_ckpt, **kwargs_p)
                                saits_base = saits_wrapper
                            else:
                                saits_base, _ = trained_models.get("saits", (None, 0.0))
                                if saits_base is None:
                                    saits_ckpt = model_dir / f"saits_seed{seed}{SAITSWrapper.checkpoint_ext}"
                                    kwargs_s = dict(n_steps=args.window, n_features=len(feature_names), epochs=args.saits_epochs, batch_size=args.saits_batch_size,
                                                    patience=args.saits_patience, d_model=args.saits_d_model, d_ffn=args.saits_d_ffn, n_heads=args.saits_heads,
                                                    n_layers=args.saits_layers, dropout=args.saits_dropout, verbose=not args.quiet)
                                    saits_base = SAITSWrapper.load_from_checkpoint(saits_ckpt, **kwargs_s)
                            base_imputed_windows = saits_base.impute(masked_test_windows)
                            base_imputed_full = reconstruct_from_windows(base_imputed_windows, test_starts, total_length=len(test_std))
                            base_imputed_full[np.isfinite(masked_test)] = masked_test[np.isfinite(masked_test)]
                            pred_std = wrapper.correct(masked_test, base_imputed_full, artificial_mask_full, gap_len_full)
                        else:
                            raise ValueError(method)
                        pred_std[np.isfinite(masked_test)] = masked_test[np.isfinite(masked_test)]
                        infer_time = time.time() - infer_t0
                        pred = inverse_transform_values(pred_std, mean, std)
                        truth = inverse_transform_values(test_std, mean, std)
                        eval_mask = artificial_mask_full
                        met = compute_metrics(truth, pred, eval_mask)
                        grp = compute_grouped_metrics(truth, pred, eval_mask, feature_names)
                        point_mask = eval_mask & (gap_len_full <= 1.0 + 1e-8)
                        long_mask = eval_mask & (gap_len_full >= 12.0 - 1e-8)
                        point_rmse = compute_metrics(truth, pred, point_mask)["rmse"]
                        long_rmse = compute_metrics(truth, pred, long_mask)["rmse"]
                        row = {
                            "seed": seed, "method": method, "mechanism": mechanism, "pattern": pattern, "rate": rate,
                            "rmse": met["rmse"], "mae": met["mae"], "pointwise_rmse": point_rmse, "long_gap_rmse": long_rmse,
                            **grp, "train_time_sec": train_time, "infer_time_sec": infer_time,
                        }
                        was_new = key not in done_keys
                        if was_new:
                            results_rows.append(row)
                            append_csv_row(run_ckpt_path, row)
                            completed += 1
                            done_keys.add(key)
                            print(f"[DONE {completed}/{total_runs}] seed={seed} method={method} scenario={mechanism}/{pattern}/rate={rate} | RMSE={met['rmse']:.4f} MAE={met['mae']:.4f} train={train_time:.2f}s infer={infer_time:.2f}s")
                        else:
                            print(f"[INFO] Rebuilt case payload for seed={seed} method={method} scenario={mechanism}/{pattern}/rate={rate}")
                        feat_batch = []
                        for j, feat in enumerate(feature_names):
                            m = compute_metrics(truth[:, [j]], pred[:, [j]], eval_mask[:, [j]])
                            feat_row = {
                                "seed": seed, "method": method, "mechanism": mechanism, "pattern": pattern, "rate": rate,
                                "feature": feat, "group": infer_feature_group(feat), "rmse": m["rmse"], "mae": m["mae"],
                            }
                            # v7.2 fix: use was_new (captured BEFORE done_keys.add
                            # above). The original code re-tested `key not in
                            # done_keys` here, which was always False because the
                            # key had just been added, so per-feature rows were
                            # never collected and per_feature_results.csv came out
                            # empty.
                            if was_new:
                                feature_rows.append(feat_row)
                            feat_batch.append(feat_row)
                        if was_new or not case_scn:
                            append_csv_rows(feat_ckpt_path, feat_batch)
                        if case_scn:
                            scenario_preds[method] = pred.copy()
                    if case_scn:
                        ref_m = args.reference_method
                        prop_m = args.case_proposed_method if args.case_proposed_method in args.methods else ("pcr_mlp_no_domain_tags" if "pcr_mlp_no_domain_tags" in args.methods else ("pcrsaitsv14_base" if "pcrsaitsv14_base" in args.methods else None))
                        if ref_m in scenario_preds and prop_m in scenario_preds:
                            feat_name = args.case_feature if args.case_feature in feature_names else feature_names[0]
                            j = feature_names.index(feat_name)
                            case_payload = {
                                "truth": truth[:, j],
                                "masked": inverse_transform_values(masked_test, mean, std)[:, j],
                                "pred_ref": scenario_preds[ref_m][:, j],
                                "pred_prop": scenario_preds[prop_m][:, j],
                                "feature_name": feat_name,
                                "title": f"{mechanism}/{pattern}/rate={rate}",
                                "reference_method": ref_m,
                                "proposed_method": prop_m,
                            }

    results_df = pd.DataFrame(results_rows)
    feature_df = pd.DataFrame(feature_rows)
    summary_rows = []
    ref = results_df[results_df["method"] == args.reference_method].copy() if args.reference_method in set(results_df["method"]) else pd.DataFrame()
    for method in sorted(results_df["method"].unique()):
        sub = results_df[results_df["method"] == method]
        row = {
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
        }
        if method != args.reference_method and not ref.empty:
            merged = sub[["seed", "mechanism", "pattern", "rate", "rmse", "mae"]].merge(
                ref[["seed", "mechanism", "pattern", "rate", "rmse", "mae"]].rename(columns={"rmse": "ref_rmse", "mae": "ref_mae"}),
                on=["seed", "mechanism", "pattern", "rate"], how="inner")
            row["rmse_improvement_vs_reference_pct_mean"] = safe_mean((merged["ref_rmse"] - merged["rmse"]) / (merged["ref_rmse"] + EPS) * 100.0)
            row["mae_improvement_vs_reference_pct_mean"] = safe_mean((merged["ref_mae"] - merged["mae"]) / (merged["ref_mae"] + EPS) * 100.0)
        else:
            row["rmse_improvement_vs_reference_pct_mean"] = np.nan
            row["mae_improvement_vs_reference_pct_mean"] = np.nan
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    wilcoxon_df = compute_wilcoxon(results_df, args.reference_method)
    grouped_summary_df = results_df.groupby("method")[["pollutant_rmse", "sensor_rmse", "meteorological_rmse"]].mean().reset_index()
    pattern_summary_df = results_df.groupby(["method", "pattern"])[["rmse", "mae"]].mean().reset_index()
    results_df.to_csv(output_dir / "run_results.csv", index=False)
    feature_df.to_csv(output_dir / "per_feature_results.csv", index=False)
    summary_df.to_csv(output_dir / "summary_results.csv", index=False)
    wilcoxon_df.to_csv(output_dir / "wilcoxon_vs_reference.csv", index=False)
    grouped_summary_df.to_csv(output_dir / "grouped_summary_results.csv", index=False)
    pattern_summary_df.to_csv(output_dir / "pattern_summary_results.csv", index=False)
    save_plot_rate(results_df, output_dir, metric="rmse")
    save_plot_rate(results_df, output_dir, metric="mae")
    save_feature_heatmap(feature_df, output_dir)
    save_grouped_plot(results_df, output_dir)
    save_pattern_improvement_plot(results_df, output_dir, args.reference_method)
    save_case_study_plot(case_payload, output_dir)
    save_excel(output_dir, {
        "summary": summary_df,
        "runs": results_df,
        "per_feature": feature_df,
        "wilcoxon": wilcoxon_df,
        "grouped": grouped_summary_df,
        "patterns": pattern_summary_df,
    })
    meta = {"dataset_path": str(dataset_path), "output_dir": str(output_dir), "features": feature_names, "reference_method": args.reference_method,
            "val_mechanism": args.val_mechanism, "val_pattern": args.val_pattern, "val_rate": args.val_rate}
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[INFO] Results saved to: {output_dir}")
    return summary_df, feature_df, meta


def build_parser():
    p = argparse.ArgumentParser(description="PCR-SAITS-v1.4 full protocol runner")
    p.add_argument("--script-dir", type=str, default=None)
    p.add_argument("--output-prefix", type=str, default="pcr_saits_v1_4_full_protocol_outputs")
    p.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    p.add_argument("--reference-method", type=str, default="saits")
    p.add_argument("--seeds", type=str, default="7,21,42,123,456")
    p.add_argument("--mechanisms", type=str, default="MCAR,MAR,MNAR")
    p.add_argument("--patterns", type=str, default="pointwise,block_6,block_12,block_24,block_48")
    p.add_argument("--rates", type=str, default="0.1,0.2,0.3,0.4,0.5")
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--window", type=int, default=48)
    p.add_argument("--train-stride", type=int, default=1)
    p.add_argument("--eval-stride", type=int, default=24)
    p.add_argument("--saits-epochs", type=int, default=50)
    p.add_argument("--saits-batch-size", type=int, default=32)
    p.add_argument("--saits-patience", type=int, default=10)
    p.add_argument("--saits-d-model", type=int, default=64)
    p.add_argument("--saits-d-ffn", type=int, default=128)
    p.add_argument("--saits-heads", type=int, default=4)
    p.add_argument("--saits-layers", type=int, default=2)
    p.add_argument("--saits-dropout", type=float, default=0.1)
    p.add_argument("--brits-epochs", type=int, default=50)
    p.add_argument("--brits-batch-size", type=int, default=32)
    p.add_argument("--brits-patience", type=int, default=10)
    p.add_argument("--brits-hidden-size", type=int, default=128)
    # v7.2: CSDI (E7) hyperparameters
    p.add_argument("--csdi-epochs", type=int, default=50)
    p.add_argument("--csdi-batch-size", type=int, default=32)
    p.add_argument("--csdi-patience", type=int, default=10)
    p.add_argument("--csdi-layers", type=int, default=4)
    p.add_argument("--csdi-heads", type=int, default=8)
    p.add_argument("--csdi-channels", type=int, default=64)
    p.add_argument("--csdi-diffusion-steps", type=int, default=50)
    p.add_argument("--csdi-sampling-times", type=int, default=10)
    p.add_argument("--pcr-epochs", type=int, default=30)
    p.add_argument("--pcr-batch-size", type=int, default=256)
    p.add_argument("--pcr-patience", type=int, default=6)
    p.add_argument("--pcr-lr", type=float, default=1e-3)
    p.add_argument("--pcr-weight-decay", type=float, default=1e-5)
    p.add_argument("--pcr-preserve-loss-weight", type=float, default=0.05)
    p.add_argument("--pcr-rel-loss-weight", type=float, default=0.5)
    p.add_argument("--pcr-sparse-loss-weight", type=float, default=0.002)

    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--reset-checkpoints", action="store_true")
    p.add_argument("--retrain-models", action="store_true")
    p.add_argument("--seasonal-lag", type=int, default=24)
    p.add_argument("--val-mechanism", type=str, default="MCAR")
    p.add_argument("--val-pattern", type=str, default="pointwise")
    p.add_argument("--val-rate", type=float, default=0.1)
    p.add_argument("--case-seed", type=int, default=7)
    p.add_argument("--case-mechanism", type=str, default="MCAR")
    p.add_argument("--case-pattern", type=str, default="block_24")
    p.add_argument("--case-rate", type=float, default=0.3)
    p.add_argument("--case-feature", type=str, default="T")
    p.add_argument("--case-proposed-method", type=str, default="pcr_mlp_no_domain_tags")
    p.add_argument("--quiet", action="store_true")
    return p


def parse_args():
    p = build_parser()
    args = p.parse_args()
    args.methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    args.seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    args.mechanisms = [s.strip().upper() for s in args.mechanisms.split(",") if s.strip()]
    args.patterns = [s.strip() for s in args.patterns.split(",") if s.strip()]
    args.rates = [float(s.strip()) for s in args.rates.split(",") if s.strip()]
    args.val_mechanism = str(args.val_mechanism).upper()
    args.val_pattern = str(args.val_pattern)
    args.case_mechanism = str(args.case_mechanism).upper()
    args.case_pattern = str(args.case_pattern)
    args.case_feature = str(args.case_feature)
    args.case_proposed_method = str(args.case_proposed_method)
    return args


def main():
    args = parse_args()
    try:
        run_experiment(args)
    except Exception as e:
        print(f"[ERROR] {e}")
        raise



# ===== Extended paper-ready suite overrides =====
_ORIG_RUN_EXPERIMENT_SINGLE = run_experiment
_ORIG_FIND_DATASET_FILE = find_dataset_file
_ORIG_LOAD_AIR_QUALITY_DATASET = load_air_quality_dataset
_ACTIVE_DATASET_NAME = "airquality"
_ACTIVE_DATASET_CACHE = None

EXTENDED_DEFAULT_METHODS = [
    "linear",
    "seasonal_naive24",
    "brits",
    "saits",
    "pcr_mlp_no_domain_tags",
    "pcrsaitsv14_base",
    "pcrsaitsv14_masked_residual",
    "pcrsaitsv14_no_local_branch",
    "pcrsaitsv14_no_seasonal_branch",
    "pcrsaitsv14_with_preserve_loss",
    "pcrsaitsv14_with_rel_loss",
    "pcrsaitsv14_with_sparse_loss",
]
BEIJING_POLLUTANT_VARS = {"PM2.5", "PM10", "SO2", "NO2", "CO", "O3"}
BEIJING_METEO_VARS = {"TEMP", "PRES", "DEWP", "RAIN", "WSPM"}
SYNTHETIC_FEATURES = [
    "CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)", "PT08.S2(NMHC)",
    "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)",
    "T", "RH", "AH",
]

def infer_feature_group(feature_name: str) -> str:
    if feature_name in METEO_VARS or feature_name in BEIJING_METEO_VARS:
        return "meteorological"
    if feature_name.startswith("PT08."):
        return "sensor"
    if feature_name in POLLUTANT_VARS or feature_name in BEIJING_POLLUTANT_VARS:
        return "pollutant"
    if feature_name not in _UNKNOWN_GROUP_WARNED:
        print(f"[WARN] Unknown feature group for '{feature_name}'. Falling back to 'sensor'.")
        _UNKNOWN_GROUP_WARNED.add(feature_name)
    return "sensor"


def _resolve_dataset_file(script_dir: Path, dataset_name: str) -> Path:
    name = str(dataset_name).strip().lower()
    if name in {"airquality", "airqualityuci", "uci_airquality"}:
        candidates = [
            "AirQualityUCI.xlsx", "AirQualityUCI.csv", "AirQualityUCI.xls", "AirQualityUCI.data",
            "airqualityuci.xlsx", "airqualityuci.csv", "airqualityuci.xls", "airqualityuci.data",
        ]
        for c in candidates:
            p = script_dir / c
            if p.exists():
                return p
        return _ORIG_FIND_DATASET_FILE(script_dir)
    if name in {"beijing", "beijingpm25", "beijing_pm25", "pm25"}:
        candidates = [
            "BeijingPM25.csv", "beijingpm25.csv", "BeijingPM25.xlsx", "beijingpm25.xlsx",
            "PRSA_data_2010.1.1-2014.12.31.csv",
        ]
        for c in candidates:
            p = script_dir / c
            if p.exists():
                return p
        raise FileNotFoundError("BeijingPM25 dataset not found in script directory.")
    if name in {"synthetic", "synth"}:
        return script_dir / "__synthetic__.synthetic"
    raise ValueError(f"Unsupported dataset name: {dataset_name}")


def find_dataset_file(script_dir: Path) -> Path:
    return _resolve_dataset_file(script_dir, globals().get("_ACTIVE_DATASET_NAME", "airquality"))


def load_beijing_pm25_dataset(path: Path):
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    # timestamp parsing
    lower = {c.lower(): c for c in df.columns}
    if {"year", "month", "day", "hour"}.issubset(lower):
        df["timestamp"] = pd.to_datetime(
            pd.DataFrame({
                "year": df[lower["year"]],
                "month": df[lower["month"]],
                "day": df[lower["day"]],
                "hour": df[lower["hour"]],
            }),
            errors="coerce",
        )
    elif "date" in lower:
        df["timestamp"] = pd.to_datetime(df[lower["date"]], errors="coerce")
    else:
        raise ValueError(f"BeijingPM25 dataset needs either year/month/day/hour or date columns. Found: {df.columns.tolist()}")
    raw_rows = len(df)
    valid_rows = df["timestamp"].notna().sum()
    unique_rows = df["timestamp"].nunique(dropna=True)
    print(f"[INFO] Timestamp diagnostics | raw={raw_rows} valid={valid_rows} unique={unique_rows}")
    # select features
    preferred = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
    features = [c for c in preferred if c in df.columns]
    if len(features) < 3:
        numeric = [c for c in df.columns if c != "timestamp"]
        features = numeric
    df = df.dropna(subset=["timestamp"]).copy()
    for c in features:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    nonnegative_cols = [c for c in ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "PRES", "RAIN", "WSPM"] if c in features]
    for c in nonnegative_cols:
        if c == "PRES":
            df.loc[df[c] <= 0, c] = np.nan
        else:
            df.loc[df[c] < 0, c] = np.nan
    dup = int(df["timestamp"].duplicated().sum())
    if dup > 0:
        print(f"[WARN] Duplicate timestamps found and removed: {dup}")
    df = df.drop_duplicates(subset=["timestamp"], keep="first").sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp"] + features], features


def load_synthetic_dataset(path: Path, n_rows: int = 5000, seed: int = 1234):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2020-01-01", periods=n_rows, freq="h")
    t = np.arange(n_rows)
    day = 2 * np.pi * t / 24.0
    week = 2 * np.pi * t / (24.0 * 7.0)
    temp = 22 + 6*np.sin(day-0.7) + 1.5*np.sin(week) + rng.normal(0, 0.8, size=n_rows)
    rh = 55 - 10*np.sin(day-0.4) + rng.normal(0, 2.0, size=n_rows)
    ah = 8 + 0.15*temp + 0.03*rh + rng.normal(0, 0.3, size=n_rows)
    co = 2.0 + 0.8*np.sin(day+0.3) + 0.4*np.sin(week) + rng.normal(0, 0.15, size=n_rows)
    nmhc = 120 + 35*np.sin(day-0.2) + 10*np.sin(week) + rng.normal(0, 5, size=n_rows)
    c6h6 = 9 + 1.1*co + rng.normal(0, 0.3, size=n_rows)
    nox = 80 + 18*np.sin(day+1.2) + rng.normal(0, 4, size=n_rows)
    no2 = 45 + 0.45*nox + rng.normal(0, 2.5, size=n_rows)
    s1 = 1000 + 90*co + rng.normal(0, 20, size=n_rows)
    s2 = 900 + 35*c6h6 + rng.normal(0, 15, size=n_rows)
    s3 = 1100 - 4*nox + rng.normal(0, 20, size=n_rows)
    s4 = 1500 + 8*no2 + rng.normal(0, 25, size=n_rows)
    s5 = 950 + 6*ah + 3*temp + rng.normal(0, 18, size=n_rows)
    df = pd.DataFrame({
        "timestamp": ts,
        "CO(GT)": co,
        "PT08.S1(CO)": s1,
        "NMHC(GT)": nmhc,
        "C6H6(GT)": c6h6,
        "PT08.S2(NMHC)": s2,
        "NOx(GT)": nox,
        "PT08.S3(NOx)": s3,
        "NO2(GT)": no2,
        "PT08.S4(NO2)": s4,
        "PT08.S5(O3)": s5,
        "T": temp,
        "RH": rh,
        "AH": ah,
    })
    # inject natural missingness lightly
    for c in SYNTHETIC_FEATURES:
        miss = rng.random(n_rows) < 0.04
        df.loc[miss, c] = np.nan
    raw_rows = len(df)
    print(f"[INFO] Timestamp diagnostics | raw={raw_rows} valid={raw_rows} unique={raw_rows}")
    return df, SYNTHETIC_FEATURES[:]


def load_air_quality_dataset(path: Path):
    global _ACTIVE_DATASET_CACHE
    cache = _ACTIVE_DATASET_CACHE
    try:
        resolved = str(path.resolve())
    except Exception:
        resolved = str(path)
    if cache is not None:
        cache_path = cache.get("path")
        cache_name = cache.get("dataset")
        if cache_path == resolved or cache_name == _ACTIVE_DATASET_NAME:
            df = cache["df"].copy()
            feats = list(cache["features"])
            return df, feats
    name = path.name.lower()
    if name.endswith('.synthetic') or '__synthetic__' in name:
        return load_synthetic_dataset(path)
    if 'beijing' in name or 'prsa' in name:
        return load_beijing_pm25_dataset(path)
    return _ORIG_LOAD_AIR_QUALITY_DATASET(path)


def compute_pairwise_wilcoxon(results_df: pd.DataFrame):
    if wilcoxon is None or results_df.empty:
        return pd.DataFrame()
    methods = sorted(results_df["method"].dropna().unique().tolist())
    rows = []
    for i, m1 in enumerate(methods):
        for m2 in methods[i+1:]:
            a = results_df[results_df["method"] == m1][["seed", "mechanism", "pattern", "rate", "rmse", "mae"]].rename(columns={"rmse": "rmse_a", "mae": "mae_a"})
            b = results_df[results_df["method"] == m2][["seed", "mechanism", "pattern", "rate", "rmse", "mae"]].rename(columns={"rmse": "rmse_b", "mae": "mae_b"})
            merged = a.merge(b, on=["seed", "mechanism", "pattern", "rate"], how="inner")
            if len(merged) < 2:
                continue
            try:
                p_rmse_two = wilcoxon(merged["rmse_a"], merged["rmse_b"], alternative="two-sided").pvalue
                p_mae_two = wilcoxon(merged["mae_a"], merged["mae_b"], alternative="two-sided").pvalue
                p_rmse = wilcoxon(merged["rmse_a"], merged["rmse_b"], alternative="less").pvalue
                p_mae = wilcoxon(merged["mae_a"], merged["mae_b"], alternative="less").pvalue
            except Exception:
                p_rmse = np.nan
                p_mae = np.nan
                p_rmse_two = np.nan
                p_mae_two = np.nan
            rows.append({"method_a": m1, "method_b": m2, "wilcoxon_p_rmse": p_rmse, "wilcoxon_p_mae": p_mae, "wilcoxon_p_rmse_two_sided": p_rmse_two, "wilcoxon_p_mae_two_sided": p_mae_two, "n_pairs": len(merged)})
    return pd.DataFrame(rows)


def compute_runtime_summary(results_df: pd.DataFrame):
    if results_df.empty:
        return pd.DataFrame()
    group_cols = [c for c in ["dataset", "method"] if c in results_df.columns]
    gb = results_df.groupby(group_cols, dropna=False)
    rows = []
    for key, sub in gb:
        if not isinstance(key, tuple):
            key = (key,)
        cols = ["dataset", "method"] if "dataset" in results_df.columns else ["method"]
        row = {cols[i]: key[i] for i in range(len(cols))}
        train_sub = sub[[c for c in ["dataset", "method", "seed", "train_time_sec"] if c in sub.columns]].drop_duplicates()
        infer_sub = sub["infer_time_sec"]
        row.update({
            "train_time_mean": safe_mean(train_sub["train_time_sec"]),
            "train_time_std": safe_std(train_sub["train_time_sec"]),
            "train_n_unique_models": len(train_sub),
            "infer_time_mean": safe_mean(infer_sub),
            "infer_time_std": safe_std(infer_sub),
            "n_runs": len(sub),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def compute_missingness_profile(df: pd.DataFrame, feature_names: List[str], dataset_name: str):
    rows = []
    synthetic_like = str(dataset_name).lower().startswith("synthetic")
    for f in feature_names:
        vals = pd.to_numeric(df[f], errors="coerce")
        miss_count = int(vals.isna().sum())
        miss_ratio = float(vals.isna().mean())
        row = {
            "dataset": dataset_name,
            "feature": f,
            "group": infer_feature_group(f),
            "missing_count": miss_count,
            "missing_ratio": miss_ratio,
        }
        if synthetic_like:
            row.update({
                "natural_missing_count": 0,
                "natural_missing_ratio": np.nan,
                "synthetic_seeded_missing_count": miss_count,
                "synthetic_seeded_missing_ratio": miss_ratio,
                "missingness_source": "synthetic_seeded",
            })
        else:
            row.update({
                "natural_missing_count": miss_count,
                "natural_missing_ratio": miss_ratio,
                "synthetic_seeded_missing_count": 0,
                "synthetic_seeded_missing_ratio": np.nan,
                "missingness_source": "natural",
            })
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_natural_missing(profile_df: pd.DataFrame):
    if profile_df.empty:
        return pd.DataFrame()
    rows = []
    for ds, sub in profile_df.groupby("dataset", dropna=False):
        source = "synthetic_seeded" if (sub.get("missingness_source") == "synthetic_seeded").any() else "natural"
        rows.append({
            "dataset": ds,
            "features": int(len(sub)),
            "missingness_source": source,
            "natural_missing_ratio_mean": safe_mean(sub["natural_missing_ratio"]) if "natural_missing_ratio" in sub.columns else np.nan,
            "natural_missing_ratio_max": float(np.nanmax(sub["natural_missing_ratio"].to_numpy(dtype=float))) if "natural_missing_ratio" in sub.columns and np.isfinite(sub["natural_missing_ratio"].to_numpy(dtype=float)).any() else np.nan,
            "synthetic_seeded_missing_ratio_mean": safe_mean(sub["synthetic_seeded_missing_ratio"]) if "synthetic_seeded_missing_ratio" in sub.columns else np.nan,
            "synthetic_seeded_missing_ratio_max": float(np.nanmax(sub["synthetic_seeded_missing_ratio"].to_numpy(dtype=float))) if "synthetic_seeded_missing_ratio" in sub.columns and np.isfinite(sub["synthetic_seeded_missing_ratio"].to_numpy(dtype=float)).any() else np.nan,
            "overall_missing_ratio_mean": safe_mean(sub["missing_ratio"]),
            "overall_missing_ratio_max": float(np.nanmax(sub["missing_ratio"].to_numpy(dtype=float))) if np.isfinite(sub["missing_ratio"].to_numpy(dtype=float)).any() else np.nan,
        })
    return pd.DataFrame(rows)


def run_sensitivity_analysis(args, script_dir: Path, base_output_dir: Path):
    if not getattr(args, 'run_sensitivity', False):
        return pd.DataFrame()
    rows = []
    preserve_vals = [float(x) for x in str(args.sensitivity_preserve_values).split(',') if str(x).strip()]
    rel_vals = [float(x) for x in str(args.sensitivity_rel_values).split(',') if str(x).strip()]
    sparse_vals = [float(x) for x in str(args.sensitivity_sparse_values).split(',') if str(x).strip()]

    sens_seeds = list(args.seeds[: min(3, len(args.seeds))]) if getattr(args, 'seeds', None) else [7]
    sens_mechs = list(args.mechanisms[: min(2, len(args.mechanisms))]) if getattr(args, 'mechanisms', None) else ["MCAR", "MAR"]
    candidate_patterns = [p for p in ["pointwise", "block_24"] if p in getattr(args, 'patterns', [])]
    sens_patterns = candidate_patterns if candidate_patterns else list(args.patterns[: min(2, len(args.patterns))])
    sens_rates = list(args.rates[: min(2, len(args.rates))]) if getattr(args, 'rates', None) else [0.1, 0.3]

    for pval in preserve_vals:
        for rval in rel_vals:
            for sval in sparse_vals:
                sargs = argparse.Namespace(**vars(args))
                sargs.methods = ["saits", "pcr_mlp_no_domain_tags"]
                sargs.seeds = sens_seeds
                sargs.mechanisms = sens_mechs
                sargs.patterns = sens_patterns
                sargs.rates = sens_rates
                try:
                    rel_base = base_output_dir.relative_to(script_dir)
                    sens_out = rel_base / f"sensitivity_p{pval}_r{rval}_s{sval}"
                    sargs.output_dir = sens_out.as_posix()
                except Exception:
                    sargs.output_dir = str((base_output_dir / f"sensitivity_p{pval}_r{rval}_s{sval}").resolve())
                sargs.reset_checkpoints = True
                sargs.retrain_models = True
                sargs.pcr_preserve_loss_weight = pval
                sargs.pcr_rel_loss_weight = rval
                sargs.pcr_sparse_loss_weight = sval
                globals()["_ACTIVE_DATASET_NAME"] = args.datasets[0]
                summary_df, _, meta = _ORIG_RUN_EXPERIMENT_SINGLE(sargs)
                if not summary_df.empty:
                    sub = summary_df[summary_df["method"] == "pcr_mlp_no_domain_tags"]
                    if not sub.empty:
                        row = sub.iloc[0].to_dict()
                        row.update({
                            "dataset": args.datasets[0],
                            "preserve_loss_weight": pval,
                            "rel_loss_weight": rval,
                            "sparse_loss_weight": sval,
                            "sensitivity_seeds": ",".join(map(str, sens_seeds)),
                            "sensitivity_mechanisms": ",".join(map(str, sens_mechs)),
                            "sensitivity_patterns": ",".join(map(str, sens_patterns)),
                            "sensitivity_rates": ",".join(map(str, sens_rates)),
                            "n_sensitivity_seeds": len(sens_seeds),
                            "n_sensitivity_scenarios": len(sens_mechs) * len(sens_patterns) * len(sens_rates),
                            "output_dir": meta.get("output_dir", ""),
                        })
                        rows.append(row)
    return pd.DataFrame(rows)


def build_parser():
    p = argparse.ArgumentParser(description="PCR-SAITS-v1.4 paper-ready suite")
    p.add_argument("--script-dir", type=str, default=None)
    p.add_argument("--output-prefix", type=str, default="pcr_saits_v1_4_paper_suite_outputs")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--reset-checkpoints", action="store_true")
    p.add_argument("--retrain-models", action="store_true")
    p.add_argument("--datasets", type=str, default="airquality,beijingpm25,synthetic")
    p.add_argument("--methods", type=str, default=",".join(EXTENDED_DEFAULT_METHODS))
    p.add_argument("--reference-method", type=str, default="saits")
    p.add_argument("--seeds", type=str, default="7,21,42,123,456")
    p.add_argument("--mechanisms", type=str, default="MCAR,MAR,MNAR")
    p.add_argument("--patterns", type=str, default="pointwise,block_6,block_12,block_24,block_48")
    p.add_argument("--rates", type=str, default="0.1,0.2,0.3,0.4,0.5")
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--window", type=int, default=48)
    p.add_argument("--train-stride", type=int, default=1)
    p.add_argument("--eval-stride", type=int, default=24)
    p.add_argument("--seasonal-lag", type=int, default=24)
    p.add_argument("--saits-epochs", type=int, default=50)
    p.add_argument("--saits-batch-size", type=int, default=32)
    p.add_argument("--saits-patience", type=int, default=10)
    p.add_argument("--saits-d-model", type=int, default=64)
    p.add_argument("--saits-d-ffn", type=int, default=128)
    p.add_argument("--saits-heads", type=int, default=4)
    p.add_argument("--saits-layers", type=int, default=2)
    p.add_argument("--saits-dropout", type=float, default=0.1)
    p.add_argument("--brits-epochs", type=int, default=50)
    p.add_argument("--brits-batch-size", type=int, default=32)
    p.add_argument("--brits-patience", type=int, default=10)
    p.add_argument("--brits-hidden-size", type=int, default=128)
    # v7.2: CSDI (E7) hyperparameters
    p.add_argument("--csdi-epochs", type=int, default=50)
    p.add_argument("--csdi-batch-size", type=int, default=32)
    p.add_argument("--csdi-patience", type=int, default=10)
    p.add_argument("--csdi-layers", type=int, default=4)
    p.add_argument("--csdi-heads", type=int, default=8)
    p.add_argument("--csdi-channels", type=int, default=64)
    p.add_argument("--csdi-diffusion-steps", type=int, default=50)
    p.add_argument("--csdi-sampling-times", type=int, default=10)
    p.add_argument("--pcr-epochs", type=int, default=30)
    p.add_argument("--pcr-batch-size", type=int, default=256)
    p.add_argument("--pcr-patience", type=int, default=6)
    p.add_argument("--pcr-lr", type=float, default=1e-3)
    p.add_argument("--pcr-weight-decay", type=float, default=1e-5)
    p.add_argument("--pcr-preserve-loss-weight", type=float, default=0.05)
    p.add_argument("--pcr-rel-loss-weight", type=float, default=0.5)
    p.add_argument("--pcr-sparse-loss-weight", type=float, default=0.002)
    p.add_argument("--val-mechanism", type=str, default="MCAR")
    p.add_argument("--val-pattern", type=str, default="pointwise")
    p.add_argument("--val-rate", type=float, default=0.1)
    p.add_argument("--case-seed", type=int, default=7)
    p.add_argument("--case-mechanism", type=str, default="MCAR")
    p.add_argument("--case-pattern", type=str, default="block_24")
    p.add_argument("--case-rate", type=float, default=0.3)
    p.add_argument("--case-feature", type=str, default="T")
    p.add_argument("--case-proposed-method", type=str, default="pcr_mlp_no_domain_tags")
    p.add_argument("--run-sensitivity", action="store_true")
    p.add_argument("--sensitivity-preserve-values", type=str, default="0.0,0.02,0.05")
    p.add_argument("--sensitivity-rel-values", type=str, default="0.0,0.25,0.5")
    p.add_argument("--sensitivity-sparse-values", type=str, default="0.0,0.001,0.002")
    p.add_argument("--quiet", action="store_true")
    return p


def parse_args():
    p = build_parser()
    args = p.parse_args()
    args.datasets = [s.strip() for s in str(args.datasets).split(',') if s.strip()]
    args.methods = [m.strip() for m in str(args.methods).split(',') if m.strip()]
    args.seeds = [int(s.strip()) for s in str(args.seeds).split(',') if s.strip()]
    args.mechanisms = [s.strip().upper() for s in str(args.mechanisms).split(',') if s.strip()]
    args.patterns = [s.strip() for s in str(args.patterns).split(',') if s.strip()]
    args.rates = [float(s.strip()) for s in str(args.rates).split(',') if s.strip()]
    args.val_mechanism = str(args.val_mechanism).upper()
    args.case_mechanism = str(args.case_mechanism).upper()
    return args


def run_experiment(args):
    script_dir = Path(args.script_dir).resolve() if args.script_dir else Path(__file__).resolve().parent
    base_output_dir = create_output_dir(script_dir, args.output_prefix, args.output_dir)
    all_summary, all_runs, all_feature = [], [], []
    all_grouped, all_pattern, all_wilcox, all_pairwise = [], [], [], []
    all_missing_profiles = []
    dataset_metas = []
    for ds in args.datasets:
        globals()["_ACTIVE_DATASET_NAME"] = ds
        ds_path = find_dataset_file(script_dir)
        ds_df, ds_features = load_air_quality_dataset(ds_path)
        globals()["_ACTIVE_DATASET_CACHE"] = {"dataset": ds, "path": str(ds_path.resolve()), "df": ds_df.copy(), "features": list(ds_features)}
        missing_profile_df = compute_missingness_profile(ds_df, ds_features, ds)
        # dataset-specific subdir inside base output
        sargs = argparse.Namespace(**vars(args))
        try:
            rel_base = base_output_dir.relative_to(script_dir)
            rel_subdir = (rel_base / ds).as_posix()
            sargs.output_dir = rel_subdir
        except Exception:
            sargs.output_dir = str((base_output_dir / ds).resolve())
        summary_df, feature_df, meta = _ORIG_RUN_EXPERIMENT_SINGLE(sargs)
        ds_out = Path(meta["output_dir"])
        # v7.2: guard empty CSVs. A file may exist but be zero-bytes / header-less
        # when a subset-methods or tiny-scenario run produces no rows for a given
        # stats file (this is exactly the pandas EmptyDataError we hit in Patch 4).
        def _read_csv_or_empty(p):
            p = Path(p)
            if (not p.exists()) or p.stat().st_size == 0:
                return pd.DataFrame()
            try:
                return pd.read_csv(p)
            except pd.errors.EmptyDataError:
                return pd.DataFrame()
        run_df = _read_csv_or_empty(ds_out / "run_results.csv")
        grouped_df = _read_csv_or_empty(ds_out / "grouped_summary_results.csv")
        pattern_df = _read_csv_or_empty(ds_out / "pattern_summary_results.csv")
        wilcox_df = _read_csv_or_empty(ds_out / "wilcoxon_vs_reference.csv")
        pairwise_df = compute_pairwise_wilcoxon(run_df)
        runtime_df = compute_runtime_summary(run_df)
        nat_sum_df = summarize_natural_missing(missing_profile_df)
        # attach dataset column
        for df_ in [summary_df, feature_df, run_df, grouped_df, pattern_df, wilcox_df, pairwise_df, runtime_df, missing_profile_df, nat_sum_df]:
            if df_ is not None and not df_.empty and "dataset" not in df_.columns:
                df_.insert(0, "dataset", ds)
        if not pairwise_df.empty:
            pairwise_df.to_csv(ds_out / "wilcoxon_pairwise_all_methods.csv", index=False)
        if not runtime_df.empty:
            runtime_df.to_csv(ds_out / "runtime_summary_results.csv", index=False)
        if not missing_profile_df.empty:
            missing_profile_df.to_csv(ds_out / "natural_missing_profile.csv", index=False)
        if not nat_sum_df.empty:
            nat_sum_df.to_csv(ds_out / "natural_missing_summary.csv", index=False)
        all_summary.append(summary_df)
        all_runs.append(run_df)
        all_feature.append(feature_df)
        all_grouped.append(grouped_df)
        all_pattern.append(pattern_df)
        all_wilcox.append(wilcox_df)
        all_pairwise.append(pairwise_df)
        all_missing_profiles.append(missing_profile_df)
        dataset_metas.append(meta)
    # combined outputs
    summary_all = pd.concat([d for d in all_summary if d is not None and not d.empty], ignore_index=True) if any(d is not None and not d.empty for d in all_summary) else pd.DataFrame()
    runs_all = pd.concat([d for d in all_runs if d is not None and not d.empty], ignore_index=True) if any(d is not None and not d.empty for d in all_runs) else pd.DataFrame()
    feature_all = pd.concat([d for d in all_feature if d is not None and not d.empty], ignore_index=True) if any(d is not None and not d.empty for d in all_feature) else pd.DataFrame()
    grouped_all = pd.concat([d for d in all_grouped if d is not None and not d.empty], ignore_index=True) if any(d is not None and not d.empty for d in all_grouped) else pd.DataFrame()
    pattern_all = pd.concat([d for d in all_pattern if d is not None and not d.empty], ignore_index=True) if any(d is not None and not d.empty for d in all_pattern) else pd.DataFrame()
    wilcox_all = pd.concat([d for d in all_wilcox if d is not None and not d.empty], ignore_index=True) if any(d is not None and not d.empty for d in all_wilcox) else pd.DataFrame()
    pairwise_all = pd.concat([d for d in all_pairwise if d is not None and not d.empty], ignore_index=True) if any(d is not None and not d.empty for d in all_pairwise) else pd.DataFrame()
    missing_all = pd.concat([d for d in all_missing_profiles if d is not None and not d.empty], ignore_index=True) if any(d is not None and not d.empty for d in all_missing_profiles) else pd.DataFrame()
    runtime_all = compute_runtime_summary(runs_all)
    if not summary_all.empty:
        summary_all.to_csv(base_output_dir / "combined_summary_results.csv", index=False)
    if not runs_all.empty:
        runs_all.to_csv(base_output_dir / "combined_run_results.csv", index=False)
    if not feature_all.empty:
        feature_all.to_csv(base_output_dir / "combined_per_feature_results.csv", index=False)
    if not grouped_all.empty:
        grouped_all.to_csv(base_output_dir / "combined_grouped_summary_results.csv", index=False)
    if not pattern_all.empty:
        pattern_all.to_csv(base_output_dir / "combined_pattern_summary_results.csv", index=False)
    if not wilcox_all.empty:
        wilcox_all.to_csv(base_output_dir / "combined_wilcoxon_vs_reference.csv", index=False)
    if not pairwise_all.empty:
        pairwise_all.to_csv(base_output_dir / "combined_wilcoxon_pairwise_all_methods.csv", index=False)
    if not runtime_all.empty:
        runtime_all.to_csv(base_output_dir / "combined_runtime_summary_results.csv", index=False)
    if not missing_all.empty:
        missing_all.to_csv(base_output_dir / "combined_natural_missing_profile.csv", index=False)
        summarize_natural_missing(missing_all).to_csv(base_output_dir / "combined_natural_missing_summary.csv", index=False)
    # optional sensitivity on first dataset
    run_sensitivity_analysis(args, script_dir, base_output_dir)
    meta = {"output_dir": str(base_output_dir), "datasets": args.datasets, "dataset_metas": dataset_metas, "reference_method": args.reference_method}
    (base_output_dir / "metadata_combined.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[INFO] Combined results saved to: {base_output_dir}")
    return summary_all, feature_all, meta


def main():
    args = parse_args()
    try:
        run_experiment(args)
    except Exception as e:
        print(f"[ERROR] {e}")
        raise


if __name__ == "__main__":
    main()


