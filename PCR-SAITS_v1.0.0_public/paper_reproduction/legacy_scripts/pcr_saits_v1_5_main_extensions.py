#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCR-SAITS v1.5 — Revision-1 Extensions Patch
============================================
Extension layer on top of v7.1 paper-suite that adds the experiments
requested by ESWA reviewers (R1 and R4) for revision round 1.

This file is invoked as a thin wrapper that imports and reuses the
v7.1 functions for SAITS/BRITS/PCR training and scenario evaluation,
then adds the five additional studies below.

Experiments added in this patch
-------------------------------
E1  (R1.4)  Window size sensitivity:     W in {24, 48, 72, 96}
E2  (R1.9)  Hyperparameter sensitivity:  hidden in {32, 64, 128, 256}
                                          layers in {1, 2, 3}
E11 (R4.8)  Realistic synthetic data:    AR(1) + seasonal structure
E12 (R4.6)  Mechanism-aware training:    PCR holdout includes MAR+MNAR
E13 (R4.9)  Confidence-aware gating:     gate |delta| > tau on validation

All outputs are written under  output_revision1/<exp_tag>/  and every
experiment is checkpointed so a crashed run can resume by re-running
the same command. The original v7.1 outputs are not touched.

Author
------
Sawet Somnugpong, KPRU. June 2026.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import pickle
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Import v7.1 base module dynamically (without modifying it)
# ---------------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
V71_FILENAME_CANDIDATES = [
    "pcr_saits_v7_1_paper_suite.py",
    "1782529209295_pcr_saits_v7_1_paper_suite.py",
]


def _find_v71_module() -> Path:
    for name in V71_FILENAME_CANDIDATES:
        p = THIS_DIR / name
        if p.exists():
            return p
    for p in THIS_DIR.glob("*pcr_saits_v7_1*paper_suite*.py"):
        return p
    raise FileNotFoundError(
        "Could not locate v7.1 main runner. Place "
        "pcr_saits_v7_1_paper_suite.py in the same folder as this patch."
    )


_v71_path = _find_v71_module()
_spec = importlib.util.spec_from_file_location("pcr_v71", str(_v71_path))
_v71 = importlib.util.module_from_spec(_spec)
sys.modules["pcr_v71"] = _v71
_spec.loader.exec_module(_v71)

# Pull frequently used names into local scope
PCRResidualNet = _v71.PCRResidualNet
PCRSAITSV1CleanWrapper = _v71.PCRSAITSV1CleanWrapper
SAITSWrapper = _v71.SAITSWrapper
build_windows = _v71.build_windows
reconstruct_from_windows = _v71.reconstruct_from_windows
fit_standardizer = _v71.fit_standardizer
transform_values = _v71.transform_values
inverse_transform_values = _v71.inverse_transform_values
chronological_split = _v71.chronological_split
generate_artificial_mask = _v71.generate_artificial_mask
apply_mask = _v71.apply_mask
compute_metrics = _v71.compute_metrics
compute_ref_feature_map = _v71.compute_ref_feature_map
set_seed = _v71.set_seed
choose_device = _v71.choose_device
now_stamp = _v71.now_stamp
load_air_quality_dataset = _v71.load_air_quality_dataset
load_beijing_pm25_dataset = _v71.load_beijing_pm25_dataset
deterministic_scenario_seed = _v71.deterministic_scenario_seed


# ---------------------------------------------------------------------------
# Output / checkpoint root
# ---------------------------------------------------------------------------

REVISION_TAG = "output_revision1"


def revision_root(script_dir: Path) -> Path:
    root = script_dir / REVISION_TAG
    root.mkdir(parents=True, exist_ok=True)
    return root


def exp_dir(script_dir: Path, exp_tag: str) -> Path:
    d = revision_root(script_dir) / exp_tag
    (d / "model_checkpoints").mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Generic checkpoint helpers
# ---------------------------------------------------------------------------


def _ckpt_paths(out_dir: Path) -> Tuple[Path, Path]:
    return (out_dir / "checkpoint_results.csv",
            out_dir / "checkpoint_state.json")


def load_completed_keys(out_dir: Path) -> set:
    """Return the set of (seed, mechanism, pattern, rate, *) keys already done."""
    ck_csv, _ = _ckpt_paths(out_dir)
    if not ck_csv.exists():
        return set()
    try:
        df = pd.read_csv(ck_csv)
    except Exception:
        return set()
    keys = set()
    key_cols = [c for c in ("variant", "seed", "mechanism", "pattern", "rate",
                            "window", "hidden_dim", "n_layers",
                            "synthetic_mode", "holdout_mix",
                            "gating_tau", "dataset")
                if c in df.columns]
    for _, r in df.iterrows():
        keys.add(tuple(str(r[c]) for c in key_cols))
    return keys


def append_result_row(out_dir: Path, row: Dict[str, Any]) -> None:
    ck_csv, _ = _ckpt_paths(out_dir)
    df = pd.DataFrame([row])
    header = (not ck_csv.exists()) or ck_csv.stat().st_size == 0
    df.to_csv(ck_csv, mode="a", header=header, index=False)


def write_state(out_dir: Path, state: Dict[str, Any]) -> None:
    _, st = _ckpt_paths(out_dir)
    state = {**state,
             "updated_at": now_stamp(),
             "script": __file__}
    st.write_text(json.dumps(state, indent=2, default=str))


def make_key(*parts) -> Tuple[str, ...]:
    return tuple(str(p) for p in parts)


# ---------------------------------------------------------------------------
# Realistic synthetic data generators (E11, R4.8)
# ---------------------------------------------------------------------------


def generate_ar1_synthetic(n_rows: int = 5000,
                           n_features: int = 13,
                           seed: int = 1234,
                           phi: float = 0.85,
                           seasonal_period: int = 24,
                           seasonal_amp: float = 0.4,
                           cross_corr: float = 0.3,
                           ) -> Tuple[pd.DataFrame, List[str]]:
    """Generate a multivariate synthetic series with AR(1) dynamics and a
    diurnal seasonal component, replacing the original i.i.d.-noise synthetic
    used in the paper. Each feature has temporal autocorrelation phi and
    shares variance through a low-rank cross-feature factor.

    IMPORTANT: the columns use the SAME 13 UCI feature names that v7.1's
    synthetic loader expects (CO(GT), PT08.S1(CO), ... AH), so the domain
    group tags (v6, v7 = pollutant/sensor/meteo) are assigned correctly and
    the result is directly comparable to the paper's synthetic column.

    Returns a DataFrame with a 'timestamp' column plus the 13 feature
    columns, and the list of feature names.
    """
    rng = np.random.default_rng(seed)

    # The 13 UCI feature names, in the order v7.1 expects
    feature_names = [
        "CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)", "PT08.S2(NMHC)",
        "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)",
        "T", "RH", "AH",
    ]
    n_features = len(feature_names)

    # Per-feature rough scale/offset so the synthetic values sit in a
    # realistic range for each channel (loosely matched to UCI magnitudes).
    scale_offset = {
        "CO(GT)": (0.4, 2.0), "PT08.S1(CO)": (90, 1000),
        "NMHC(GT)": (35, 120), "C6H6(GT)": (3, 9),
        "PT08.S2(NMHC)": (60, 900), "NOx(GT)": (18, 80),
        "PT08.S3(NOx)": (40, 1100), "NO2(GT)": (12, 45),
        "PT08.S4(NO2)": (50, 1500), "PT08.S5(O3)": (60, 950),
        "T": (6, 22), "RH": (10, 55), "AH": (1.0, 8.0),
    }

    # Shared latent AR(1) factor providing cross-feature correlation
    factor = np.zeros(n_rows)
    eps_f = rng.standard_normal(n_rows)
    for t in range(1, n_rows):
        factor[t] = phi * factor[t - 1] + eps_f[t]

    cols = {}
    for f, fname in enumerate(feature_names):
        eps = rng.standard_normal(n_rows)
        x = np.zeros(n_rows)
        for t in range(1, n_rows):
            x[t] = phi * x[t - 1] + eps[t]
        phase = 2 * np.pi * f / max(1, n_features)
        season = seasonal_amp * np.sin(
            2 * np.pi * np.arange(n_rows) / seasonal_period + phase)
        signal = (1 - cross_corr) * x + cross_corr * factor + season
        amp, off = scale_offset[fname]
        cols[fname] = off + amp * signal

    ts = pd.date_range("2020-01-01", periods=n_rows, freq="h")
    df = pd.DataFrame({"timestamp": ts, **cols})

    # Inject light natural missingness (4%) to mirror the paper's synthetic
    for c in feature_names:
        miss = rng.random(n_rows) < 0.04
        df.loc[miss, c] = np.nan

    return df, feature_names


# ---------------------------------------------------------------------------
# Mechanism-aware holdout (E12, R4.6)
# ---------------------------------------------------------------------------


def build_mixed_holdout_mask(values,
                             seed: int,
                             holdout_ratio: float = 0.15,
                             ref_map: Optional[Dict[int, int]] = None,
                             block_lengths: Sequence[int] = (6, 12, 24, 48),
                             mechanisms: Sequence[str] = ("MCAR", "MAR", "MNAR"),
                             **_ignored,
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """Construct a holdout mask whose hidden positions come in equal
    proportion from MCAR-, MAR-, and MNAR-like patterns (Reviewer 4,
    Comment 6). Drop-in replacement for v7.1's make_holdout_train_mask, so
    it MUST match that signature (values, seed, ...) and return the pair
    (mask, gap_len) — the caller unpacks two values and uses gap_len to
    build the gap-length feature v5.

    The TOTAL fraction of observed positions hidden equals holdout_ratio
    (default 0.15), matched to v7.1's MCAR-only holdout so the mixed-vs-
    MCAR-only comparison in E12 is fair. That budget is split equally across
    the three mechanisms; within each mechanism it is split ~half pointwise,
    ~half block. Only observed positions are eligible, mirroring v7.1.
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    T, F = values.shape
    observed = np.isfinite(values)
    if ref_map is None:
        ref_map = compute_ref_feature_map(values)

    holdout_mask = np.zeros((T, F), dtype=bool)
    gap_len = np.zeros((T, F), dtype=float)

    total_observed = int(observed.sum())
    total_budget = max(1, int(round(total_observed * holdout_ratio)))
    per_mech_budget = max(1, total_budget // len(mechanisms))

    def _mech_probs(mech):
        """Per-position selection weights (flattened over observed cells)."""
        obs_idx = np.argwhere(observed)  # (N,2) rows of [t,f]
        if mech == "MCAR":
            w = np.ones(len(obs_idx))
        elif mech == "MAR":
            w = np.empty(len(obs_idx))
            for i, (t, f) in enumerate(obs_idx):
                rf = ref_map.get(int(f), (int(f) + 1) % F)
                z = values[t, rf]
                w[i] = 1.0 / (1.0 + np.exp(-1.0 * np.nan_to_num(z)))
        else:  # MNAR
            w = np.empty(len(obs_idx))
            for i, (t, f) in enumerate(obs_idx):
                z = values[t, f]
                w[i] = 1.0 / (1.0 + np.exp(-1.0 * np.nan_to_num(z)))
        w = w / max(w.sum(), 1e-9)
        return obs_idx, w

    for mech in mechanisms:
        point_budget = per_mech_budget // 2
        block_budget = per_mech_budget - point_budget

        # ---- pointwise portion: draw individual observed cells ----
        obs_idx, w = _mech_probs(mech)
        # exclude already-hidden cells from the draw
        avail = ~holdout_mask[obs_idx[:, 0], obs_idx[:, 1]]
        if avail.any() and point_budget > 0:
            pool = np.where(avail)[0]
            wp = w[pool]
            wp = wp / max(wp.sum(), 1e-9)
            k = min(point_budget, pool.size)
            pick = rng.choice(pool, size=k, replace=False, p=wp)
            rows = obs_idx[pick, 0]
            cols = obs_idx[pick, 1]
            holdout_mask[rows, cols] = True
            gap_len[rows, cols] = 1.0

        # ---- block portion: place contiguous spans until budget met ----
        placed = 0
        guard = 0
        while placed < block_budget and guard < 1000:
            guard += 1
            f = int(rng.integers(0, F))
            blen = int(rng.choice(block_lengths))
            if T <= blen + 4:
                continue
            start = int(rng.integers(2, T - blen - 2))
            sl = slice(start, start + blen)
            obs_span = observed[sl, f] & ~holdout_mask[sl, f]
            n_new = int(obs_span.sum())
            if n_new == 0:
                continue
            holdout_mask[sl, f] |= obs_span
            gap_len[sl, f] = np.where(obs_span, float(blen), gap_len[sl, f])
            placed += n_new

    return holdout_mask, gap_len


# ---------------------------------------------------------------------------
# Confidence-aware gating wrapper (E13, R4.9)
# ---------------------------------------------------------------------------


def apply_confidence_gate(base_pred: np.ndarray,
                          corrected_pred: np.ndarray,
                          disagreement: np.ndarray,
                          tau: float) -> np.ndarray:
    """Return final prediction where corrected value is used only where
    |disagreement| >= tau; otherwise fall back to the SAITS base prediction.
    Implements R4.9's recommendation that PCR abstain on low-disagreement
    positions where the backbone is already reliable.
    """
    use_corr = np.abs(disagreement) >= float(tau)
    out = np.where(use_corr, corrected_pred, base_pred)
    return out


def sweep_gating_tau(base_pred: np.ndarray,
                     corrected_pred: np.ndarray,
                     disagreement: np.ndarray,
                     y_true: np.ndarray,
                     eval_mask: np.ndarray,
                     tau_grid: Sequence[float],
                     ) -> Tuple[float, pd.DataFrame]:
    """Sweep tau over a validation set, return best-tau and full sweep table."""
    rows = []
    for tau in tau_grid:
        gated = apply_confidence_gate(base_pred, corrected_pred,
                                      disagreement, tau)
        m = compute_metrics(y_true, gated, eval_mask)
        rmse, mae = m["rmse"], m["mae"]
        rows.append({"tau": float(tau), "rmse": rmse, "mae": mae,
                     "n_corrected_used": int(np.sum(
                         (np.abs(disagreement) >= tau) & eval_mask))})
    df = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    best_tau = float(df.iloc[0]["tau"]) if not df.empty else 0.0
    return best_tau, df


# ---------------------------------------------------------------------------
# E1: Window size sensitivity
# ---------------------------------------------------------------------------


def run_E1_window_sensitivity(args, script_dir: Path) -> None:
    """Re-train PCR-SAITS with multiple SAITS window sizes W and measure how
    the proposed-method RMSE changes across a fixed scenario subset.
    """
    out_dir = exp_dir(script_dir, "E1_window_sensitivity")
    print(f"[E1] writing to {out_dir}")
    done_keys = load_completed_keys(out_dir)

    windows = [int(w) for w in str(args.window_grid).split(",") if w.strip()]
    sens_seed = int(args.sensitivity_seed)
    sens_dataset = args.sensitivity_dataset

    write_state(out_dir, {"experiment": "E1_window_sensitivity",
                          "windows": windows,
                          "seed": sens_seed,
                          "dataset": sens_dataset})

    for W in windows:
        key = make_key("E1", sens_dataset, sens_seed, W)
        if key in done_keys:
            print(f"[E1] skip W={W} (checkpointed)")
            continue
        print(f"[E1] running W={W} on dataset={sens_dataset} seed={sens_seed}")
        try:
            row = _run_proposed_one_config(
                script_dir=script_dir,
                dataset=sens_dataset,
                seed=sens_seed,
                window=W,
                hidden_dim=int(args.default_hidden_dim),
                n_layers=int(args.default_n_layers),
                synthetic_mode="gaussian",
                holdout_mix=False,
                ckpt_subdir=out_dir / "model_checkpoints" / f"W{W}",
                full_grid=True,  # match the paper's full 375-scenario protocol
            )
            row.update({"experiment": "E1", "window": W,
                        "hidden_dim": int(args.default_hidden_dim),
                        "n_layers": int(args.default_n_layers),
                        "seed": sens_seed, "dataset": sens_dataset})
            append_result_row(out_dir, row)
        except Exception as exc:
            print(f"[E1][ERROR] W={W}: {exc}")
            append_result_row(out_dir, {
                "experiment": "E1", "window": W,
                "seed": sens_seed, "dataset": sens_dataset,
                "error": str(exc)})


# ---------------------------------------------------------------------------
# E2: Hyperparameter sensitivity (hidden dim, n_layers)
# ---------------------------------------------------------------------------


def run_E2_hyperparameter_sensitivity(args, script_dir: Path) -> None:
    out_dir = exp_dir(script_dir, "E2_hyperparameter_sensitivity")
    print(f"[E2] writing to {out_dir}")
    done_keys = load_completed_keys(out_dir)

    hiddens = [int(h) for h in str(args.mlp_hidden_grid).split(",") if h.strip()]
    layers_grid = [int(l) for l in str(args.mlp_layers_grid).split(",") if l.strip()]
    sens_seed = int(args.sensitivity_seed)
    sens_dataset = args.sensitivity_dataset

    write_state(out_dir, {"experiment": "E2_hyperparameter_sensitivity",
                          "hidden_dim_grid": hiddens,
                          "n_layers_grid": layers_grid,
                          "seed": sens_seed,
                          "dataset": sens_dataset})

    for hd in hiddens:
        for nl in layers_grid:
            key = make_key("E2", sens_dataset, sens_seed, hd, nl)
            if key in done_keys:
                print(f"[E2] skip hidden={hd} layers={nl}")
                continue
            print(f"[E2] running hidden={hd} layers={nl}")
            try:
                row = _run_proposed_one_config(
                    script_dir=script_dir,
                    dataset=sens_dataset,
                    seed=sens_seed,
                    window=int(args.default_window),
                    hidden_dim=hd,
                    n_layers=nl,
                    synthetic_mode="gaussian",
                    holdout_mix=False,
                    ckpt_subdir=out_dir / "model_checkpoints" / f"h{hd}_l{nl}",
                    full_grid=True,  # match the paper's full 375-scenario protocol
                )
                # Parameter count for record
                n_params = _count_params(hidden_dim=hd, n_layers=nl,
                                         input_dim=7)
                row.update({"experiment": "E2",
                            "hidden_dim": hd, "n_layers": nl,
                            "n_params": n_params,
                            "window": int(args.default_window),
                            "seed": sens_seed, "dataset": sens_dataset})
                append_result_row(out_dir, row)
            except Exception as exc:
                print(f"[E2][ERROR] hidden={hd} layers={nl}: {exc}")
                append_result_row(out_dir, {
                    "experiment": "E2",
                    "hidden_dim": hd, "n_layers": nl,
                    "seed": sens_seed, "dataset": sens_dataset,
                    "error": str(exc)})


def _count_params(hidden_dim: int, n_layers: int, input_dim: int = 7) -> int:
    """Parameter count for an MLP with the given configuration plus the
    proposed tanh-bounded delta head. Matches the architecture used by
    PCRResidualNet (without the masked-residual gate)."""
    n = (input_dim * hidden_dim + hidden_dim)         # input layer
    n += (n_layers - 1) * (hidden_dim * hidden_dim + hidden_dim)
    n += hidden_dim + 1                               # delta head
    return int(n)


# ---------------------------------------------------------------------------
# E11: Realistic synthetic data
# ---------------------------------------------------------------------------


def run_E11_realistic_synthetic(args, script_dir: Path) -> None:
    out_dir = exp_dir(script_dir, "E11_ar1_synthetic")
    print(f"[E11] writing to {out_dir}")
    done_keys = load_completed_keys(out_dir)

    seed = int(args.sensitivity_seed)
    key = make_key("E11", "ar1_synthetic", seed)
    if key in done_keys:
        print(f"[E11] skip (checkpointed)")
        return

    # Generate AR(1) + seasonal data with UCI feature names, save as CSV.
    # Note: n_features is fixed to the 13 UCI channels inside the generator
    # so the domain-group tags are assigned correctly; the CLI value is not
    # used for the column set.
    df_syn, feature_names = generate_ar1_synthetic(
        n_rows=int(args.ar1_n_rows),
        seed=seed,
        phi=float(args.ar1_phi),
        seasonal_period=int(args.ar1_period),
        seasonal_amp=float(args.ar1_seasonal_amp),
    )
    out_csv = out_dir / "synthetic_ar1.csv"
    df_syn.to_csv(out_csv, index=False)
    print(f"[E11] generated AR(1) synthetic at {out_csv} "
          f"(shape={df_syn.shape}, phi={args.ar1_phi})")

    write_state(out_dir, {"experiment": "E11_ar1_synthetic",
                          "phi": float(args.ar1_phi),
                          "seasonal_period": int(args.ar1_period),
                          "n_rows": int(args.ar1_n_rows),
                          "n_features": len(feature_names),
                          "data_path": str(out_csv)})

    try:
        row = _run_proposed_one_config(
            script_dir=script_dir,
            dataset="synthetic_ar1",
            seed=seed,
            window=int(args.default_window),
            hidden_dim=int(args.default_hidden_dim),
            n_layers=int(args.default_n_layers),
            synthetic_mode="ar1",
            holdout_mix=False,
            ckpt_subdir=out_dir / "model_checkpoints",
            ar1_csv_path=str(out_csv),
            full_grid=True,  # E11 must match the main Table 3 protocol
            all_methods=True,  # run Linear/Seasonal/BRITS/SAITS/PCR like Table 3
        )
        row.update({"experiment": "E11", "synthetic_mode": "ar1",
                    "seed": seed, "dataset": "synthetic_ar1"})
        append_result_row(out_dir, row)
    except Exception as exc:
        print(f"[E11][ERROR] {exc}")
        append_result_row(out_dir, {"experiment": "E11",
                                    "synthetic_mode": "ar1",
                                    "seed": seed,
                                    "dataset": "synthetic_ar1",
                                    "error": str(exc)})


# ---------------------------------------------------------------------------
# E12: Mechanism-aware training (mixed-mechanism holdout)
# ---------------------------------------------------------------------------


def run_E12_mechanism_aware_training(args, script_dir: Path) -> None:
    out_dir = exp_dir(script_dir, "E12_mechanism_aware_training")
    print(f"[E12] writing to {out_dir}")
    done_keys = load_completed_keys(out_dir)

    seed = int(args.sensitivity_seed)
    dataset = args.sensitivity_dataset

    write_state(out_dir, {"experiment": "E12_mechanism_aware_training",
                          "seed": seed, "dataset": dataset,
                          "holdout_mechanisms": ["MCAR", "MAR", "MNAR"]})

    for mix in (False, True):
        tag = "mix" if mix else "mcar_only"
        key = make_key("E12", dataset, seed, tag)
        if key in done_keys:
            print(f"[E12] skip {tag}")
            continue
        try:
            row = _run_proposed_one_config(
                script_dir=script_dir,
                dataset=dataset,
                seed=seed,
                window=int(args.default_window),
                hidden_dim=int(args.default_hidden_dim),
                n_layers=int(args.default_n_layers),
                synthetic_mode="gaussian",
                holdout_mix=mix,
                ckpt_subdir=out_dir / "model_checkpoints" / tag,
                full_grid=True,  # E12 needs per-mechanism MNAR disaggregation
            )
            row.update({"experiment": "E12", "holdout_mix": mix,
                        "seed": seed, "dataset": dataset})
            append_result_row(out_dir, row)
        except Exception as exc:
            print(f"[E12][ERROR] mix={mix}: {exc}")
            append_result_row(out_dir, {"experiment": "E12",
                                        "holdout_mix": mix,
                                        "seed": seed, "dataset": dataset,
                                        "error": str(exc)})


# ---------------------------------------------------------------------------
# E13: Confidence-aware gating (post-hoc, uses saved per-position outputs)
# ---------------------------------------------------------------------------


def run_E13_confidence_gating(args, script_dir: Path) -> None:
    """Sweep gating threshold tau on the validation predictions stored by the
    base run, then evaluate on the test set with the validation-selected tau.
    This requires per-position predictions saved by main runner; if they are
    not available, this routine writes a stub explaining what to enable.
    """
    out_dir = exp_dir(script_dir, "E13_confidence_gating")
    print(f"[E13] writing to {out_dir}")
    done_keys = load_completed_keys(out_dir)

    seed = int(args.sensitivity_seed)
    dataset = args.sensitivity_dataset
    tau_grid_raw = str(args.gating_tau_grid)
    tau_grid = [float(t) for t in tau_grid_raw.split(",") if t.strip()]

    write_state(out_dir, {"experiment": "E13_confidence_gating",
                          "tau_grid": tau_grid,
                          "seed": seed, "dataset": dataset})

    pred_dir = Path(args.per_position_dir) if args.per_position_dir else None
    if pred_dir is None or not pred_dir.exists():
        msg = ("E13 requires per-position predictions stored under "
               "--per-position-dir. Run the main runner with "
               "--save-per-position to enable this study.")
        print(f"[E13][WARN] {msg}")
        (out_dir / "WHY_NO_OUTPUT.txt").write_text(msg)
        return

    # Load the val + test predictions stored by main runner
    try:
        with open(pred_dir / "val_predictions.pkl", "rb") as fh:
            val_blob = pickle.load(fh)
        with open(pred_dir / "test_predictions.pkl", "rb") as fh:
            tst_blob = pickle.load(fh)
    except Exception as exc:
        print(f"[E13][ERROR] cannot load per-position predictions: {exc}")
        return

    # Validation sweep -> select best tau
    best_tau, sweep_df = sweep_gating_tau(
        base_pred=val_blob["base_pred"],
        corrected_pred=val_blob["pcr_pred"],
        disagreement=val_blob["disagreement"],
        y_true=val_blob["y_true"],
        eval_mask=val_blob["eval_mask"],
        tau_grid=tau_grid)
    sweep_df.to_csv(out_dir / "validation_sweep.csv", index=False)
    print(f"[E13] best tau on validation: {best_tau:.4f}")

    # Apply selected tau on test set
    gated_test = apply_confidence_gate(
        base_pred=tst_blob["base_pred"],
        corrected_pred=tst_blob["pcr_pred"],
        disagreement=tst_blob["disagreement"],
        tau=best_tau)

    _m_base = compute_metrics(tst_blob["y_true"], tst_blob["base_pred"],
                              tst_blob["eval_mask"])
    _m_pcr = compute_metrics(tst_blob["y_true"], tst_blob["pcr_pred"],
                             tst_blob["eval_mask"])
    _m_gated = compute_metrics(tst_blob["y_true"], gated_test,
                               tst_blob["eval_mask"])
    rmse_base, mae_base = _m_base["rmse"], _m_base["mae"]
    rmse_pcr, mae_pcr = _m_pcr["rmse"], _m_pcr["mae"]
    rmse_gated, mae_gated = _m_gated["rmse"], _m_gated["mae"]

    key = make_key("E13", dataset, seed, f"tau{best_tau:.4f}")
    if key in done_keys:
        print("[E13] result already checkpointed")
        return
    append_result_row(out_dir, {
        "experiment": "E13",
        "dataset": dataset, "seed": seed,
        "gating_tau": best_tau,
        "rmse_saits": rmse_base, "mae_saits": mae_base,
        "rmse_pcr": rmse_pcr, "mae_pcr": mae_pcr,
        "rmse_gated": rmse_gated, "mae_gated": mae_gated,
        "delta_rmse_pcr_vs_saits_pct": 100 * (rmse_base - rmse_pcr) / rmse_base,
        "delta_rmse_gated_vs_saits_pct": 100 * (rmse_base - rmse_gated) / rmse_base,
    })


# ---------------------------------------------------------------------------
# Inner runner that re-uses v7.1 logic for ONE (dataset, seed, hyperparam) cell
# ---------------------------------------------------------------------------


def _run_proposed_one_config(script_dir: Path,
                             dataset: str,
                             seed: int,
                             window: int,
                             hidden_dim: int,
                             n_layers: int,
                             synthetic_mode: str,
                             holdout_mix: bool,
                             ckpt_subdir: Path,
                             ar1_csv_path: Optional[str] = None,
                             full_grid: bool = False,
                             all_methods: bool = False,
                             ) -> Dict[str, Any]:
    """Run pcrsaitsv14_no_seasonal_branch on a single (dataset, seed) with
    the given window and MLP geometry. Returns a result dictionary with the
    aggregate RMSE/MAE over the chosen scenario grid.

    full_grid selects between the reduced sensitivity grid (E1/E2) and the
    full paper protocol (E11/E12/E13); see _build_inner_namespace.

    The proposed method is patched to use the requested hidden_dim and
    n_layers via PCRResidualNet's existing kwargs.
    """
    ckpt_subdir.mkdir(parents=True, exist_ok=True)

    # Build v7.1-compatible argparse Namespace
    base_args = _build_inner_namespace(
        script_dir=script_dir,
        dataset=dataset,
        seed=seed,
        window=window,
        ar1_csv_path=ar1_csv_path,
        ckpt_subdir=ckpt_subdir,
        full_grid=full_grid,
        all_methods=all_methods,
    )

    # Monkey-patch PCRResidualNet to take new hidden_dim / n_layers from us
    import torch.nn as nn

    class _PatchedPCRResidualNet(nn.Module):
        def __init__(self, input_dim, hidden_dim=hidden_dim, _hd=hidden_dim, _nl=n_layers):
            super().__init__()
            layers = []
            in_dim = input_dim
            for _ in range(_nl):
                layers.append(nn.Linear(in_dim, _hd))
                layers.append(nn.ReLU())
                in_dim = _hd
            self.backbone = nn.Sequential(*layers)
            self.delta_head = nn.Linear(_hd, 1)
            self.mask_head = nn.Linear(_hd, 1)
            for m in self.backbone:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    nn.init.zeros_(m.bias)
            nn.init.zeros_(self.delta_head.weight)
            nn.init.zeros_(self.delta_head.bias)
            nn.init.zeros_(self.mask_head.weight)
            nn.init.constant_(self.mask_head.bias, -1.0)

        def forward(self, x, direct_residual=False):
            import torch
            h = self.backbone(x)
            delta = torch.tanh(self.delta_head(h)) * 4.0
            corr_mask = (torch.ones_like(delta) if direct_residual
                         else torch.sigmoid(self.mask_head(h)))
            return delta, corr_mask

    _v71.PCRResidualNet = _PatchedPCRResidualNet

    # Monkey-patch make_holdout_train_mask for mixed-mechanism mode (E12).
    # v7.1 calls make_holdout_train_mask(values, seed=..., holdout_ratio=...)
    # and unpacks (mask, gap_len); our replacement mirrors that exactly.
    if holdout_mix:
        _original_holdout = _v71.make_holdout_train_mask

        def _mixed_holdout(values, seed, holdout_ratio=0.15, **kw):
            return build_mixed_holdout_mask(values, seed=seed,
                                            holdout_ratio=holdout_ratio)

        _v71.make_holdout_train_mask = _mixed_holdout

    # Monkey-patch the synthetic loader for E11 so the dataset name
    # "synthetic" actually loads our AR(1)+seasonal CSV instead of v7.1's
    # built-in sin/cos generator. We patch BOTH the synthetic loader and the
    # dispatch wrapper, AND clear the module-level dataset cache, because
    # v7.1 caches the first-loaded dataset in _ACTIVE_DATASET_CACHE and would
    # otherwise return the stale built-in synthetic on the inner reload.
    _patched_synth = (synthetic_mode == "ar1" and ar1_csv_path)
    if _patched_synth:
        _original_load_synth = _v71.load_synthetic_dataset
        _original_load_aq = _v71.load_air_quality_dataset
        _ar1_path = ar1_csv_path

        def _read_ar1_csv():
            sdf = pd.read_csv(_ar1_path)
            if "timestamp" in sdf.columns:
                sdf["timestamp"] = pd.to_datetime(sdf["timestamp"],
                                                  errors="coerce")
            feats = [c for c in sdf.columns if c != "timestamp"]
            # AR(1) diagnostic: mean lag-1 autocorrelation across features.
            # Gaussian i.i.d. synthetic gives ~0; AR(1) gives ~phi (~0.85).
            try:
                acs = []
                for c in feats:
                    x = sdf[c].dropna().values
                    if len(x) > 5:
                        acs.append(np.corrcoef(x[:-1], x[1:])[0, 1])
                mean_lag1 = float(np.nanmean(acs)) if acs else float("nan")
            except Exception:
                mean_lag1 = float("nan")
            print(f"[E11][LOADER-CHECK] using AR(1)+seasonal CSV: {_ar1_path} "
                  f"| shape={sdf.shape} | {len(feats)} feats "
                  f"| mean_lag1={mean_lag1:.3f}")
            return sdf, feats

        def _load_ar1_synth(path, n_rows=5000, seed=1234):
            return _read_ar1_csv()

        # Dispatch wrapper: whatever path v7.1 resolves for "synthetic",
        # force it to our AR(1) CSV. For non-synthetic paths, fall back to
        # the ORIGINAL underlying loader (not the dispatch wrapper) to avoid
        # infinite recursion.
        _real_aq_loader = getattr(_v71, "_ORIG_LOAD_AIR_QUALITY_DATASET",
                                  _original_load_aq)

        def _load_aq_dispatch(path):
            name = str(getattr(path, "name", path)).lower()
            if name.endswith(".synthetic") or "__synthetic__" in name \
               or "synthetic" in name:
                return _read_ar1_csv()
            # Non-synthetic: reproduce v7.1's own dispatch for beijing/uci
            if "beijing" in name or "prsa" in name:
                return _v71.load_beijing_pm25_dataset(path)
            return _real_aq_loader(path)

        _v71.load_synthetic_dataset = _load_ar1_synth
        _v71.load_air_quality_dataset = _load_aq_dispatch
        # CRITICAL: clear the cached dataset so the inner reload does not
        # return a previously cached (built-in) synthetic frame.
        _v71._ACTIVE_DATASET_CACHE = None

    try:
        summary_df, feature_df, meta = _v71.run_experiment(base_args)
    finally:
        # Restore originals
        _v71.PCRResidualNet = PCRResidualNet
        if holdout_mix:
            _v71.make_holdout_train_mask = _original_holdout
        if _patched_synth:
            _v71.load_synthetic_dataset = _original_load_synth
            _v71.load_air_quality_dataset = _original_load_aq
            _v71._ACTIVE_DATASET_CACHE = None

    # Extract aggregate metrics for the proposed method and the SAITS baseline.
    # v7.1 summary_df columns are:
    #   method, overall_rmse_mean, overall_rmse_std,
    #   overall_mae_mean, overall_mae_std, ...,
    #   rmse_improvement_vs_reference_pct_mean,
    #   mae_improvement_vs_reference_pct_mean
    if summary_df is None or summary_df.empty:
        return {"warning": "empty summary_df"}

    def _col(df, *candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    rmse_col = _col(summary_df, "overall_rmse_mean", "rmse")
    mae_col = _col(summary_df, "overall_mae_mean", "mae")
    imp_col = _col(summary_df, "rmse_improvement_vs_reference_pct_mean")
    imp_mae_col = _col(summary_df, "mae_improvement_vs_reference_pct_mean")

    prop = summary_df[summary_df["method"] == "pcrsaitsv14_no_seasonal_branch"]
    saits = summary_df[summary_df["method"] == "saits"]

    row = {"n_scenarios_method_rows": int(len(prop))}
    if not prop.empty and rmse_col:
        row["rmse_pcr"] = float(prop[rmse_col].iloc[0])
    if not prop.empty and mae_col:
        row["mae_pcr"] = float(prop[mae_col].iloc[0])
    if not saits.empty and rmse_col:
        row["rmse_saits"] = float(saits[rmse_col].iloc[0])
    if not saits.empty and mae_col:
        row["mae_saits"] = float(saits[mae_col].iloc[0])

    # When the full Table 3 method set was run (E11), also record the other
    # baselines so the new synthetic column can be reported in full.
    for mname, prefix in [("linear", "linear"),
                          ("seasonal_naive24", "seasonal"),
                          ("brits", "brits")]:
        msub = summary_df[summary_df["method"] == mname]
        if not msub.empty and rmse_col:
            row[f"rmse_{prefix}"] = float(msub[rmse_col].iloc[0])
        if not msub.empty and mae_col:
            row[f"mae_{prefix}"] = float(msub[mae_col].iloc[0])

    # Prefer v7.1's own per-scenario improvement (matches the paper's Delta).
    if not prop.empty and imp_col and pd.notna(prop[imp_col].iloc[0]):
        row["delta_rmse_pct"] = float(prop[imp_col].iloc[0])
    elif "rmse_pcr" in row and "rmse_saits" in row and row["rmse_saits"] > 0:
        # Fallback: aggregate-level delta
        row["delta_rmse_pct"] = (
            100 * (row["rmse_saits"] - row["rmse_pcr"]) / row["rmse_saits"]
        )
    if not prop.empty and imp_mae_col and pd.notna(prop[imp_mae_col].iloc[0]):
        row["delta_mae_pct"] = float(prop[imp_mae_col].iloc[0])

    return row


def _build_inner_namespace(script_dir: Path,
                           dataset: str,
                           seed: int,
                           window: int,
                           ar1_csv_path: Optional[str],
                           ckpt_subdir: Path,
                           full_grid: bool = False,
                           all_methods: bool = False,
                           ) -> argparse.Namespace:
    """Build a namespace compatible with v7.1's run_experiment by deriving
    all defaults from the v7.1 parser, then overriding only the fields we
    need for the revision experiments. This guarantees every attribute the
    v7.1 code expects is present (output_prefix, saits_d_model,
    pcr_weight_decay, etc.).

    Scenario grid depends on full_grid:
      * full_grid=True (default for all revision experiments E1, E2,
        E11, E12): the full paper protocol
            3 mechanisms x 5 patterns x 5 rates x 5 seeds = 375 cells
            per (dataset, config).
        Using the same grid as the main results table means every number
        the sensitivity studies report is directly comparable to Table 3,
        leaving no gap for a reviewer to question.
      * full_grid=False: a reduced 12-cell grid (kept only as a fast
        smoke-test option; not used by the reviewer-facing experiments).
    """
    # Map dataset alias to v7.1 dataset key
    if dataset.lower() == "synthetic_ar1":
        v71_dataset = "synthetic"
    elif dataset.lower() in ("uci_air_quality", "airquality"):
        v71_dataset = "airquality"
    elif dataset.lower() in ("beijing_pm25", "beijingpm25"):
        v71_dataset = "beijingpm25"
    else:
        v71_dataset = dataset

    # Derive defaults from v7.1 parser
    v71_parser = _v71.build_parser()
    ns = v71_parser.parse_args([])

    # Override only the fields the revision experiments need to control
    ns.script_dir = str(script_dir)
    ns.output_dir = str(ckpt_subdir)
    ns.datasets = v71_dataset
    # Method set: sensitivity studies (E1/E2/E12) only need SAITS + proposed,
    # but E11 must reproduce the full Table 3 method set so the new AR(1)
    # synthetic column is directly comparable to the original paper.
    if all_methods:
        ns.methods = ("linear,seasonal_naive24,brits,saits,"
                      "pcrsaitsv14_no_seasonal_branch")
    else:
        ns.methods = "saits,pcrsaitsv14_no_seasonal_branch"
    ns.window = int(window)

    if full_grid:
        ns.seeds = "7,21,42,123,456"
        ns.mechanisms = "MCAR,MAR,MNAR"
        ns.patterns = "pointwise,block_6,block_12,block_24,block_48"
        ns.rates = "0.1,0.2,0.3,0.4,0.5"
    else:
        ns.seeds = str(int(seed))
        ns.mechanisms = "MCAR,MAR,MNAR"
        ns.patterns = "pointwise,block_24"
        ns.rates = "0.1,0.3"

    # Mimic v7.1 parse_args() coercion: string -> list of typed values
    ns.datasets = [s.strip() for s in str(ns.datasets).split(',') if s.strip()]
    ns.methods = [m.strip() for m in str(ns.methods).split(',') if m.strip()]
    ns.seeds = [int(s.strip()) for s in str(ns.seeds).split(',') if s.strip()]
    ns.mechanisms = [s.strip().upper()
                     for s in str(ns.mechanisms).split(',') if s.strip()]
    ns.patterns = [s.strip()
                   for s in str(ns.patterns).split(',') if s.strip()]
    ns.rates = [float(s.strip())
                for s in str(ns.rates).split(',') if s.strip()]
    ns.val_mechanism = str(ns.val_mechanism).upper()
    ns.case_mechanism = str(ns.case_mechanism).upper()

    # Allow shrinking SAITS/BRITS epochs via env vars for the deadline
    if os.environ.get("SAITS_EPOCHS"):
        ns.saits_epochs = int(os.environ["SAITS_EPOCHS"])
    if os.environ.get("BRITS_EPOCHS"):
        ns.brits_epochs = int(os.environ["BRITS_EPOCHS"])

    ns.case_seed = int(seed)
    ns.case_proposed_method = "pcrsaitsv14_no_seasonal_branch"
    ns.reset_checkpoints = False
    ns.retrain_models = False
    ns.quiet = True

    # E11: custom AR(1) synthetic csv path (the inner runner is patched
    # to honour this attribute via dataset loader monkey-patch)
    if ar1_csv_path:
        ns.synthetic_csv_path = ar1_csv_path

    return ns


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PCR-SAITS v1.5 revision-1 extensions (E1, E2, E11, E12, E13).")

    # Which experiments to run
    p.add_argument("--run-E1", action="store_true",
                   help="Window size sensitivity (R1.4)")
    p.add_argument("--run-E2", action="store_true",
                   help="Hyperparameter sensitivity (R1.9)")
    p.add_argument("--run-E11", action="store_true",
                   help="AR(1) realistic synthetic (R4.8)")
    p.add_argument("--run-E12", action="store_true",
                   help="Mechanism-aware training (R4.6)")
    p.add_argument("--run-E13", action="store_true",
                   help="Confidence-aware gating (R4.9)")
    p.add_argument("--run-all", action="store_true",
                   help="Run E1, E2, E11, E12 (E13 requires per-position dir)")

    # Shared defaults
    p.add_argument("--sensitivity-seed", type=int, default=7)
    p.add_argument("--sensitivity-dataset", type=str, default="uci_air_quality",
                   choices=["uci_air_quality", "beijing_pm25"])
    p.add_argument("--default-window", type=int, default=48)
    p.add_argument("--default-hidden-dim", type=int, default=64)
    p.add_argument("--default-n-layers", type=int, default=2)

    # E1 (window sensitivity)
    p.add_argument("--window-grid", type=str, default="24,48,72,96")

    # E2 (hyperparameter sensitivity)
    p.add_argument("--mlp-hidden-grid", type=str, default="32,64,128,256")
    p.add_argument("--mlp-layers-grid", type=str, default="1,2,3")

    # E11 (AR(1) synthetic)
    p.add_argument("--ar1-n-rows", type=int, default=5000)
    p.add_argument("--ar1-n-features", type=int, default=13)
    p.add_argument("--ar1-phi", type=float, default=0.85)
    p.add_argument("--ar1-period", type=int, default=24)
    p.add_argument("--ar1-seasonal-amp", type=float, default=0.4)

    # E13 (confidence gating)
    p.add_argument("--gating-tau-grid", type=str,
                   default="0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50")
    p.add_argument("--per-position-dir", type=str, default="",
                   help="Folder containing val_predictions.pkl and "
                        "test_predictions.pkl from main runner with "
                        "--save-per-position")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    script_dir = THIS_DIR

    if args.run_all:
        args.run_E1 = args.run_E2 = args.run_E11 = args.run_E12 = True

    if not any([args.run_E1, args.run_E2, args.run_E11, args.run_E12,
                args.run_E13]):
        print("Nothing to do. Use --run-all or one of --run-E1/E2/E11/E12/E13.")
        parser.print_help()
        return

    root = revision_root(script_dir)
    summary = {
        "session_started": now_stamp(),
        "script": __file__,
        "output_root": str(root),
        "device": choose_device(),
    }
    (root / "session_state.json").write_text(json.dumps(summary, indent=2))

    if args.run_E1:
        run_E1_window_sensitivity(args, script_dir)
    if args.run_E2:
        run_E2_hyperparameter_sensitivity(args, script_dir)
    if args.run_E11:
        run_E11_realistic_synthetic(args, script_dir)
    if args.run_E12:
        run_E12_mechanism_aware_training(args, script_dir)
    if args.run_E13:
        run_E13_confidence_gating(args, script_dir)

    print(f"\n[DONE] All requested experiments finished. Outputs in {root}")


if __name__ == "__main__":
    main()