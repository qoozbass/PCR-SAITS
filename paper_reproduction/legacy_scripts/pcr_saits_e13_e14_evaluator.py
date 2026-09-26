#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCR-SAITS Revision-1 — E13 + E14 Evaluator
============================================
Reads per-position parquet files exported by pcr_saits_v7_1b_per_position.py
and computes the two remaining reviewer experiments:

  E13 (R4.9)  Confidence-gating: sweep threshold tau over abs_delta from
               validation parquets, select optimal tau, apply to test parquets,
               report RMSE improvement vs un-gated SAITS and PCR-SAITS.

  E14 (R4.12) Downstream forecasting: use complete imputed series from test
               parquets (saits_pred / pcr_pred) as input to a simple linear
               forecaster; compare MAE/RMSE of downstream forecasts between
               SAITS-imputed and PCR-imputed inputs.

Usage
-----
python pcr_saits_e13_e14_evaluator.py \\
    --per-position-dir  output_revision1/E13_E14_per_position \\
    --output-dir        output_revision1/E13_E14_results \\
    --run-E13 --run-E14

Author
------
Sawet Somnugpong, KPRU. 2026.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

RUN_CODE_VERSION = "e13e14_evaluator_v2_e14c"

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_parquets(pp_dir: Path, split: str) -> pd.DataFrame:
    """Load and concatenate all per-position parquets (or CSV fallback) for a
    given split (split='val' for E13 tau selection, split='test' for evaluation).
    Supports both .parquet and .csv files so CSV fallback from the exporter
    is also readable."""
    frames = []
    for ds_subdir in sorted(pp_dir.iterdir()):
        if not ds_subdir.is_dir():
            continue
        pp_sub = ds_subdir / "per_position"
        if not pp_sub.exists():
            continue
        # Bug fix: also read CSV fallback files (exporter writes .csv when parquet fails).
        # Prefer parquet over csv when both exist for the same stem, to avoid double-
        # counting rows (e.g. an earlier CSV-fallback run followed by a later successful
        # parquet run left both files on disk with the same scenario name).
        parquet_by_stem = {p.stem: p for p in pp_sub.glob("*.parquet")}
        csv_by_stem = {p.stem: p for p in pp_sub.glob("*.csv")}
        files = list(parquet_by_stem.values())
        files += [p for stem, p in csv_by_stem.items() if stem not in parquet_by_stem]
        files = sorted(files)
        for p in files:
            if split == "val" and "_val_" not in p.name:
                continue
            if split == "test" and "_val_" in p.name:
                continue
            try:
                if p.suffix.lower() == ".parquet":
                    df = pd.read_parquet(p)
                else:
                    df = pd.read_csv(p)
                # Inject dataset name from subdir if column missing
                if "dataset" not in df.columns:
                    df["dataset"] = ds_subdir.name
                frames.append(df)
            except Exception as exc:
                print(f"[WARN] could not load {p}: {exc}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ---------------------------------------------------------------------------
# E13 — Confidence Gating
# ---------------------------------------------------------------------------

def run_E13(pp_dir: Path, out_dir: Path, tau_n_grid: int = 50) -> None:
    """E13 R4.9: select optimal gating threshold tau from validation,
    evaluate on test, report gated vs un-gated RMSE."""
    print("\n[E13] Loading val parquets for tau selection...")
    val_df = _load_parquets(pp_dir, split="val")
    if val_df.empty:
        raise RuntimeError("No val parquets found. Run v7.1b with --save-per-position first.")

    # Restrict to artificial missing positions only for RMSE computation
    val_ev = val_df[val_df["is_artificial_missing"] == 1].copy()
    if val_ev.empty:
        raise RuntimeError("Val data has no artificial missing positions (is_artificial_missing=1).")

    # Bug fix: drop rows with non-finite y_true/saits_pred/pcr_pred/abs_delta.
    # A stray NaN/Inf prediction (e.g. from a diverged training run) would
    # otherwise silently poison RMSE and the tau sweep via np.mean/np.sqrt.
    _e13_cols = ["y_true", "saits_pred", "pcr_pred", "abs_delta"]
    _val_finite = np.isfinite(val_ev[_e13_cols].to_numpy(dtype=float)).all(axis=1)
    _n_val_dropped = int((~_val_finite).sum())
    if _n_val_dropped:
        print(f"[E13][WARN] dropping {_n_val_dropped:,} val rows with non-finite values")
    val_ev = val_ev[_val_finite].copy()
    if val_ev.empty:
        raise RuntimeError("All val rows dropped as non-finite; check the per-position export.")

    print(f"[E13] val rows for tau sweep: {len(val_ev):,}")

    # Tau grid: low tau → only very small-|δ| positions use PCR (conservative);
    # high tau → nearly all positions use PCR (close to un-gated PCR).
    # np.inf endpoint ensures the grid covers "always use PCR" as the upper bound.
    tau_max = float(val_ev["abs_delta"].max())
    tau_grid = np.unique(np.r_[np.linspace(0.0, tau_max, tau_n_grid), np.inf])

    # For each tau: use PCR where abs_delta <= tau; otherwise fallback to SAITS.
    # Compute RMSE improvement vs SAITS baseline on val
    best_tau = 0.0
    best_val_rmse = float("inf")
    val_saits_rmse = _rmse(val_ev["y_true"].values, val_ev["saits_pred"].values)
    val_pcr_rmse   = _rmse(val_ev["y_true"].values, val_ev["pcr_pred"].values)

    # Bug fix 1: gating logic was reversed.
    # abs_delta = |PCR - SAITS| = size of correction.
    # LOW abs_delta → small correction, high confidence → USE PCR.
    # HIGH abs_delta → large disagreement, high risk → FALLBACK to SAITS.
    # Tau sweep: find the threshold that maximises improvement by using PCR
    # only when abs_delta <= tau (confident corrections only).
    tau_rows = []
    for tau in tau_grid:
        use_pcr = val_ev["abs_delta"].values <= tau
        gated_pred = np.where(use_pcr,
                               val_ev["pcr_pred"].values,
                               val_ev["saits_pred"].values)
        r = _rmse(val_ev["y_true"].values, gated_pred)
        improve_vs_saits = 100.0 * (val_saits_rmse - r) / val_saits_rmse
        tau_rows.append({"tau": tau, "val_rmse": r,
                          "improve_vs_saits_pct": improve_vs_saits,
                          "pcr_used_rate_pct": 100.0 * use_pcr.mean()})
        if r < best_val_rmse:
            best_val_rmse = r
            best_tau = tau

    tau_df = pd.DataFrame(tau_rows)
    print(f"[E13] best tau={best_tau:.4f} → val RMSE={best_val_rmse:.4f} "
          f"(SAITS={val_saits_rmse:.4f}, PCR={val_pcr_rmse:.4f})")

    # --- Apply selected tau to test set ---
    print("[E13] Loading test parquets for evaluation...")
    test_df = _load_parquets(pp_dir, split="test")
    if test_df.empty:
        raise RuntimeError("No test parquets found.")
    test_ev = test_df[test_df["is_artificial_missing"] == 1].copy()

    _test_finite = np.isfinite(test_ev[_e13_cols].to_numpy(dtype=float)).all(axis=1)
    _n_test_dropped = int((~_test_finite).sum())
    if _n_test_dropped:
        print(f"[E13][WARN] dropping {_n_test_dropped:,} test rows with non-finite values")
    test_ev = test_ev[_test_finite].copy()
    if test_ev.empty:
        raise RuntimeError("All test rows dropped as non-finite; check the per-position export.")

    print(f"[E13] test rows: {len(test_ev):,}")

    test_saits_rmse = _rmse(test_ev["y_true"].values, test_ev["saits_pred"].values)
    test_pcr_rmse   = _rmse(test_ev["y_true"].values, test_ev["pcr_pred"].values)
    # Same logic as val: abs_delta <= tau → use PCR, > tau → fallback SAITS
    use_pcr_test = test_ev["abs_delta"].values <= best_tau
    gated_pred_test = np.where(use_pcr_test,
                                test_ev["pcr_pred"].values,
                                test_ev["saits_pred"].values)
    test_gated_rmse = _rmse(test_ev["y_true"].values, gated_pred_test)
    improve_vs_saits  = 100.0 * (test_saits_rmse - test_gated_rmse) / test_saits_rmse
    improve_vs_pcr    = 100.0 * (test_pcr_rmse   - test_gated_rmse) / test_pcr_rmse
    pcr_used_rate     = 100.0 * use_pcr_test.mean()
    saits_fallback_rate = 100.0 * (1.0 - use_pcr_test.mean())

    # Per-dataset breakdown
    per_ds_rows = []
    for ds, grp in test_ev.groupby("dataset"):
        ds_saits = _rmse(grp["y_true"].values, grp["saits_pred"].values)
        ds_pcr   = _rmse(grp["y_true"].values, grp["pcr_pred"].values)
        ds_use_pcr = grp["abs_delta"].values <= best_tau
        ds_gated = np.where(ds_use_pcr, grp["pcr_pred"].values, grp["saits_pred"].values)
        ds_rmse  = _rmse(grp["y_true"].values, ds_gated)
        per_ds_rows.append({
            "dataset": ds, "saits_rmse": ds_saits, "pcr_rmse": ds_pcr,
            "gated_rmse": ds_rmse,
            "improve_vs_saits_pct": 100.0*(ds_saits-ds_rmse)/ds_saits,
            "improve_vs_pcr_pct":   100.0*(ds_pcr-ds_rmse)/ds_pcr,
            "pcr_used_rate_pct": 100.0*ds_use_pcr.mean(),
            "saits_fallback_rate_pct": 100.0*(1.0-ds_use_pcr.mean()),
        })

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    tau_df.to_csv(out_dir / "E13_tau_sweep.csv", index=False)
    pd.DataFrame(per_ds_rows).to_csv(out_dir / "E13_per_dataset.csv", index=False)
    summary = {
        "run_code_version": RUN_CODE_VERSION,
        "best_tau": best_tau,
        "tau_n_grid": tau_n_grid,
        "gating_rule": "use PCR if abs_delta <= tau, else fallback SAITS",
        "val_saits_rmse": val_saits_rmse,
        "val_pcr_rmse": val_pcr_rmse,
        "val_gated_rmse": best_val_rmse,
        "test_saits_rmse": test_saits_rmse,
        "test_pcr_rmse": test_pcr_rmse,
        "test_gated_rmse": test_gated_rmse,
        "improve_vs_saits_pct": improve_vs_saits,
        "improve_vs_pcr_pct": improve_vs_pcr,
        "pcr_used_rate_pct": pcr_used_rate,
        "saits_fallback_rate_pct": saits_fallback_rate,
    }
    pd.DataFrame([summary]).to_csv(out_dir / "E13_summary.csv", index=False)

    print(f"\n[E13] RESULTS (tau={best_tau:.4f}, rule: PCR if abs_delta<=tau else SAITS)")
    print(f"  SAITS:  {test_saits_rmse:.4f}")
    print(f"  PCR:    {test_pcr_rmse:.4f}")
    print(f"  Gated:  {test_gated_rmse:.4f}  "
          f"(+{improve_vs_saits:.1f}% vs SAITS, {improve_vs_pcr:+.1f}% vs PCR)")
    print(f"  PCR used (|δ|<=τ): {pcr_used_rate:.1f}%  |  SAITS fallback: {saits_fallback_rate:.1f}%")
    print(f"[E13] outputs saved to {out_dir}")


# ---------------------------------------------------------------------------
# E14 — Downstream Forecasting
# ---------------------------------------------------------------------------

def _build_forecasting_sets(
    df: pd.DataFrame, horizon: int, history: int, imputed_col: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y) for a simple AR forecaster from a reconstructed series.
    X: history-length window of imputed values, y: next `horizon` true values.

    Bug fixes vs v1:
    - Bug 3: filter windows with NaN/Inf in X or y (from natural missing in y_true)
    - Bug 4: split train/test WITHIN each group (not across all groups mixed),
      so each temporal group contributes its own train/test split.
      This avoids temporal leakage across scenarios.
    """
    X_train_all, y_train_all = [], []
    X_test_all,  y_test_all  = [], []

    for (ds, seed, mechanism, pattern, rate), grp in df.groupby(
            ["dataset", "seed", "mechanism", "pattern", "rate"]):
        grp_sorted = grp.sort_values("t")
        features = grp_sorted["feature"].unique()
        for feat in features:
            fg = grp_sorted[grp_sorted["feature"] == feat].sort_values("t")
            y_imp = fg[imputed_col].values
            y_tr  = fg["y_true"].values
            obs   = fg["is_observed"].values.astype(bool)
            T = len(y_imp)
            Xs, ys = [], []
            for start in range(0, T - history - horizon + 1, horizon):
                end = start + history
                if obs[start:end].mean() < 0.8:
                    continue
                X_row = y_imp[start:end]
                y_row = y_tr[end:end + horizon]
                if len(y_row) != horizon:
                    continue
                # Bug fix 3: skip windows with NaN/Inf
                if not np.all(np.isfinite(X_row)):
                    continue
                if not np.all(np.isfinite(y_row)):
                    continue
                Xs.append(X_row); ys.append(y_row)
            if not Xs:
                continue
            # Bug fix 4: split within this group (not globally)
            n = len(Xs)
            n_test = max(1, int(n * 0.2))
            X_train_all.extend(Xs[:-n_test])
            y_train_all.extend(ys[:-n_test])
            X_test_all.extend(Xs[-n_test:])
            y_test_all.extend(ys[-n_test:])

    if not X_train_all or not X_test_all:
        return np.empty((0, history)), np.empty((0, horizon)), \
               np.empty((0, history)), np.empty((0, horizon))
    return (np.array(X_train_all, dtype=np.float32),
            np.array(y_train_all, dtype=np.float32),
            np.array(X_test_all,  dtype=np.float32),
            np.array(y_test_all,  dtype=np.float32))


def run_E14(pp_dir: Path, out_dir: Path,
            history: int = 24, horizon: int = 12) -> None:
    """E14 R4.12: downstream forecasting.
    Train a linear AR model on SAITS-imputed vs PCR-imputed series; compare
    MAE/RMSE of downstream forecasts relative to ground-truth y_true.

    NOTE on pcr_pred: 'pcr_pred' in the parquet corresponds to whichever
    method was passed as --per-position-methods in the exporter.
    Default = 'pcrsaitsv14_no_seasonal_branch' (the proposed method).
    Verify this matches the paper's proposed method before reporting.
    """
    print("\n[E14] Loading test parquets for downstream forecasting...")
    test_df = _load_parquets(pp_dir, split="test")
    if test_df.empty:
        raise RuntimeError("No test parquets found.")
    print(f"[E14] test rows: {len(test_df):,}")

    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        lin_impl = "ridge"
    except ImportError:
        lin_impl = "numpy_lstsq"
        print("[E14] sklearn not found; using numpy lstsq as forecaster.")

    results = []
    # Protocol fix: PhysioNet2012's per-position export concatenates ~2000
    # independent patient stays (48 timesteps each). This evaluator's AR
    # window-builder does not respect patient boundaries, so a fixed
    # history/horizon window can silently splice the tail of one patient's
    # stay onto the head of the next. Exclude PhysioNet from E14a until
    # per-patient windowing is implemented; the limitation is already
    # disclosed for E10 (R1.10) and applies here for the same reason.
    # Matches the alias set used by _load_physionet_dataset's own resolution.
    _PHYSIONET_ALIASES = {"physionet2012", "physionet", "physio", "physio2012"}
    for ds in sorted(test_df["dataset"].unique()):
        if str(ds).strip().lower() in _PHYSIONET_ALIASES:
            print(f"[E14][SKIP] {ds}: excluded from downstream forecasting "
                  f"(cross-patient boundary limitation, same as R1.10 E10 disclosure).")
            continue
        ds_df = test_df[test_df["dataset"] == ds].copy()
        for col, label in [("saits_pred", "saits_imputed"), ("pcr_pred", "pcr_imputed")]:
            result = _build_forecasting_sets(ds_df, horizon=horizon,
                                              history=history, imputed_col=col)
            X_tr, y_tr, X_te, y_te = result
            if len(X_tr) < 5 or len(X_te) < 1:
                print(f"[E14][WARN] {ds}/{label}: insufficient samples "
                      f"(train={len(X_tr)}, test={len(X_te)}); skipping.")
                continue
            # Normalise inputs
            scaler = None
            if lin_impl == "ridge":
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_te_s = scaler.transform(X_te)
                model = Ridge(alpha=1.0)
                model.fit(X_tr_s, y_tr)
                y_hat = model.predict(X_te_s)
            else:
                X_aug = np.hstack([X_tr, np.ones((len(X_tr), 1))])
                w, _, _, _ = np.linalg.lstsq(X_aug, y_tr, rcond=None)
                X_te_aug = np.hstack([X_te, np.ones((len(X_te), 1))])
                y_hat = X_te_aug @ w
            # Bug fix: sklearn's Ridge silently returns a 1D prediction array
            # when the target has a single column (n_targets==1), even though
            # y_tr/y_te were built as 2D (n, horizon). At horizon=1 this makes
            # y_hat.shape=(n,) while y_te.shape=(n,1); "y_hat - y_te" then
            # broadcasts to an (n,n) matrix instead of raising a shape error —
            # for a large test set this silently tries to allocate a
            # terabyte-scale array (observed: 1.12 TiB for n=553,955) instead
            # of failing fast. horizon>1 was never affected because sklearn
            # keeps multi-output predictions 2D. Force y_hat to match y_te.
            y_hat = np.asarray(y_hat).reshape(np.asarray(y_te).shape)
            mae  = float(np.mean(np.abs(y_hat - y_te)))
            rmse = float(np.sqrt(np.mean((y_hat - y_te) ** 2)))
            results.append({
                "dataset": ds, "imputation": label,
                "n_train": len(X_tr), "n_test": len(X_te),
                "forecaster": lin_impl,
                "history": history, "horizon": horizon,
                "mae": mae, "rmse": rmse,
            })
            print(f"[E14] {ds:>16} {label:>16}: MAE={mae:.4f} RMSE={rmse:.4f}")

    if not results:
        print("[E14] No results produced.")
        return

    res_df = pd.DataFrame(results)
    # Pivot: SAITS vs PCR side by side
    pivot = res_df.pivot_table(
        index="dataset", columns="imputation",
        values=["mae", "rmse"], aggfunc="mean"
    ).round(4)
    # Compute delta (PCR - SAITS): negative = PCR downstream better
    summary_rows = []
    for ds in res_df["dataset"].unique():
        ds_data = res_df[res_df["dataset"] == ds]
        saits_row = ds_data[ds_data["imputation"] == "saits_imputed"]
        pcr_row   = ds_data[ds_data["imputation"] == "pcr_imputed"]
        if saits_row.empty or pcr_row.empty:
            continue
        smae  = float(saits_row["mae"].values[0])
        srmse = float(saits_row["rmse"].values[0])
        pmae  = float(pcr_row["mae"].values[0])
        prmse = float(pcr_row["rmse"].values[0])
        summary_rows.append({
            "dataset": ds,
            "saits_mae": smae, "pcr_mae": pmae,
            "mae_improve_pct": 100.0*(smae - pmae)/smae,
            "saits_rmse": srmse, "pcr_rmse": prmse,
            "rmse_improve_pct": 100.0*(srmse - prmse)/srmse,
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_dir / "E14_forecasting_results.csv", index=False)
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(out_dir / "E14_summary.csv", index=False)

    print(f"\n[E14] RESULTS (history={history} horizon={horizon})")
    for r in summary_rows:
        print(f"  {r['dataset']:>16}: "
              f"SAITS MAE={r['saits_mae']:.4f} vs PCR MAE={r['pcr_mae']:.4f} "
              f"({r['mae_improve_pct']:+.1f}%)")
    print(f"[E14] outputs saved to {out_dir}")


# ---------------------------------------------------------------------------
# E14b — Threshold Exceedance Classification (environmental decision-making)
# ---------------------------------------------------------------------------

def run_E14b(pp_dir: Path, out_dir: Path,
             threshold_pct: float = 75.0) -> None:
    """E14b R4.12 (environmental decision-making): threshold exceedance
    classification on pollutant features of environmental sensor datasets.

    Scope: UCI Air Quality and Beijing PM2.5 only — datasets with
    well-defined pollutant thresholds that map to real environmental
    decisions (air quality index, exceedance alerts). ETT/PhysioNet
    are excluded because their features (power load, clinical vitals)
    do not have meaningful environmental threshold semantics.

    Threshold calibration: computed from validation parquets (not test)
    to avoid test-set calibration. Falls back to test observed positions
    with a warning if val parquets are absent.

    Classification rule: predict 1 if imputed_value > threshold.
    No trained classifier — result reflects imputation quality at the
    decision boundary without confounding downstream model capacity.
    """
    # Datasets and features with environmental threshold semantics
    ENV_DATASETS = {"airquality", "beijingpm25"}
    POLLUTANT_FEATURES = {
        # UCI Air Quality
        "CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)",
        # Beijing PM2.5
        "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
    }

    print("\n[E14b] Loading val parquets for threshold calibration...")
    val_df = _load_parquets(pp_dir, split="val")
    print(f"[E14b] Loading test parquets for evaluation...")
    test_df = _load_parquets(pp_dir, split="test")
    if test_df.empty:
        raise RuntimeError("No test parquets found.")

    ev = test_df[test_df["is_artificial_missing"] == 1].copy()
    print(f"[E14b] artificial missing positions: {len(ev):,}")

    try:
        from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
        _HAS_SKLEARN = True
    except ImportError:
        print("[E14b][WARN] sklearn not found; reporting accuracy only.")
        _HAS_SKLEARN = False

    results = []
    all_feat_rows = []   # Bug fix 4: accumulate per-feature rows for audit CSV

    for ds in sorted(ev["dataset"].unique()):
        # Bug fix 1: limit to environmental pollutant datasets only
        if ds not in ENV_DATASETS:
            print(f"[E14b][SKIP] {ds}: not an environmental threshold dataset "
                  f"(E14b covers UCI/Beijing only).")
            continue

        ds_ev = ev[ev["dataset"] == ds]
        # Bug fix 1: limit to pollutant features
        features = [f for f in sorted(ds_ev["feature"].unique())
                    if f in POLLUTANT_FEATURES]
        if not features:
            print(f"[E14b][SKIP] {ds}: no recognized pollutant features.")
            continue

        # Bug fix 2 (protocol): threshold from validation split, fallback to test observed.
        # Was: filtered to is_observed==1, which is defined by each scenario's artificial
        # mask — under MNAR in particular this can bias the threshold distribution toward
        # whichever positions that mechanism happens to leave observed. Use all finite
        # y_true in validation instead (y_true is ground truth regardless of mask).
        if not val_df.empty and "dataset" in val_df.columns:
            val_src = val_df[
                (val_df["dataset"] == ds)
                & np.isfinite(val_df["y_true"].to_numpy(dtype=float))
            ].copy()
        else:
            val_src = pd.DataFrame()
        if val_src.empty:
            print(f"[E14b][WARN] {ds}: no val parquets found; "
                  f"falling back to test observed for threshold calibration.")
            test_obs = test_df[(test_df["dataset"] == ds) & (test_df["is_observed"] == 1)]
            thr_src = test_obs
        else:
            thr_src = val_src

        feat_rows = []
        for feat in features:
            feat_ev  = ds_ev[ds_ev["feature"] == feat].copy()
            feat_thr = thr_src[thr_src["feature"] == feat] if "feature" in thr_src.columns else thr_src

            if feat_ev.empty or feat_thr.empty:
                continue

            # Protocol fix: the same underlying ground-truth position (same
            # seed/t/feature) recurs across many mechanism/pattern/rate
            # scenarios sharing that seed, since y_true doesn't depend on the
            # masking mechanism. Without dedup, positions that happen to
            # appear in more scenarios get overweighted in the percentile.
            _dedup_cols = [c for c in ["dataset", "seed", "t", "feature", "y_true"]
                           if c in feat_thr.columns]
            if _dedup_cols:
                feat_thr = feat_thr.drop_duplicates(subset=_dedup_cols)

            # Bug fix 3: finite filtering
            thr_vals = feat_thr["y_true"].values
            thr_vals = thr_vals[np.isfinite(thr_vals)]
            if len(thr_vals) == 0:
                continue
            thr = float(np.nanpercentile(thr_vals, threshold_pct))

            valid_mask = (
                np.isfinite(feat_ev["y_true"].values) &
                np.isfinite(feat_ev["saits_pred"].values) &
                np.isfinite(feat_ev["pcr_pred"].values)
            )
            feat_ev = feat_ev.loc[valid_mask].copy()
            if feat_ev.empty:
                continue

            y_true_bin = (feat_ev["y_true"].values > thr).astype(int)
            if y_true_bin.sum() == 0 or y_true_bin.sum() == len(y_true_bin):
                continue

            for col, label in [("saits_pred", "saits"), ("pcr_pred", "pcr")]:
                y_pred_cont = feat_ev[col].values
                y_pred_bin  = (y_pred_cont > thr).astype(int)
                acc = float((y_pred_bin == y_true_bin).mean())
                row = {"dataset": ds, "feature": feat, "imputation": label,
                       "threshold": thr, "threshold_pct": threshold_pct,
                       "threshold_source": "val" if not val_src.empty else "test_observed",
                       "n_positive": int(y_true_bin.sum()),
                       "n_total": len(y_true_bin),
                       "accuracy": acc}
                if _HAS_SKLEARN:
                    row["f1"]    = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))
                    try:
                        row["auroc"] = float(roc_auc_score(y_true_bin, y_pred_cont))
                        row["auprc"] = float(average_precision_score(y_true_bin, y_pred_cont))
                    except Exception:
                        row["auroc"] = row["auprc"] = float("nan")
                feat_rows.append(row)

        all_feat_rows.extend(feat_rows)  # Bug fix 4

        if not feat_rows:
            print(f"[E14b][WARN] {ds}: no valid features after filtering.")
            continue

        feat_df = pd.DataFrame(feat_rows)
        for label in ["saits", "pcr"]:
            sub = feat_df[feat_df["imputation"] == label]
            r = {"dataset": ds, "imputation": f"{label}_imputed",
                 "n_features": len(sub),
                 "accuracy_mean": sub["accuracy"].mean()}
            if _HAS_SKLEARN:
                r["f1_mean"]    = sub["f1"].mean()
                r["auroc_mean"] = sub["auroc"].mean()
                r["auprc_mean"] = sub["auprc"].mean()
            results.append(r)
            print(f"[E14b] {ds:>16} {label:>6}: "
                  f"acc={r['accuracy_mean']:.4f}"
                  + (f" F1={r['f1_mean']:.4f} AUROC={r['auroc_mean']:.4f}"
                     if _HAS_SKLEARN else ""))

    if not results:
        print("[E14b] No results produced.")
        return

    res_df = pd.DataFrame(results)
    summary_rows = []
    for ds in res_df["dataset"].unique():
        s = res_df[(res_df["dataset"] == ds) & (res_df["imputation"] == "saits_imputed")]
        p = res_df[(res_df["dataset"] == ds) & (res_df["imputation"] == "pcr_imputed")]
        if s.empty or p.empty:
            continue
        row = {"dataset": ds,
               "saits_accuracy": float(s["accuracy_mean"].values[0]),
               "pcr_accuracy":   float(p["accuracy_mean"].values[0]),
               "accuracy_improve_pct": 100.0*(float(p["accuracy_mean"].values[0])
                                              - float(s["accuracy_mean"].values[0]))
                                      / max(float(s["accuracy_mean"].values[0]), 1e-9)}
        if _HAS_SKLEARN:
            row["saits_f1"]      = float(s["f1_mean"].values[0])
            row["pcr_f1"]        = float(p["f1_mean"].values[0])
            row["f1_improve_pct"] = 100.0*(row["pcr_f1"] - row["saits_f1"]) / max(row["saits_f1"], 1e-9)
            row["saits_auroc"]   = float(s["auroc_mean"].values[0])
            row["pcr_auroc"]     = float(p["auroc_mean"].values[0])
            row["auroc_improve_pct"] = 100.0*(row["pcr_auroc"] - row["saits_auroc"]) / max(row["saits_auroc"], 1e-9)
        summary_rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_dir / "E14b_classification_results.csv", index=False)
    if all_feat_rows:  # Bug fix 4: save per-feature CSV for audit
        pd.DataFrame(all_feat_rows).to_csv(
            out_dir / "E14b_per_feature_results.csv", index=False)
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(out_dir / "E14b_summary.csv", index=False)

    print(f"\n[E14b] RESULTS (threshold=p{threshold_pct:.0f}, rule: predict 1 if imputed > threshold)")
    for r in summary_rows:
        print(f"  {r['dataset']:>16}: "
              f"SAITS acc={r['saits_accuracy']:.4f} vs PCR acc={r['pcr_accuracy']:.4f} "
              f"({r['accuracy_improve_pct']:+.2f}%)"
              + (f"  F1: SAITS={r['saits_f1']:.4f} PCR={r['pcr_f1']:.4f} "
                 f"AUROC: SAITS={r['saits_auroc']:.4f} PCR={r['pcr_auroc']:.4f}"
                 if _HAS_SKLEARN else ""))
    print(f"[E14b] outputs saved to {out_dir}")
    print("[E14b] NOTE: pollutant features only (UCI+Beijing); threshold calibrated "
          "on val split; no trained downstream model — result reflects imputation "
          "quality at the environmental decision boundary (R4.12).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# =============================================================================
# E14c (R4.12 extension): EPA Air Quality Index (AQI) breakpoint classification
# =============================================================================
# Unlike E14b (threshold derived from each dataset's own validation-split
# percentile), E14c uses a FIXED external threshold: the lower bound of US
# EPA's "Unhealthy for Sensitive Groups" (USG) AQI category — a real
# public-health advisory trigger, not a threshold chosen to fit the data.
# This is a stronger "real-world decision-making" claim for R4.12 than E14b's
# percentile threshold, at the cost of extra unit-conversion assumptions.
#
# CRITICAL CAVEATS — read before trusting these numbers for the manuscript:
#   1. EPA breakpoints are defined in EPA's native units (ppm/ppb for gases,
#      ug/m3 for particulates). Converting ug/m3 <-> ppm/ppb requires a molar
#      volume at some temperature/pressure; we use 25C/1atm (24.45 L/mol,
#      EPA's own convention), but the actual ambient T/P at measurement time
#      in these datasets is unknown. This conversion carries irreducible
#      uncertainty for every pollutant EXCEPT particulates.
#   2. Column units were inferred from the observed data scale (see the
#      sanity-check print at runtime and the p75 values already seen in a
#      prior E14b run), NOT confirmed against each dataset's original
#      documentation. Beijing's CO was corrected from an initial mg/m3
#      assumption to ug/m3 this way — verify independently before publishing.
#   3. UCI's NOx(GT) is excluded: NOx (=NO+NO2) is a different pollutant from
#      NO2, and EPA's AQI breakpoints are defined for NO2 specifically, not NOx.
#   4. UCI's NMHC(GT)/C6H6(GT) are excluded: EPA's AQI does not define
#      breakpoints for non-methane hydrocarbons or benzene.
#
# Default scope (--epa-features not given) is the CLEAN set: PM2.5, PM10 only
# — both already in EPA's native ug/m3, zero conversion assumptions. Pass
# --epa-features to opt into the EXTENDED set (CO, NO2, SO2, O3), which
# carries the STP-conversion caveat above.

_MOLAR_MASS_G_PER_MOL = {"CO": 28.01, "NO2": 46.0055, "SO2": 64.07, "O3": 48.00}
_MOLAR_VOLUME_L_PER_MOL_25C = 24.45  # EPA convention, 25C / 1 atm

def _epa_native_threshold(pollutant: str, epa_value: float, epa_unit: str, data_unit: str) -> float:
    """Convert an EPA breakpoint (in EPA's native ppm/ppb/ug_m3 unit) into
    whatever unit a given dataset column actually uses, so it can be compared
    directly against that column's raw values without converting the data."""
    if epa_unit == data_unit:
        return epa_value
    M = _MOLAR_MASS_G_PER_MOL[pollutant]
    V = _MOLAR_VOLUME_L_PER_MOL_25C
    if epa_unit == "ppb" and data_unit == "ug_m3":
        return epa_value * M / V
    if epa_unit == "ppm" and data_unit == "mg_m3":
        return epa_value * M / V
    if epa_unit == "ppm" and data_unit == "ug_m3":
        return epa_value * M * 1000.0 / V
    raise ValueError(f"Unhandled unit conversion: {epa_unit} -> {data_unit} for {pollutant}")

# EPA "Unhealthy for Sensitive Groups" (USG) lower-bound breakpoints, EPA's native units.
_EPA_USG_BREAKPOINTS = {
    "PM2.5": (35.5, "ug_m3"),
    "PM10":  (155.0, "ug_m3"),
    "CO":    (9.5, "ppm"),
    "NO2":   (101.0, "ppb"),
    "SO2":   (76.0, "ppb"),
    "O3":    (0.071, "ppm"),
}

# (dataset, feature) -> (pollutant key into _EPA_USG_BREAKPOINTS, this column's actual unit).
# Unit assumption per column — verify against your own dataset's documentation.
_E14C_FEATURE_UNITS = {
    ("beijingpm25", "PM2.5"): ("PM2.5", "ug_m3"),
    ("beijingpm25", "PM10"):  ("PM10",  "ug_m3"),
    ("beijingpm25", "CO"):    ("CO",  "ug_m3"),   # corrected from mg_m3 via p75 sanity check
    ("beijingpm25", "NO2"):   ("NO2", "ug_m3"),
    ("beijingpm25", "SO2"):   ("SO2", "ug_m3"),
    ("beijingpm25", "O3"):    ("O3",  "ug_m3"),
    ("airquality", "CO(GT)"):  ("CO",  "mg_m3"),  # per UCI documentation
    ("airquality", "NO2(GT)"): ("NO2", "ug_m3"),
}

_E14C_CLEAN_FEATURES = {"PM2.5", "PM10"}  # default scope — zero conversion assumptions


def run_E14c(pp_dir: Path, out_dir: Path, epa_features: list = None) -> None:
    """E14c (R4.12 extension): binary classification against a FIXED EPA AQI
    breakpoint (USG lower bound), rather than E14b's percentile threshold.
    See the module-level comment block above this function for unit-
    conversion caveats before trusting these numbers in the manuscript.
    """
    scope = set(epa_features) if epa_features else set(_E14C_CLEAN_FEATURES)
    unknown = scope - set(_EPA_USG_BREAKPOINTS.keys())
    if unknown:
        raise ValueError(f"No EPA breakpoint defined for: {sorted(unknown)}. "
                          f"Available: {sorted(_EPA_USG_BREAKPOINTS.keys())}")
    is_extended = bool(scope - _E14C_CLEAN_FEATURES)
    if is_extended:
        print(f"[E14c][WARN] Extended scope requested ({sorted(scope - _E14C_CLEAN_FEATURES)}): "
              f"these use STP-assumed unit conversion (25C/1atm). Verify column units "
              f"independently before using these numbers in the manuscript.")

    print("\n[E14c] Loading val parquets for EPA breakpoint sanity check...")
    val_df = _load_parquets(pp_dir, split="val")
    print("[E14c] Loading test parquets for evaluation...")
    test_df = _load_parquets(pp_dir, split="test")
    if test_df.empty:
        raise RuntimeError("No test parquets found.")

    ev = test_df[test_df["is_artificial_missing"] == 1].copy()

    try:
        from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
        _HAS_SKLEARN = True
    except ImportError:
        print("[E14c][WARN] sklearn not found; reporting accuracy only.")
        _HAS_SKLEARN = False

    results = []
    all_feat_rows = []

    active_keys = [(ds, feat) for (ds, feat) in _E14C_FEATURE_UNITS.keys()
                   if _E14C_FEATURE_UNITS[(ds, feat)][0] in scope]
    for ds in sorted({k[0] for k in active_keys}):
        ds_ev = ev[ev["dataset"] == ds]
        if ds_ev.empty:
            print(f"[E14c][SKIP] {ds}: not present in per-position export.")
            continue

        feat_rows = []
        for (_ds, feat) in [k for k in active_keys if k[0] == ds]:
            pollutant, data_unit = _E14C_FEATURE_UNITS[(ds, feat)]
            epa_value, epa_unit = _EPA_USG_BREAKPOINTS[pollutant]
            thr = _epa_native_threshold(pollutant, epa_value, epa_unit, data_unit)

            feat_ev = ds_ev[ds_ev["feature"] == feat].copy()
            if feat_ev.empty:
                print(f"[E14c][SKIP] {ds}/{feat}: not present in per-position export.")
                continue

            # Sanity check: compare the EPA-derived native-unit threshold
            # against the actual observed data range. A threshold wildly
            # outside the observed range (e.g. 100x off) is a strong signal
            # of a unit mismatch — printed for every feature, not just
            # suspicious ones, so it can be visually cross-checked.
            src_for_range = val_df[(val_df.get("dataset") == ds) & (val_df.get("feature") == feat)] \
                if not val_df.empty else pd.DataFrame()
            range_src = src_for_range if not src_for_range.empty else feat_ev
            obs = range_src["y_true"].to_numpy(dtype=float)
            obs = obs[np.isfinite(obs)]
            if len(obs):
                print(f"[E14c] {ds}/{feat} ({pollutant}, assumed {data_unit}): "
                      f"EPA USG threshold={thr:.3f} | observed y_true "
                      f"min={obs.min():.2f} p50={np.median(obs):.2f} "
                      f"p95={np.percentile(obs,95):.2f} max={obs.max():.2f}")
                if thr > obs.max() * 5 or thr < obs.min() / 5:
                    print(f"[E14c][WARN] {ds}/{feat}: threshold is >5x outside the observed "
                          f"range — check the unit assumption ({data_unit}) before trusting this feature.")

            valid_mask = (
                np.isfinite(feat_ev["y_true"].values) &
                np.isfinite(feat_ev["saits_pred"].values) &
                np.isfinite(feat_ev["pcr_pred"].values)
            )
            feat_ev = feat_ev.loc[valid_mask].copy()
            if feat_ev.empty:
                continue

            y_true_bin = (feat_ev["y_true"].values > thr).astype(int)
            n_pos, n_tot = int(y_true_bin.sum()), len(y_true_bin)
            pos_rate = n_pos / max(n_tot, 1)
            if n_pos == 0 or n_pos == n_tot:
                print(f"[E14c][SKIP] {ds}/{feat}: degenerate class "
                      f"(positive rate={pos_rate:.4f}) — EPA threshold rarely/always exceeded "
                      f"in this data; not usable for classification.")
                continue
            if pos_rate < 0.02 or pos_rate > 0.98:
                print(f"[E14c][WARN] {ds}/{feat}: highly imbalanced (positive rate={pos_rate:.4f}). "
                      f"Metrics below may be unstable / dominated by the majority class.")

            for col, label in [("saits_pred", "saits"), ("pcr_pred", "pcr")]:
                y_pred_cont = feat_ev[col].values
                y_pred_bin  = (y_pred_cont > thr).astype(int)
                acc = float((y_pred_bin == y_true_bin).mean())
                row = {"dataset": ds, "feature": feat, "pollutant": pollutant,
                       "assumed_unit": data_unit, "imputation": label,
                       "epa_threshold_native_unit": thr,
                       "epa_category": "unhealthy_for_sensitive_groups_lower_bound",
                       "n_positive": n_pos, "n_total": n_tot, "positive_rate": pos_rate,
                       "accuracy": acc}
                if _HAS_SKLEARN:
                    row["f1"] = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))
                    try:
                        row["auroc"] = float(roc_auc_score(y_true_bin, y_pred_cont))
                        row["auprc"] = float(average_precision_score(y_true_bin, y_pred_cont))
                    except Exception:
                        row["auroc"] = row["auprc"] = float("nan")
                feat_rows.append(row)

        all_feat_rows.extend(feat_rows)
        if not feat_rows:
            print(f"[E14c][WARN] {ds}: no usable features after filtering.")
            continue

        feat_df = pd.DataFrame(feat_rows)
        for label in ["saits", "pcr"]:
            sub = feat_df[feat_df["imputation"] == label]
            r = {"dataset": ds, "imputation": f"{label}_imputed",
                 "n_features": len(sub), "accuracy_mean": sub["accuracy"].mean()}
            if _HAS_SKLEARN:
                r["f1_mean"] = sub["f1"].mean()
                r["auroc_mean"] = sub["auroc"].mean()
                r["auprc_mean"] = sub["auprc"].mean()
            results.append(r)
            print(f"[E14c] {ds:>16} {label:>6}: acc={r['accuracy_mean']:.4f}"
                  + (f" F1={r['f1_mean']:.4f} AUROC={r['auroc_mean']:.4f}" if _HAS_SKLEARN else ""))

    if not results:
        print("[E14c] No results produced (all features degenerate/missing — see SKIP lines above).")
        return

    res_df = pd.DataFrame(results)
    summary_rows = []
    for ds in res_df["dataset"].unique():
        s = res_df[(res_df["dataset"] == ds) & (res_df["imputation"] == "saits_imputed")]
        p = res_df[(res_df["dataset"] == ds) & (res_df["imputation"] == "pcr_imputed")]
        if s.empty or p.empty:
            continue
        row = {"dataset": ds,
               "saits_accuracy": float(s["accuracy_mean"].values[0]),
               "pcr_accuracy": float(p["accuracy_mean"].values[0])}
        row["accuracy_improve_pct"] = 100.0 * (row["pcr_accuracy"] - row["saits_accuracy"]) \
                                       / max(row["saits_accuracy"], 1e-9)
        if _HAS_SKLEARN:
            row["saits_f1"] = float(s["f1_mean"].values[0])
            row["pcr_f1"] = float(p["f1_mean"].values[0])
            row["f1_improve_pct"] = 100.0 * (row["pcr_f1"] - row["saits_f1"]) / max(row["saits_f1"], 1e-9)
            row["saits_auroc"] = float(s["auroc_mean"].values[0])
            row["pcr_auroc"] = float(p["auroc_mean"].values[0])
            row["auroc_improve_pct"] = 100.0 * (row["pcr_auroc"] - row["saits_auroc"]) / max(row["saits_auroc"], 1e-9)
        summary_rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_dir / "E14c_classification_results.csv", index=False)
    if all_feat_rows:
        pd.DataFrame(all_feat_rows).to_csv(out_dir / "E14c_per_feature_results.csv", index=False)
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(out_dir / "E14c_summary.csv", index=False)

    print(f"\n[E14c] RESULTS (EPA USG breakpoint, rule: predict 1 if imputed > threshold)")
    for r in summary_rows:
        print(f"  {r['dataset']:>16}: SAITS acc={r['saits_accuracy']:.4f} vs "
              f"PCR acc={r['pcr_accuracy']:.4f} ({r['accuracy_improve_pct']:+.2f}%)"
              + (f"  F1: SAITS={r['saits_f1']:.4f} PCR={r['pcr_f1']:.4f} "
                 f"AUROC: SAITS={r['saits_auroc']:.4f} PCR={r['pcr_auroc']:.4f}"
                 if _HAS_SKLEARN else ""))
    print(f"[E14c] outputs saved to {out_dir}")
    print("[E14c] NOTE: threshold is a FIXED EPA public-health breakpoint (not fit to this "
          "data). See run_E14c docstring / module comment for unit-conversion caveats.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="E13 (confidence gating) + E14a (forecasting) + E14b (classification) "
                    "+ E14c (EPA breakpoint classification) evaluator.")
    p.add_argument("--per-position-dir", type=str, required=True,
                   help="Directory containing dataset subdirs with per_position/*.parquet")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--run-E13",  action="store_true")
    p.add_argument("--run-E14",  action="store_true", help="E14a: Ridge AR forecasting (null result)")
    p.add_argument("--run-E14b", action="store_true", help="E14b: threshold exceedance classification (main)")
    p.add_argument("--run-all",  action="store_true",
                   help="Run E13 + E14a + E14b. E14c must be requested explicitly (EPA unit caveats).")
    p.add_argument("--tau-n-grid", type=int, default=50,
                   help="Number of tau values to sweep in E13 (default 50)")
    p.add_argument("--forecast-history", type=int, default=24,
                   help="History window for E14a forecaster (default 24)")
    p.add_argument("--forecast-horizon", type=int, default=12,
                   help="Forecast horizon for E14a (default 12)")
    p.add_argument("--threshold-pct", type=float, default=75.0,
                   help="Percentile threshold for E14b exceedance (default 75)")
    p.add_argument("--run-E14c", action="store_true",
                   help="E14c: EPA AQI breakpoint classification (fixed threshold, R4.12 extension)")
    p.add_argument("--epa-features", type=str, default=None,
                   help="Comma list of pollutants for E14c, e.g. 'PM2.5,PM10,NO2,CO,SO2,O3'. "
                        "Default: PM2.5,PM10 (no unit-conversion assumption). Anything beyond "
                        "that opts into the STP-assumed conversion caveat.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.run_all:
        args.run_E13 = args.run_E14 = args.run_E14b = True
        # E14c stays explicit-only by default (EPA unit-conversion caveats).
        # Uncomment to fold it into --run-all: args.run_E14c = True
    if not (args.run_E13 or args.run_E14 or args.run_E14b or args.run_E14c):
        print("Nothing to do. Use --run-E13, --run-E14, --run-E14b, --run-E14c, or --run-all.")
        build_parser().print_help()
        return
    pp_dir  = Path(args.per_position_dir)
    out_dir = Path(args.output_dir)
    if args.run_E13:
        run_E13(pp_dir, out_dir / "E13_confidence_gating",
                tau_n_grid=args.tau_n_grid)
    if args.run_E14:
        run_E14(pp_dir, out_dir / "E14a_downstream_forecasting",
                history=args.forecast_history,
                horizon=args.forecast_horizon)
    if args.run_E14b:
        run_E14b(pp_dir, out_dir / "E14b_exceedance_classification",
                 threshold_pct=args.threshold_pct)
    if args.run_E14c:
        epa_feats = [s.strip() for s in args.epa_features.split(",") if s.strip()] \
            if args.epa_features else None
        run_E14c(pp_dir, out_dir / "E14c_epa_breakpoint_classification",
                  epa_features=epa_feats)
    print("\n[DONE] E13/E14 evaluator finished.")


if __name__ == "__main__":
    main()