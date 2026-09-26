#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCR-SAITS Revision-1 — Patch 4: Post-hoc Analysis
=================================================
Post-hoc studies that interpret a REPRESENTATIVE trained proposed corrector.
Patch 4 trains one proposed corrector (frozen SAITS backbone + PCR MLP) for a
single seed, intercepts it plus the exact train/val examples it was built on,
and analyses THAT model. The corrector is trained once per seed on
make_holdout_train_mask, whose holdout already mixes pointwise (len 1) and
block (len 6/12/24/48) gaps, so a single fit's examples span every gap length
E3 buckets over. The manuscript should describe E3/E4/E5 as "a representative
single-configuration corrector trained for post-hoc analysis", not as the
exact model behind every main-table cell.

Experiments added in this patch
-------------------------------
E3  (R1.7)  Gap-length-stratified analysis: how correction quality varies
                                      across pointwise and long block gaps in
                                      held-out PCR validation examples.
E4  (R1.8)  SHAP feature importance: which of the PCR corrector's input
                                      features drive the residual correction.
E5  (R4.4)  Correction-baseline sanity check: compare the learned PCR
                                      corrector against simpler residual
                                      correctors on the same extracted features.

Design / honesty notes
----------------------
* E3/E4/E5 operate on the SAME 8-D (or 10-D with domain tags) feature set the
  PCR corrector actually consumes, extracted via v7.1's own
  PCRSAITSV1CleanWrapper._build_examples, so the analysis reflects the real
  model, not a re-derived approximation.
* E4 uses shap if installed; otherwise it falls back to a permutation-
  importance estimate and SAYS SO in the output, so the numbers are never
  silently a different method than claimed.
* E5's XGBoost baseline is only run if xgboost is installed; otherwise a
  gradient-boosting regressor from scikit-learn is used as a stand-in and the
  output records which was used.

Author
------
Sawet Somnugpong, KPRU. 2026.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Import v7.1 base module + reuse Patch-1 helpers (single shared instance)
# ---------------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
RUN_CODE_VERSION = "patch4_posthoc_v3"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _find(pattern_list: List[str]) -> Path:
    for name in pattern_list:
        p = THIS_DIR / name
        if p.exists():
            return p
    for name in pattern_list:
        stem = name.replace(".py", "").split("*")[0]
        for probe in (stem, "_".join(stem.split("_")[:5]),
                      "_".join(stem.split("_")[:4])):
            if not probe:
                continue
            hits = sorted(THIS_DIR.glob(f"*{probe}*.py"))
            if hits:
                return hits[0]
    raise FileNotFoundError(f"Could not find any of {pattern_list} in {THIS_DIR}")


# Load Patch 1 first, reuse its v7.1 instance (same pattern as Patch 2).
_ext_path = _find(["pcr_saits_v1_5_main_extensions.py"])
_ext = _load_module(_ext_path, "pcr_ext")
_v71 = _ext._v71

now_stamp = _v71.now_stamp
choose_device = _v71.choose_device
compute_metrics = _v71.compute_metrics
revision_root = _ext.revision_root
exp_dir = _ext.exp_dir
append_result_row = _ext.append_result_row
write_state = _ext.write_state
_build_inner_namespace = _ext._build_inner_namespace

# v7.1 building blocks
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
make_holdout_train_mask = _v71.make_holdout_train_mask
compute_ref_feature_map = _v71.compute_ref_feature_map
set_seed = _v71.set_seed
load_air_quality_dataset = _v71.load_air_quality_dataset


# The names of the PCR corrector's input features, in the exact order that
# PCRSAITSV1CleanWrapper._build_examples stacks them (verified against v7.1).
PCR_FEATURE_NAMES_BASE = [
    "base_imputed",          # SAITS base prediction
    "local_estimate",        # local interpolation estimate
    "seasonal_estimate",     # seasonal (lag-24) estimate
    "base_minus_local",      # disagreement: base vs local
    "base_minus_seasonal",   # disagreement: base vs seasonal
    "local_valid",           # is local estimate available
    "seasonal_valid",        # is seasonal estimate available
    "gap_len_norm",          # normalised gap length (gl/48)
]
# v7.1 domain flags: group_id 0=pollutant, 1=sensor, 2=meteo. The
# corrector appends float(feat_group==0), float(feat_group==1) =
# [is_pollutant, is_sensor]; meteo is the (0,0) case, not a flag.
PCR_FEATURE_NAMES_DOMAIN = ["is_pollutant", "is_sensor"]


# ---------------------------------------------------------------------------
# Shared: rebuild the trained model + a labelled example set for one dataset
# ---------------------------------------------------------------------------


def _prepare_trained_corrector(script_dir: Path,
                               dataset: str,
                               seed: int,
                               window: int = 48,
                               allow_env_epochs: bool = False,
                               ):
    """Return the REAL trained PCR corrector plus the exact example matrix it
    was trained on, by running v7.1's own single-scenario pipeline and
    intercepting the corrector at the moment v7.1 finishes training it.

    Why interception rather than rebuild: v7.1 constructs the corrector with a
    long list of hyperparameters (learning_rate, weight_decay, three loss
    weights, batch_size, patience, base_impute_stride, ...) and a specific
    SAITS backbone (d_model, d_ffn, n_heads, n_layers, dropout). Re-deriving
    all of that here would risk analysing a model that is subtly NOT the
    proposed method. Instead we monkey-patch PCRSAITSV1CleanWrapper.fit to
    grab `self` and the exact (X, y, gap_len) built inside it. The analysed
    object is therefore identical to the one the paper reports.
    """
    captured: Dict[str, Any] = {"fits": []}

    _orig_build = PCRSAITSV1CleanWrapper._build_examples
    _orig_fit = PCRSAITSV1CleanWrapper.fit

    # per-fit scratch: the two _build_examples calls (train then val) land here
    _scratch: Dict[str, Any] = {}

    def _capturing_build(self, original_values, masked_values,
                         base_imputed_values, target_mask, gap_len):
        X, y, base_err, easy = _orig_build(
            self, original_values, masked_values, base_imputed_values,
            target_mask, gap_len)
        # first call after a fit starts = TRAIN, second = VAL
        slot = "train" if "X_train" not in _scratch else "val"
        _scratch[f"X_{slot}"] = np.asarray(X)
        _scratch[f"y_{slot}"] = np.asarray(y)
        return X, y, base_err, easy

    def _capturing_fit(self, *a, **kw):
        _scratch.clear()                 # reset per fit
        out = _orig_fit(self, *a, **kw)
        # Store the single captured fit's train/val examples. The list form
        # is kept intentionally so we can fail loudly if future v7.1 versions
        # start training more than one proposed corrector per seed.
        if "X_train" in _scratch and "X_val" in _scratch:
            captured["fits"].append({
                "X_train": _scratch["X_train"], "y_train": _scratch["y_train"],
                "X_val": _scratch["X_val"], "y_val": _scratch["y_val"],
            })
        # Keep the trained corrector + its metadata for prediction.
        captured["corrector"] = self
        captured["feature_names_model"] = list(self.feature_names)
        captured["use_domain_tags"] = bool(getattr(self, "use_domain_tags", False))
        captured["direct_residual"] = bool(getattr(self, "direct_residual", False))
        captured["input_dim"] = int(getattr(self, "input_dim", 8))
        return out

    # Build a namespace that trains ONE proposed corrector on a single seed.
    # IMPORTANT (verified against v7.1): the PCR corrector is trained ONCE per
    # seed on make_holdout_train_mask, whose holdout already mixes pointwise
    # gaps (len 1) with block gaps of length 6/12/24/48. So a single fit's
    # examples already span the full range of gap lengths E3 buckets over — we
    # do NOT need multiple pattern fits, and the corrector we analyse matches
    # the examples exactly (one fit, one example set). args.patterns only
    # affects the held-out EVALUATION scenarios, which we don't reach here.
    ns = _build_inner_namespace(
        script_dir=script_dir, dataset=dataset, seed=seed, window=window,
        ar1_csv_path=None, ckpt_subdir=exp_dir(script_dir, "E_posthoc_tmp"),
        full_grid=False, all_methods=False,
    )
    ns.methods = ["saits", "pcrsaitsv14_no_seasonal_branch"]
    ns.seeds = [int(seed)]
    ns.mechanisms = ["MCAR"]
    ns.patterns = ["block_24"]   # eval-only; irrelevant to the captured fit
    ns.rates = [0.2]
    ns.reset_checkpoints = True
    ns.retrain_models = True

    # Guard against silent smoke-test epoch overrides inherited from Patch 1.
    # _build_inner_namespace honours SAITS_EPOCHS/BRITS_EPOCHS environment
    # variables; for reviewer-facing post-hoc analysis we restore v7.1 defaults
    # unless the user explicitly opts in via --allow-env-epochs.
    env_epoch_vars = {k: os.environ.get(k) for k in ("SAITS_EPOCHS", "BRITS_EPOCHS")
                      if os.environ.get(k) is not None}
    env_epochs_ignored = bool(env_epoch_vars) and not allow_env_epochs
    if not allow_env_epochs:
        try:
            _defaults = _v71.build_parser().parse_args([])
            ns.saits_epochs = _defaults.saits_epochs
            ns.brits_epochs = _defaults.brits_epochs
        except Exception:
            pass
    epochs_used = {
        "saits_epochs": int(getattr(ns, "saits_epochs", -1)),
        "brits_epochs": int(getattr(ns, "brits_epochs", -1)),
    }
    if env_epochs_ignored:
        print(f"[PATCH4][TRAIN-CONFIG] ignoring env epoch overrides {env_epoch_vars}; "
              f"using defaults saits_epochs={epochs_used['saits_epochs']} "
              f"brits_epochs={epochs_used['brits_epochs']}")
    else:
        print(f"[PATCH4][TRAIN-CONFIG] saits_epochs={epochs_used['saits_epochs']} "
              f"brits_epochs={epochs_used['brits_epochs']} "
              f"allow_env_epochs={bool(allow_env_epochs)}")

    PCRSAITSV1CleanWrapper._build_examples = _capturing_build
    PCRSAITSV1CleanWrapper.fit = _capturing_fit
    try:
        # Call v7.1's INNER single-dataset runner, not the outer run_experiment.
        # The outer wrapper does post-run stats aggregation (reads
        # run_results.csv / wilcoxon_vs_reference.csv / ...) that crashes on a
        # tiny single-scenario run where some of those CSVs are empty
        # (pandas EmptyDataError). We only need the corrector to be TRAINED so
        # our monkey-patched fit/_build_examples fire and capture it; the inner
        # runner trains everything and returns without that fragile
        # aggregation. We resolve the dataset name -> the outer loop's
        # _ACTIVE_DATASET_NAME / cache exactly as run_experiment does, so the
        # patched loader still serves the right frame.
        globals_v71 = _v71.__dict__
        globals_v71["_ACTIVE_DATASET_NAME"] = ns.datasets[0] if getattr(ns, "datasets", None) else dataset
        # point single-run output at the tmp dir
        ns.output_dir = str(exp_dir(script_dir, "E_posthoc_tmp"))
        try:
            _v71._ORIG_RUN_EXPERIMENT_SINGLE(ns)
        except Exception as inner_exc:
            # If the inner runner itself post-processes and trips, that's fine
            # as long as we already captured a trained corrector below.
            if not captured.get("fits"):
                raise
            print(f"[PATCH4][INFO] inner run raised after capture "
                  f"({type(inner_exc).__name__}); continuing with captured "
                  f"corrector.")
    finally:
        PCRSAITSV1CleanWrapper._build_examples = _orig_build
        PCRSAITSV1CleanWrapper.fit = _orig_fit

    if not captured.get("fits") or "corrector" not in captured:
        raise RuntimeError(
            "Failed to intercept train+val PCR examples from v7.1. The "
            "run may not have trained the proposed method.")

    # Sanity: the proposed corrector is trained exactly once per seed. If more
    # than one fit was captured (e.g. a future v7.1 trains per scenario), the
    # single-corrector assumption below would be wrong, so fail loudly instead
    # of silently analysing the last corrector against all examples.
    n_fits = len(captured["fits"])
    if n_fits != 1:
        raise RuntimeError(
            f"Expected exactly ONE proposed-corrector fit, captured {n_fits}. "
            f"v7.1's training structure may have changed; the post-hoc "
            f"analysis assumes one corrector per seed. Aborting to avoid "
            f"mismatching the corrector with examples from other fits.")

    # Extract the single captured fit. Because n_fits is asserted above, the
    # corrector and examples are guaranteed to come from the same fit. Gap
    # variety comes from make_holdout_train_mask, not from multiple pattern fits.
    fit0 = captured["fits"][0]
    X_train = fit0["X_train"]
    y_train = fit0["y_train"]
    X_val = fit0["X_val"]
    y_val = fit0["y_val"]

    width = X_train.shape[1]
    # v7.1's `self.feature_names` are data-column names, not the PCR input
    # feature names. The PCR input feature order is fixed by _build_examples,
    # so label by the verified PCR-input list when the width matches; otherwise
    # fall back to generic names rather than risk mislabelled feature importances.
    known = list(PCR_FEATURE_NAMES_BASE)
    if width == len(PCR_FEATURE_NAMES_BASE) + len(PCR_FEATURE_NAMES_DOMAIN):
        known = known + list(PCR_FEATURE_NAMES_DOMAIN)
    feat_names = known if len(known) == width else [f"f{i}" for i in range(width)]

    gl_idx = PCR_FEATURE_NAMES_BASE.index("gap_len_norm")

    def _gl_from(Xarr):
        return Xarr[:, gl_idx] * 48.0

    return {
        "corrector": captured["corrector"],
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "gap_len_train": _gl_from(X_train),
        "gap_len_val": _gl_from(X_val),
        "feature_names": feat_names,
        "direct_residual": captured.get("direct_residual", False),
        "n_pattern_fits": len(captured["fits"]),
        "epochs_used": epochs_used,
        "allow_env_epochs": bool(allow_env_epochs),
        "env_epoch_vars": env_epoch_vars,
        "env_epochs_ignored": bool(env_epochs_ignored),
        "dataset": dataset, "seed": seed, "window": window,
    }


def _predict_residual(corrector, X: np.ndarray) -> np.ndarray:
    """Predict the corrector's residual for a feature matrix, matching exactly
    what v7.1's correct() computes: residual = delta * corr_mask, with the
    model's OWN direct_residual flag (not a guessed default). This ensures the
    analysed prediction is the proposed soft-gated residual, not a bypass."""
    import torch
    Xf = np.asarray(X, dtype=np.float32)
    net = corrector.model
    net.eval()
    dr = bool(getattr(corrector, "direct_residual", False))
    with torch.no_grad():
        xb = torch.from_numpy(Xf).to(corrector.device)
        delta, corr_mask = net(xb, direct_residual=dr)
        pred = (delta.squeeze(-1) * corr_mask.squeeze(-1)).cpu().numpy().reshape(-1)
    return pred


# ---------------------------------------------------------------------------
# E3: Long-gap visualisation (R1.7)
# ---------------------------------------------------------------------------


def run_E3_longgap_viz(args, script_dir: Path) -> None:
    """Quantify how correction quality varies with gap length.
    Produces a CSV and aggregate figure of RMSE/MAE for SAITS base vs
    SAITS+PCR, bucketed by gap length on held-out PCR validation examples.
    This is a gap-length-stratified post-hoc analysis, not a single time-series
    case-study plot.
    """
    out_dir = exp_dir(script_dir, "E3_longgap_viz")
    print(f"[E3] writing to {out_dir}")
    seed = int(args.sensitivity_seed)
    dataset = args.sensitivity_dataset

    prep = _prepare_trained_corrector(script_dir, dataset, seed,
                                      allow_env_epochs=args.allow_env_epochs)
    # Evaluate on the held-out VAL examples (corrector fit gradients on TRAIN
    # only), so E3's RMSE-by-gap is not optimistic training-set performance.
    X, y = prep["X_val"], prep["y_val"]
    gl = prep["gap_len_val"]
    corrector = prep["corrector"]
    assert len(gl) == len(X) == len(y), (
        f"gap/X/y length mismatch: {len(gl)}/{len(X)}/{len(y)}")

    # Predicted residual for every example
    pred_res = _predict_residual(corrector, X)  # shape (N,)
    # base error = |y| (since y = true - base); corrected error = |y - pred|
    base_abs = np.abs(y)
    corr_abs = np.abs(y - pred_res)

    # Bucket by gap length
    buckets = [(1, 1), (2, 6), (7, 12), (13, 24), (25, 48)]
    rows = []
    for lo, hi in buckets:
        m = (gl >= lo) & (gl <= hi)
        if not np.any(m):
            continue
        rows.append({
            "gap_bucket": f"{lo}-{hi}",
            "n_examples": int(m.sum()),
            "rmse_base": float(np.sqrt(np.mean(base_abs[m] ** 2))),
            "rmse_corrected": float(np.sqrt(np.mean(corr_abs[m] ** 2))),
            "mae_base": float(np.mean(base_abs[m])),
            "mae_corrected": float(np.mean(corr_abs[m])),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("E3 produced no gap buckets; check holdout mask/gap_len_norm.")
    expected_buckets = {"1-1", "2-6", "7-12", "13-24", "25-48"}
    seen_buckets = set(df["gap_bucket"].astype(str))
    if not expected_buckets.issubset(seen_buckets):
        raise RuntimeError(
            f"E3 gap coverage incomplete: expected {sorted(expected_buckets)}, "
            f"seen {sorted(seen_buckets)}. Refusing reviewer-facing output.")
    df["rmse_improvement_pct"] = 100 * (df["rmse_base"] - df["rmse_corrected"]) / df["rmse_base"]
    df.to_csv(out_dir / "longgap_rmse_by_bucket.csv", index=False)
    print(df.to_string(index=False))

    # Figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].bar(np.arange(len(df)) - 0.2, df["rmse_base"], width=0.4,
                  label="SAITS base")
        ax[0].bar(np.arange(len(df)) + 0.2, df["rmse_corrected"], width=0.4,
                  label="SAITS + PCR")
        ax[0].set_xticks(np.arange(len(df)))
        ax[0].set_xticklabels(df["gap_bucket"])
        ax[0].set_xlabel("gap length (steps)")
        ax[0].set_ylabel("RMSE (standardised)")
        ax[0].set_title("Correction quality vs gap length")
        ax[0].legend()
        ax[1].plot(np.arange(len(df)), df["rmse_improvement_pct"],
                   marker="o")
        ax[1].set_xticks(np.arange(len(df)))
        ax[1].set_xticklabels(df["gap_bucket"])
        ax[1].set_xlabel("gap length (steps)")
        ax[1].set_ylabel("RMSE reduction (%)")
        ax[1].set_title("PCR improvement by gap length")
        ax[1].axhline(0, color="gray", lw=0.5)
        fig.tight_layout()
        fig.savefig(out_dir / "longgap_analysis.png", dpi=140)
        plt.close(fig)
        print(f"[E3] figure saved to {out_dir/'longgap_analysis.png'}")
    except Exception as exc:
        print(f"[E3][WARN] figure not generated: {exc}")

    write_state(out_dir, {"experiment": "E3_longgap_viz",
                          "run_code_version": RUN_CODE_VERSION,
                          "dataset": dataset, "seed": seed,
                          "n_examples": int(len(gl)),
                          "n_train_examples": int(len(prep["X_train"])),
                          "n_pattern_fits": int(prep["n_pattern_fits"]),
                          "gap_buckets_seen": sorted(seen_buckets),
                          "saits_epochs": prep["epochs_used"]["saits_epochs"],
                          "brits_epochs": prep["epochs_used"]["brits_epochs"],
                          "allow_env_epochs": bool(prep["allow_env_epochs"]),
                          "env_epochs_ignored": bool(prep["env_epochs_ignored"])})
    append_result_row(out_dir, {"experiment": "E3", "dataset": dataset,
                                "seed": seed,
                                "run_code_version": RUN_CODE_VERSION,
                                "saits_epochs": prep["epochs_used"]["saits_epochs"],
                                "brits_epochs": prep["epochs_used"]["brits_epochs"],
                                "n_pattern_fits": int(prep["n_pattern_fits"]),
                                "n_buckets": int(len(df)),
                                "max_improvement_pct": float(df["rmse_improvement_pct"].max()),
                                "min_improvement_pct": float(df["rmse_improvement_pct"].min())})
    print("[E3] done.")


# ---------------------------------------------------------------------------
# E4: SHAP feature importance (R1.8)
# ---------------------------------------------------------------------------


def run_E4_shap(args, script_dir: Path) -> None:
    """Explain which PCR input features drive the residual correction, using
    SHAP if available and permutation importance otherwise. The method used is
    recorded so the reported numbers are never silently a different technique.
    """
    out_dir = exp_dir(script_dir, "E4_shap")
    print(f"[E4] writing to {out_dir}")
    seed = int(args.sensitivity_seed)
    dataset = args.sensitivity_dataset

    prep = _prepare_trained_corrector(script_dir, dataset, seed,
                                      allow_env_epochs=args.allow_env_epochs)
    # Compute importance on held-out VAL examples so it reflects generalisation,
    # not memorised training rows.
    X, y = prep["X_val"], prep["y_val"]
    corrector = prep["corrector"]
    feat_names = prep["feature_names"]

    # Predict function mapping feature matrix -> predicted residual
    def _predict(Xin):
        return _predict_residual(corrector, Xin)

    method = None
    importances = None

    # Try SHAP
    try:
        import shap
        # subsample for speed
        n = min(2000, X.shape[0])
        idx = np.random.default_rng(seed).choice(X.shape[0], n, replace=False)
        Xs = X[idx]
        bg = shap.sample(Xs, min(100, n), random_state=seed)
        explainer = shap.KernelExplainer(_predict, bg)
        sv = explainer.shap_values(Xs, nsamples=100, silent=True)
        # SHAP can return a list (one array per output) or a 3D array
        # (n_samples, n_features, n_outputs); normalise to (n_samples, n_features).
        arr = np.asarray(sv)
        if isinstance(sv, list):
            arr = np.asarray(sv[0])
        if arr.ndim == 3:
            arr = arr[..., 0]
        importances = np.abs(arr).mean(axis=0).reshape(-1)
        method = "shap_kernel"
    except Exception as exc:
        print(f"[E4][INFO] SHAP unavailable/failed ({exc}); "
              f"using permutation importance instead.")
        # Permutation importance on the residual MSE
        base_pred = _predict(X)
        base_mse = float(np.mean((base_pred - y) ** 2))
        rng = np.random.default_rng(seed)
        imp = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            mse_j = float(np.mean((_predict(Xp) - y) ** 2))
            imp[j] = mse_j - base_mse   # increase in error when feature broken
        importances = imp
        method = "permutation_importance"

    importances = np.asarray(importances).reshape(-1)
    # guard: importance vector must match the feature-name list length
    if len(importances) != len(feat_names):
        raise RuntimeError(
            f"importance length {len(importances)} != n_features "
            f"{len(feat_names)} — feature/shape misalignment")

    order = np.argsort(importances)[::-1]
    df = pd.DataFrame({
        "feature": [feat_names[i] for i in order],
        "importance": [float(importances[i]) for i in order],
    })
    df["importance_method"] = method
    df.to_csv(out_dir / "feature_importance.csv", index=False)
    print(f"[E4] importance method = {method}")
    print(df.to_string(index=False))

    # Figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(df["feature"][::-1], df["importance"][::-1])
        ax.set_xlabel(f"importance ({method})")
        ax.set_title("PCR corrector feature importance")
        fig.tight_layout()
        fig.savefig(out_dir / "feature_importance.png", dpi=140)
        plt.close(fig)
        print(f"[E4] figure saved to {out_dir/'feature_importance.png'}")
    except Exception as exc:
        print(f"[E4][WARN] figure not generated: {exc}")

    write_state(out_dir, {"experiment": "E4_shap",
                          "run_code_version": RUN_CODE_VERSION,
                          "dataset": dataset, "seed": seed,
                          "importance_method": method,
                          "n_val_examples": int(len(X)),
                          "n_train_examples": int(len(prep["X_train"])),
                          "n_pattern_fits": int(prep["n_pattern_fits"]),
                          "saits_epochs": prep["epochs_used"]["saits_epochs"],
                          "brits_epochs": prep["epochs_used"]["brits_epochs"],
                          "allow_env_epochs": bool(prep["allow_env_epochs"]),
                          "env_epochs_ignored": bool(prep["env_epochs_ignored"]),
                          "top_feature": df["feature"].iloc[0]})
    append_result_row(out_dir, {"experiment": "E4", "dataset": dataset,
                                "seed": seed,
                                "run_code_version": RUN_CODE_VERSION,
                                "saits_epochs": prep["epochs_used"]["saits_epochs"],
                                "brits_epochs": prep["epochs_used"]["brits_epochs"],
                                "n_pattern_fits": int(prep["n_pattern_fits"]),
                                "importance_method": method,
                                "top_feature": df["feature"].iloc[0],
                                "top_importance": float(df["importance"].iloc[0])})
    print("[E4] done.")


# ---------------------------------------------------------------------------
# E5: Correction baselines (R4.4)
# ---------------------------------------------------------------------------


def run_E5_correction_baselines(args, script_dir: Path) -> None:
    """Post-hoc sanity-check comparison against simple residual correctors.
    The comparison uses the same extracted PCR feature matrix for: no
    correction, mean-blend, a fixed tree regressor, and the learned MLP
    corrector. Because the MLP used parent VAL for early stopping, this is a
    sanity check rather than a headline, definitive model-selection result.
    """
    out_dir = exp_dir(script_dir, "E5_correction_baselines")
    print(f"[E5] writing to {out_dir}")
    seed = int(args.sensitivity_seed)
    dataset = args.sensitivity_dataset

    prep = _prepare_trained_corrector(script_dir, dataset, seed,
                                      allow_env_epochs=args.allow_env_epochs)
    corrector = prep["corrector"]
    fn = prep["feature_names"]

    # Fair protocol. The proposed MLP fit gradients on TRAIN and used the FULL
    # VAL for early stopping. To reduce (though not fully eliminate) the MLP's
    # val-based advantage, we split VAL into val_tune / val_test and run the E5
    # comparison on val_test only. The tree baseline is trained on TRAIN and
    # tuned on nothing (fixed hyperparameters), so val_test is unseen by every
    # method at comparison time. Residual caveat (disclosed in the manuscript):
    # the MLP's early-stopping used the parent VAL, of which val_test is a
    # subset, so a small optimistic bias may remain — acceptable for a
    # sanity-check baseline comparison, not a headline result.
    Xtr, ytr = prep["X_train"], prep["y_train"]
    X_val_all, y_val_all = prep["X_val"], prep["y_val"]
    rng = np.random.default_rng(seed)
    n_val = X_val_all.shape[0]
    perm = rng.permutation(n_val)
    n_test = max(1, int(0.5 * n_val))
    te_idx = perm[:n_test]
    Xte, yte = X_val_all[te_idx], y_val_all[te_idx]

    results = {}

    # (a) No correction: predicted residual = 0 -> corrected == base
    results["no_correction_saits"] = float(np.sqrt(np.mean(yte ** 2)))

    # (b) Mean-blend: corrected = 0.5*base + 0.5*local => residual = 0.5*(local-base)
    base_i = fn.index("base_imputed") if "base_imputed" in fn else 0
    local_i = fn.index("local_estimate") if "local_estimate" in fn else 1
    blend_res = 0.5 * (Xte[:, local_i] - Xte[:, base_i])
    results["mean_blend_base_local"] = float(np.sqrt(np.mean((yte - blend_res) ** 2)))

    # (c) Tree regressor trained on TRAIN, evaluated on held-out val_test
    tree_name = None
    try:
        from xgboost import XGBRegressor
        reg = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1,
                           subsample=0.8, random_state=seed, n_jobs=2)
        tree_name = "xgboost"
    except Exception:
        from sklearn.ensemble import GradientBoostingRegressor
        reg = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                        learning_rate=0.1, random_state=seed)
        tree_name = "sklearn_gbr"
    reg.fit(Xtr, ytr)
    tree_pred = reg.predict(Xte)
    results[f"tree_{tree_name}"] = float(np.sqrt(np.mean((yte - tree_pred) ** 2)))

    # (d) Proposed MLP corrector, evaluated on the SAME held-out val_test
    mlp_pred = _predict_residual(corrector, Xte)
    results["proposed_mlp"] = float(np.sqrt(np.mean((yte - mlp_pred) ** 2)))

    # Build result rows (RMSE on residual target; lower is better)
    base_rmse = results["no_correction_saits"]
    rows = []
    for name, rmse in results.items():
        rows.append({"corrector": name, "residual_rmse": rmse,
                     "improvement_vs_saits_pct":
                         100 * (base_rmse - rmse) / base_rmse})
    df = pd.DataFrame(rows).sort_values("residual_rmse").reset_index(drop=True)
    df["tree_impl"] = tree_name
    df.to_csv(out_dir / "correction_baselines.csv", index=False)
    print(df.to_string(index=False))

    write_state(out_dir, {"experiment": "E5_correction_baselines",
                          "run_code_version": RUN_CODE_VERSION,
                          "dataset": dataset, "seed": seed,
                          "tree_impl": tree_name,
                          "n_train_examples": int(len(Xtr)),
                          "n_parent_val_examples": int(n_val),
                          "n_val_test_examples": int(len(Xte)),
                          "n_pattern_fits": int(prep["n_pattern_fits"]),
                          "saits_epochs": prep["epochs_used"]["saits_epochs"],
                          "brits_epochs": prep["epochs_used"]["brits_epochs"],
                          "allow_env_epochs": bool(prep["allow_env_epochs"]),
                          "env_epochs_ignored": bool(prep["env_epochs_ignored"]),
                          "interpretation": "post-hoc sanity check; not a headline definitive comparison"})
    best = df.iloc[0]
    append_result_row(out_dir, {"experiment": "E5", "dataset": dataset,
                                "seed": seed,
                                "run_code_version": RUN_CODE_VERSION,
                                "saits_epochs": prep["epochs_used"]["saits_epochs"],
                                "brits_epochs": prep["epochs_used"]["brits_epochs"],
                                "n_pattern_fits": int(prep["n_pattern_fits"]),
                                "tree_impl": tree_name,
                                "best_corrector": str(best["corrector"]),
                                "proposed_mlp_rmse": results["proposed_mlp"],
                                "proposed_improvement_pct":
                                    float(100 * (base_rmse - results["proposed_mlp"]) / base_rmse)})
    print("[E5] done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PCR-SAITS Patch 4: post-hoc analysis (E3, E4, E5).")
    p.add_argument("--run-E3", action="store_true",
                   help="Gap-length-stratified analysis (R1.7)")
    p.add_argument("--run-E4", action="store_true",
                   help="SHAP / feature importance (R1.8)")
    p.add_argument("--run-E5", action="store_true",
                   help="Correction-baseline sanity check vs MLP (R4.4)")
    p.add_argument("--run-all", action="store_true",
                   help="Run E3, E4, E5")
    p.add_argument("--allow-env-epochs", action="store_true",
                   help=("Allow SAITS_EPOCHS/BRITS_EPOCHS environment overrides. "
                         "Default: ignore them and restore v7.1 defaults."))

    p.add_argument("--sensitivity-seed", type=int, default=7)
    p.add_argument("--sensitivity-dataset", type=str, default="uci_air_quality",
                   choices=["uci_air_quality", "beijing_pm25"])
    return p


def main() -> None:
    args = build_parser().parse_args()
    script_dir = THIS_DIR

    if args.run_all:
        args.run_E3 = args.run_E4 = args.run_E5 = True

    if not any([args.run_E3, args.run_E4, args.run_E5]):
        print("Nothing to do. Use --run-all or --run-E3/E4/E5.")
        build_parser().print_help()
        return

    root = revision_root(script_dir)
    (root / "patch4_session.json").write_text(json.dumps({
        "session_started": now_stamp(),
        "run_code_version": RUN_CODE_VERSION,
        "device": choose_device(),
        "script": __file__,
        "allow_env_epochs": bool(args.allow_env_epochs),
    }, indent=2))

    if args.run_E3:
        run_E3_longgap_viz(args, script_dir)
    if args.run_E4:
        run_E4_shap(args, script_dir)
    if args.run_E5:
        run_E5_correction_baselines(args, script_dir)

    print(f"\n[DONE] Patch 4 finished. Outputs in {root}")


if __name__ == "__main__":
    main()