#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCR-SAITS Revision-1 — Patch 2: New-Domain Datasets & Downstream Task
=====================================================================
Adds the experiments that require datasets outside the environmental-sensor
domain, plus a downstream-utility experiment, as requested by ESWA reviewers.

Experiments added in this patch
-------------------------------
E9  (R1.10)  ETT-h1  full pipeline   — electricity transformer / power load
E10 (R1.10)  PhysioNet 2012 pipeline — medical ICU vital signs
E14 (R4.12)  Downstream forecasting on ETT-h1 imputed series

Design principles (must match the paper protocol)
-------------------------------------------------
* Every scenario grid is the FULL paper grid: 3 mechanisms x 5 patterns x
  5 rates x 5 seeds = 375 scenarios per method (via full_grid=True on the
  v7.1 runner). No reduced grids for reviewer-facing numbers.
* Proposed method is pcrsaitsv14_no_seasonal_branch, compared against SAITS,
  with the same linear / seasonal_naive24 / BRITS baselines that appear in
  the paper's main table so the new-domain results can sit in the same table.
* Datasets are loaded through the PyPOTS ecosystem (BenchPOTS + TSDB), which
  auto-downloads ETT and PhysioNet-2012. We monkey-patch v7.1's dataset
  loader so its "airquality" code path transparently receives the new frame,
  reusing all of v7.1's masking / training / evaluation code unchanged.

Because the two new datasets have different shapes (ETT is one long regular
series; PhysioNet is many short 48-step ICU stays), E10 flattens PhysioNet
into a single long multivariate series by concatenating patient stays, which
is the standard way to feed a windowed imputer a multi-sample benchmark.

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
# Import v7.1 base module + reuse the extension helpers from Patch 1
# ---------------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
RUN_CODE_VERSION = "patch2_e9_e10_v6_readonly_sort_safe"
FULL_GRID_SEED_SET = "7,21,42,123,456"


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
    # glob fallback: strip .py and any trailing markers, match loosely
    for name in pattern_list:
        stem = name.replace(".py", "").split("*")[0]
        # try progressively shorter prefixes so *_E11_fixed.py etc. still match
        for probe in (stem, "_".join(stem.split("_")[:5]),
                      "_".join(stem.split("_")[:4])):
            if not probe:
                continue
            hits = sorted(THIS_DIR.glob(f"*{probe}*.py"))
            if hits:
                return hits[0]
    raise FileNotFoundError(
        f"Could not find any of {pattern_list} in {THIS_DIR}")


# Reuse Patch-1 helpers (checkpointing, output dirs). Load Patch 1 FIRST;
# it imports v7.1 as its module-level `_v71`. We then reuse THAT SAME v7.1
# instance so that monkey-patching here affects exactly the module whose
# run_experiment / build_parser the shared helpers call. Loading a second,
# independent v7.1 instance would mean patches land on a different object
# than the one _build_inner_namespace uses (Bug #1).
_ext_path = _find([
    "pcr_saits_v1_5_main_extensions.py",
])
_ext = _load_module(_ext_path, "pcr_ext")

# CRITICAL: use Patch 1's v7.1 instance, not a fresh one.
_v71 = _ext._v71

# Pull shared helpers
now_stamp = _v71.now_stamp
choose_device = _v71.choose_device
compute_metrics = _v71.compute_metrics
revision_root = _ext.revision_root
exp_dir = _ext.exp_dir
load_completed_keys = _ext.load_completed_keys
append_result_row = _ext.append_result_row
write_state = _ext.write_state
make_key = _ext.make_key
_build_inner_namespace = _ext._build_inner_namespace


# ---------------------------------------------------------------------------
# Dataset loaders via PyPOTS / BenchPOTS
# ---------------------------------------------------------------------------


def _safe_feature_names(raw, n: int, prefix: str) -> List[str]:
    """Coerce a possibly-ndarray / pandas.Index / None feature-name object
    into a plain Python list of length n. Using `if not raw` on an ndarray
    raises 'truth value is ambiguous', so we convert first (Bug #2)."""
    if raw is None:
        return [f"{prefix}_{i:02d}" for i in range(n)]
    try:
        names = list(raw)
    except Exception:
        return [f"{prefix}_{i:02d}" for i in range(n)]
    if len(names) != n:
        return [f"{prefix}_{i:02d}" for i in range(n)]
    return [str(x) for x in names]


def _import_benchpots():
    """Import benchpots lazily and return the datasets module, or raise a
    clear error telling the user what to install."""
    try:
        import benchpots.datasets as bd  # noqa
        return bd
    except Exception as exc:
        raise RuntimeError(
            "benchpots is required for the new-domain datasets (E9/E10/E14). "
            "Install with:  pip install pypots benchpots tsdb pygrinder\n"
            f"Underlying import error: {exc}"
        )


def load_ett_h1_frame(n_rows: Optional[int] = None,
                      allow_windowed_fallback: bool = False
                      ) -> Tuple[pd.DataFrame, List[str]]:
    """Load ETT-h1 as a single long multivariate DataFrame with a timestamp
    column plus the 7 ETT channels (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT).

    Uses BenchPOTS/TSDB, which downloads the raw ETT CSV automatically. We do
    NOT use benchpots' own missingness injection here — v7.1 performs all
    masking. We only need the clean series.
    """
    # Preferred: pull the raw series through TSDB (clean, no injected gaps).
    try:
        import tsdb
        data = tsdb.load("electricity_transformer_temperature")
        # tsdb returns a dict; ETTh1 lives under a well-known key
        df = None
        for key in ("ETTh1", "etth1", "ETT_h1"):
            if isinstance(data, dict) and key in data:
                df = data[key]
                break
        if df is None and isinstance(data, dict):
            # Some versions nest under 'dataframes'
            dfs = data.get("dataframes") or {}
            for key in ("ETTh1", "etth1"):
                if key in dfs:
                    df = dfs[key]
                    break
        if df is None:
            raise KeyError("ETTh1 not found in tsdb payload")
        df = df.reset_index()
    except Exception as exc:
        # Reviewer-facing E9 must use the raw long ETTh1 series. BenchPOTS'
        # preprocess_ett may return windowed samples; flattening those windows
        # can duplicate/permute time points and change the experiment. Therefore
        # the fallback is disabled by default and only allowed for smoke tests.
        if not allow_windowed_fallback:
            raise RuntimeError(
                "Could not load raw ETTh1 via tsdb. Refusing to use the "
                "BenchPOTS windowed fallback for reviewer-facing E9 because "
                "it can reshape sliding windows into a non-raw time series. "
                "Install/fix tsdb raw ETTh1 loading, or rerun with "
                "--allow-ett-windowed-fallback for smoke testing only. "
                f"Underlying error: {exc}"
            )
        print("[E9][WARN] using BenchPOTS windowed fallback. "
              "Do NOT use this as reviewer-facing final E9.")
        bd = _import_benchpots()
        prep = bd.preprocess_ett(subset="ETTh1", rate=0.0, pattern="point",
                                 n_steps=48)
        # prep['train_X'] can be windowed samples; this path is smoke-test only.
        X = prep["train_X"]
        if hasattr(X, "numpy"):
            X = X.numpy()
        X = np.asarray(X, dtype=float)
        feats = _safe_feature_names(prep.get("feature_names"),
                                    X.shape[-1], "ETT")
        flat = X.reshape(-1, X.shape[-1])
        df = pd.DataFrame(flat, columns=feats)
        df.insert(0, "timestamp",
                  pd.date_range("2016-07-01", periods=len(df), freq="h"))

    # Normalise column naming
    time_col = None
    for c in df.columns:
        if str(c).lower() in ("date", "timestamp", "index"):
            time_col = c
            break
    if time_col is None:
        df.insert(0, "timestamp",
                  pd.date_range("2016-07-01", periods=len(df), freq="h"))
        time_col = "timestamp"
    else:
        df = df.rename(columns={time_col: "timestamp"})
        time_col = "timestamp"
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    feats = [c for c in df.columns if c != "timestamp"]
    # keep only numeric feature columns
    feats = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    df = df[["timestamp"] + feats].copy()

    if n_rows is not None and len(df) > n_rows:
        df = df.iloc[:n_rows].reset_index(drop=True)

    print(f"[E9][DATA] ETT-h1 loaded: {df.shape[0]} rows x {len(feats)} feats "
          f"({feats})")
    return df, feats


def _manual_preprocess_physionet2012_seta(random_state: int = 7) -> Dict[str, Any]:
    """Manual fallback for BenchPOTS PhysioNet-2012 preprocessing.

    Some BenchPOTS/pandas combinations raise
    KeyError("['RecordID'] not found in axis") inside
    preprocess_physionet2012(), because pandas differs on whether
    RecordID remains a column after groupby-apply. The current upstream
    implementation handles this with errors="ignore", but older installed
    wheels may not. This fallback implements the same essential set-a path
    from raw TSDB data, returning train_X/val_X/test_X arrays.
    """
    try:
        import tsdb
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        raise RuntimeError(
            "Manual PhysioNet fallback requires tsdb and scikit-learn. "
            f"Underlying import error: {exc}"
        )

    data = tsdb.load("physionet_2012")
    if "set-a" not in data:
        raise KeyError("TSDB payload does not contain 'set-a'")

    df = data["set-a"].copy().reset_index(drop=True)
    if "RecordID" not in df.columns or "Time" not in df.columns:
        raise KeyError(
            f"PhysioNet set-a must contain RecordID and Time columns; got {list(df.columns)}"
        )

    # Match BenchPOTS intent: drop static features except ICUType, because
    # ICUType is retained only for stratified metadata and then removed from X.
    static_features = list(data.get("static_features", []))
    drop_static = [c for c in static_features if c != "ICUType" and c in df.columns]
    if drop_static:
        df = df.drop(columns=drop_static, errors="ignore")

    def _pad_truncate_one_stay(stay: pd.DataFrame) -> pd.DataFrame:
        stay = stay.copy()
        stay["Time"] = pd.to_numeric(stay["Time"], errors="coerce")
        stay = stay.dropna(subset=["Time"])
        stay["Time"] = stay["Time"].astype(int)
        missing = sorted(set(range(48)).difference(set(stay["Time"].tolist())))
        if missing:
            stay = pd.concat([stay, pd.DataFrame({"Time": missing})], ignore_index=True, sort=False)
        stay = stay.set_index("Time").sort_index().reset_index()
        return stay.iloc[:48]

    X = df.groupby("RecordID", group_keys=True).apply(_pad_truncate_one_stay)
    # pandas-version guard: RecordID may or may not still be a normal column.
    X = X.drop(columns=["RecordID"], errors="ignore").reset_index()
    X = X.drop(columns=["level_1"], errors="ignore")

    if "RecordID" not in X.columns:
        raise KeyError(
            "Manual fallback could not recover RecordID after groupby/reset_index"
        )

    # Keep ICUType only as metadata; do not feed it as a temporal variable.
    X = X.drop(columns=["ICUType"], errors="ignore")

    feature_names = [c for c in X.columns if c not in ("RecordID", "Time")]
    # Keep numeric feature columns only.
    feature_names = [c for c in feature_names if pd.api.types.is_numeric_dtype(X[c])]
    if not feature_names:
        raise RuntimeError("No numeric PhysioNet features found after preprocessing")

    # pandas/NumPy may return a read-only array from unique()/to_numpy() on some
    # versions. Do NOT call in-place .sort(); build a fresh writable array.
    all_ids = np.array(sorted(pd.Series(X["RecordID"].unique()).dropna().tolist()))

    # Prefer the same positive/negative split logic as BenchPOTS when outcomes exist.
    y = data.get("outcomes-a")
    try:
        y = y.loc[all_ids]
        if isinstance(y, pd.DataFrame) and "In-hospital_death" in y.columns:
            positive_ids = y.index[y["In-hospital_death"] == 1].to_numpy()
        else:
            yy = pd.Series(np.asarray(y).reshape(-1), index=all_ids)
            positive_ids = yy.index[yy == 1].to_numpy()
        negative_ids = np.setdiff1d(all_ids, positive_ids)

        if len(positive_ids) >= 3 and len(negative_ids) >= 3:
            tr_pos, te_pos = train_test_split(positive_ids, test_size=0.2, random_state=random_state)
            tr_pos, va_pos = train_test_split(tr_pos, test_size=0.2, random_state=random_state)
            tr_neg, te_neg = train_test_split(negative_ids, test_size=0.2, random_state=random_state)
            tr_neg, va_neg = train_test_split(tr_neg, test_size=0.2, random_state=random_state)
            train_ids = np.sort(np.concatenate([tr_pos, tr_neg]))
            val_ids = np.sort(np.concatenate([va_pos, va_neg]))
            test_ids = np.sort(np.concatenate([te_pos, te_neg]))
        else:
            raise ValueError("Too few positive/negative IDs for class-preserving split")
    except Exception:
        train_ids, test_ids = train_test_split(all_ids, test_size=0.2, random_state=random_state)
        train_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=random_state)
        train_ids, val_ids, test_ids = map(np.sort, (train_ids, val_ids, test_ids))

    def _ids_to_flat(ids: np.ndarray) -> np.ndarray:
        part = X[X["RecordID"].isin(ids)].sort_values(["RecordID", "Time"])
        arr = part[feature_names].to_numpy(dtype=float)
        expected = len(ids) * 48
        if arr.shape[0] != expected:
            raise RuntimeError(
                f"Expected {expected} rows for {len(ids)} stays, got {arr.shape[0]}"
            )
        return arr

    train_flat = _ids_to_flat(train_ids)
    val_flat = _ids_to_flat(val_ids)
    test_flat = _ids_to_flat(test_ids)

    scaler = StandardScaler()
    train_flat = scaler.fit_transform(train_flat)
    val_flat = scaler.transform(val_flat)
    test_flat = scaler.transform(test_flat)

    n_features = len(feature_names)
    return {
        "train_X": train_flat.reshape(len(train_ids), 48, n_features),
        "val_X": val_flat.reshape(len(val_ids), 48, n_features),
        "test_X": test_flat.reshape(len(test_ids), 48, n_features),
        "feature_names": [str(c) for c in feature_names],
        "source": "manual_tsdb_recordid_fallback",
    }


def load_physionet2012_frame(max_patients: int = 2000,
                             random_state: int = 7,
                             use_all_splits: bool = True,
                             ) -> Tuple[pd.DataFrame, List[str]]:
    """Load PhysioNet-2012 and flatten patient ICU stays into one long
    multivariate surrogate series.

    v5 fix: BenchPOTS preprocessing can fail in some installed versions with
    KeyError("['RecordID'] not found in axis"). We first try BenchPOTS; if it
    fails with that known issue, we fall back to a local TSDB-based
    implementation that mirrors the essential set-a preprocessing path.
    """
    bd = _import_benchpots()
    try:
        prep = bd.preprocess_physionet2012(subset="set-a", rate=0.0,
                                           pattern="point",
                                           random_state=int(random_state))
        prep_source = "benchpots_preprocess_physionet2012"
    except KeyError as exc:
        if "RecordID" not in str(exc):
            raise
        print("[E10][WARN] BenchPOTS preprocess_physionet2012 failed with "
              f"{exc}; using manual TSDB RecordID-safe fallback.")
        prep = _manual_preprocess_physionet2012_seta(random_state=int(random_state))
        prep_source = prep.get("source", "manual_tsdb_recordid_fallback")

    def _to_np(name: str) -> np.ndarray:
        arr = prep[name]
        if hasattr(arr, "numpy"):
            arr = arr.numpy()
        return np.asarray(arr, dtype=float)

    train_X = _to_np("train_X")
    val_X = _to_np("val_X") if "val_X" in prep else np.empty((0,) + train_X.shape[1:])
    test_X = _to_np("test_X") if "test_X" in prep else np.empty((0,) + train_X.shape[1:])

    if use_all_splits:
        X = np.concatenate([train_X, val_X, test_X], axis=0)
        split_desc = (f"train+val+test ({train_X.shape[0]}+"
                      f"{val_X.shape[0]}+{test_X.shape[0]})")
    else:
        X = train_X
        split_desc = f"train_only ({train_X.shape[0]})"

    n_pat = min(int(max_patients), X.shape[0])
    X = X[:n_pat]
    n, steps, feats_n = X.shape

    feat_names = _safe_feature_names(prep.get("feature_names"),
                                     feats_n, "vital")

    # PhysioNet has heavy natural missingness. v7.1's artificial-mask metrics
    # require observed ground truth, so E10 builds a fully observed surrogate
    # series before applying the same artificial masking protocol as the main
    # paper experiments.
    with np.errstate(all="ignore"):
        global_median = np.nanmedian(X.reshape(n * steps, feats_n), axis=0)
    global_median = np.where(np.isfinite(global_median), global_median, 0.0)

    cleaned = np.empty_like(X)
    for p in range(n):
        stay = pd.DataFrame(X[p], columns=feat_names)
        stay = stay.interpolate(method="linear", limit_direction="both",
                                axis=0)
        for j, c in enumerate(feat_names):
            if stay[c].isna().any():
                stay[c] = stay[c].fillna(global_median[j])
        cleaned[p] = stay.to_numpy()
    cleaned = np.nan_to_num(cleaned, nan=0.0)

    natural_missing = float(np.isnan(X).mean())
    flat = cleaned.reshape(n * steps, feats_n)
    df = pd.DataFrame(flat, columns=feat_names)
    df.insert(0, "timestamp",
              pd.date_range("2012-01-01", periods=len(df), freq="h"))

    print(f"[E10][DATA] PhysioNet-2012 loaded via {prep_source}: {split_desc}; "
          f"using {n} patients -> {df.shape[0]} rows x {feats_n} feats "
          f"| random_state={int(random_state)} "
          f"| natural missing cleaned: {natural_missing:.1%}")
    print(f"[E10][LIMITATION] patient stays concatenated into one long series; "
          f"reported E10 numbers are on a cleaned surrogate series under "
          f"artificial masking. If v7.1 uses sliding windows, many windows "
          f"may straddle patient boundaries; report this as robustness/"
          f"supplementary or implement patient-blocked evaluation.")
    return df, list(feat_names)

# ---------------------------------------------------------------------------
# Generic new-dataset runner (E9, E10) — reuses v7.1 via monkey-patch
# ---------------------------------------------------------------------------


def _run_new_dataset(script_dir: Path,
                     exp_tag: str,
                     exp_code: str,
                     comment: str,
                     frame_loader,
                     seed: int,
                     all_methods: bool,
                     allow_env_epochs: bool = False,
                     ) -> None:
    """Run the full paper protocol (375 scenarios) with the proposed method
    and baselines on a NEW dataset, by monkey-patching v7.1's air-quality
    loader to return our frame. Results are checkpointed under
    output_revision1/<exp_tag>/.
    """
    out_dir = exp_dir(script_dir, exp_tag)
    print(f"[{exp_code}] writing to {out_dir}  ({comment})")

    # Bug #3 fix: a previous run may have appended an ERROR/WARNING row (e.g.
    # download failed). We must NOT treat that as completed. Read the result
    # CSV directly and consider this (seed, dataset) done only if there is a
    # row with matching seed/dataset that has NO 'error' and NO 'warning'.
    def _already_succeeded(expected_n_rows: int,
                           expected_n_features: int,
                           expected_seed_set: str) -> bool:
        ck = out_dir / "checkpoint_results.csv"
        if not ck.exists():
            return False
        try:
            prev = pd.read_csv(ck)
        except Exception:
            return False
        if prev.empty or "dataset" not in prev.columns:
            return False
        sub = prev[prev["dataset"].astype(str) == exp_tag]
        if sub.empty:
            return False

        required = ["rmse_linear", "rmse_seasonal", "rmse_brits",
                    "rmse_saits", "rmse_pcr", "delta_rmse_pct"]
        meta_required = ["run_code_version", "n_rows", "n_features",
                         "loader_calls", "seed_set", "saits_epochs",
                         "brits_epochs"]
        for _, r in sub.iterrows():
            has_err = ("error" in sub.columns and pd.notna(r.get("error")))
            has_warn = ("warning" in sub.columns and pd.notna(r.get("warning")))
            has_all = all((c in sub.columns and pd.notna(r.get(c)))
                          for c in required)
            has_meta = all((c in sub.columns and pd.notna(r.get(c)))
                           for c in meta_required)
            same_version = str(r.get("run_code_version", "")) == RUN_CODE_VERSION
            same_shape = (str(r.get("n_rows", "")) == str(expected_n_rows)
                          and str(r.get("n_features", "")) == str(expected_n_features))
            same_seed_set = str(r.get("seed_set", "")) == expected_seed_set
            loader_ok = False
            try:
                loader_ok = int(float(r.get("loader_calls", 0))) > 0
            except Exception:
                loader_ok = False
            if (has_all and has_meta and same_version and same_shape
                    and same_seed_set and loader_ok and not has_err and not has_warn):
                return True
        return False

    # Load the clean frame once. We intentionally load before the skip check so
    # completion is validated against the exact current row/feature shape and
    # cannot silently reuse results from a different dataset cap or protocol.
    frame, feats = frame_loader()
    expected_seed_set = FULL_GRID_SEED_SET
    if _already_succeeded(int(frame.shape[0]), len(feats), expected_seed_set):
        print(f"[{exp_code}] already completed successfully "
              f"({exp_tag}, full method set, version={RUN_CODE_VERSION}) — skipping")
        return

    # Cache to CSV for provenance
    prov_csv = out_dir / f"{exp_tag}_source.csv"
    try:
        frame.to_csv(prov_csv, index=False)
    except Exception:
        pass

    # AR(1)-style diagnostic so the log proves which frame was used.
    try:
        _obs = frame[feats].to_numpy(dtype=float)
        _acs = []
        for j in range(_obs.shape[1]):
            col = _obs[:, j]
            col = col[np.isfinite(col)]
            if col.size > 5:
                _acs.append(np.corrcoef(col[:-1], col[1:])[0, 1])
        _mean_lag1 = float(np.nanmean(_acs)) if _acs else float("nan")
    except Exception:
        _mean_lag1 = float("nan")
    print(f"[{exp_code}][DATA-CHECK] {exp_tag} frame shape={frame.shape} "
          f"| {len(feats)} feats | mean_lag1={_mean_lag1:.3f}")

    _limitation = None
    if "physionet" in exp_tag.lower():
        _limitation = ("PhysioNet has ~80% natural missingness; it was cleaned "
                       "(within-stay interpolation + global-median fill) to "
                       "produce a fully-observed base series, then patient "
                       "stays were concatenated. Reported numbers are on this "
                       "cleaned surrogate series under artificial masking, and "
                       "a minority of windows straddle patient boundaries. "
                       "Report as robustness/supplementary with this "
                       "limitation, or switch to patient-blocked evaluation.")
        print(f"[{exp_code}][PROTOCOL-NOTE] {_limitation}")

    write_state(out_dir, {"experiment": exp_tag, "comment": comment,
                          "run_code_version": RUN_CODE_VERSION,
                          "n_rows": int(frame.shape[0]),
                          "n_features": len(feats),
                          "features": feats,
                          "mean_lag1": _mean_lag1,
                          "protocol_limitation": _limitation,
                          "seed": seed})

    # Build a v7.1 namespace pointing at the "airquality" code path with the
    # full grid; we then patch the loader to hand back our frame.
    ckpt_subdir = out_dir / "model_checkpoints"
    ckpt_subdir.mkdir(parents=True, exist_ok=True)

    base_args = _build_inner_namespace(
        script_dir=script_dir,
        dataset="uci_air_quality",   # routed; loader is patched below
        seed=seed,
        window=48,
        ar1_csv_path=None,
        ckpt_subdir=ckpt_subdir,
        full_grid=True,
        all_methods=all_methods,
    )

    # ---- issue #1 fix: force fresh training on the NEW dataset ----
    # The shared namespace sets reset_checkpoints/retrain_models = False, which
    # is right for reusing a SAITS backbone across E1/E2 configs on the SAME
    # dataset. But E9/E10 are DIFFERENT datasets; a checkpoint left in this
    # folder from an earlier (possibly mis-loaded) run must never be reused, or
    # the reported numbers would not be for ETT/PhysioNet. Force retrain.
    base_args.reset_checkpoints = True
    base_args.retrain_models = True

    # ---- issue #3 guard: refuse silent smoke-test epoch overrides ----
    # If SAITS_EPOCHS / BRITS_EPOCHS were exported for quick tests, the shared
    # builder would have shrunk training. For a reviewer-facing run we restore
    # the model defaults unless the user explicitly opts in via
    # --allow-env-epochs, and we always record the epochs actually used.
    if not allow_env_epochs:
        try:
            _defaults = _v71.build_parser().parse_args([])
            base_args.saits_epochs = _defaults.saits_epochs
            base_args.brits_epochs = _defaults.brits_epochs
        except Exception:
            pass
    _epochs_used = {"saits_epochs": int(getattr(base_args, "saits_epochs", -1)),
                    "brits_epochs": int(getattr(base_args, "brits_epochs", -1))}
    print(f"[{exp_code}][TRAIN-CONFIG] retrain=True reset_ckpt=True "
          f"saits_epochs={_epochs_used['saits_epochs']} "
          f"brits_epochs={_epochs_used['brits_epochs']}")
    # v7.1's real dispatcher is load_air_quality_dataset (it inspects the
    # path and routes to synthetic/beijing/uci). We replace it AND the
    # underlying synthetic loader, so no matter which path v7.1 resolves for
    # our routed "airquality" request, it receives OUR frame. There is no
    # load_dataset() in v7.1 (verified), so we do not patch that name.
    _orig_load_aq = _v71.load_air_quality_dataset
    _orig_load_synth = _v71.load_synthetic_dataset
    _loader_called = {"n": 0}

    def _patched_loader(path=None, *a, **kw):
        _loader_called["n"] += 1
        print(f"[{exp_code}][LOADER-CHECK] serving {exp_tag} frame "
              f"(call #{_loader_called['n']}) shape={frame.shape}")
        return frame.copy(), list(feats)

    _v71.load_air_quality_dataset = _patched_loader
    _v71.load_synthetic_dataset = _patched_loader
    _v71._ACTIVE_DATASET_CACHE = None

    try:
        summary_df, feature_df, meta = _v71.run_experiment(base_args)
    except Exception as exc:
        print(f"[{exp_code}][ERROR] {exc}")
        append_result_row(out_dir, {"experiment": exp_code,
                                    "run_code_version": RUN_CODE_VERSION,
                                    "outer_case_seed": seed,
                                    "seed_set": FULL_GRID_SEED_SET,
                                    "dataset": exp_tag,
                                    "n_rows": int(frame.shape[0]),
                                    "n_features": len(feats),
                                    "error": str(exc)})
        return
    finally:
        _v71.load_air_quality_dataset = _orig_load_aq
        _v71.load_synthetic_dataset = _orig_load_synth
        _v71._ACTIVE_DATASET_CACHE = None

    if _loader_called["n"] == 0:
        # Hard failure: v7.1 produced results without ever calling our loader,
        # so the numbers are NOT for this dataset. Never record them.
        err = (f"patched loader was never called — v7.1 used a cached/other "
               f"dataset path; refusing to record {exp_tag} results.")
        print(f"[{exp_code}][FATAL] {err}")
        append_result_row(out_dir, {"experiment": exp_code,
                                    "run_code_version": RUN_CODE_VERSION,
                                    "outer_case_seed": seed,
                                    "seed_set": FULL_GRID_SEED_SET,
                                    "dataset": exp_tag,
                                    "n_rows": int(frame.shape[0]),
                                    "n_features": len(feats),
                                    "loader_calls": _loader_called["n"],
                                    "error": err})
        raise RuntimeError(err)

    # ---- extract results (all methods present) ----
    if summary_df is None or summary_df.empty:
        append_result_row(out_dir, {"experiment": exp_code,
                                    "run_code_version": RUN_CODE_VERSION,
                                    "outer_case_seed": seed,
                                    "seed_set": FULL_GRID_SEED_SET,
                                    "dataset": exp_tag,
                                    "n_rows": int(frame.shape[0]),
                                    "n_features": len(feats),
                                    "loader_calls": _loader_called["n"],
                                    "warning": "empty summary"})
        return

    def _col(df, *cs):
        for c in cs:
            if c in df.columns:
                return c
        return None

    rmse_col = _col(summary_df, "overall_rmse_mean", "rmse")
    mae_col = _col(summary_df, "overall_mae_mean", "mae")
    imp_col = _col(summary_df, "rmse_improvement_vs_reference_pct_mean")

    # issue #2 fix: full_grid uses 5 seeds (7,21,42,123,456), so the numbers
    # are aggregated over that set. Record the real seed set (and the epochs
    # actually used) rather than a single misleading seed=7 label.
    seed_set = ",".join(str(s) for s in getattr(base_args, "seeds", [seed]))
    row = {"experiment": exp_code, "dataset": exp_tag,
           "run_code_version": RUN_CODE_VERSION,
           "outer_case_seed": seed, "seed_set": seed_set,
           "n_rows": int(frame.shape[0]), "n_features": len(feats),
           "mean_lag1": _mean_lag1,
           "loader_calls": int(_loader_called["n"]),
           "protocol_limitation": _limitation,
           "saits_epochs": _epochs_used["saits_epochs"],
           "brits_epochs": _epochs_used["brits_epochs"]}
    for mname, prefix in [("linear", "linear"),
                          ("seasonal_naive24", "seasonal"),
                          ("brits", "brits"),
                          ("saits", "saits"),
                          ("pcrsaitsv14_no_seasonal_branch", "pcr")]:
        msub = summary_df[summary_df["method"] == mname]
        if not msub.empty:
            if rmse_col:
                row[f"rmse_{prefix}"] = float(msub[rmse_col].iloc[0])
            if mae_col:
                row[f"mae_{prefix}"] = float(msub[mae_col].iloc[0])

    prop = summary_df[summary_df["method"] == "pcrsaitsv14_no_seasonal_branch"]
    if not prop.empty and imp_col and pd.notna(prop[imp_col].iloc[0]):
        row["delta_rmse_pct"] = float(prop[imp_col].iloc[0])
    elif "rmse_pcr" in row and "rmse_saits" in row and row["rmse_saits"] > 0:
        row["delta_rmse_pct"] = (
            100 * (row["rmse_saits"] - row["rmse_pcr"]) / row["rmse_saits"])

    append_result_row(out_dir, row)
    print(f"[{exp_code}] done. seed_set={seed_set} delta_rmse_pct="
          f"{row.get('delta_rmse_pct', float('nan')):.3f}")


# ---------------------------------------------------------------------------
# E9: ETT-h1
# ---------------------------------------------------------------------------


def run_E9_ett(args, script_dir: Path) -> None:
    _run_new_dataset(
        script_dir=script_dir,
        exp_tag="E9_ett_h1",
        exp_code="E9",
        comment="R1.10 new domain: power load (ETT-h1)",
        frame_loader=lambda: load_ett_h1_frame(
            n_rows=args.ett_n_rows,
            allow_windowed_fallback=args.allow_ett_windowed_fallback),
        seed=int(args.sensitivity_seed),
        all_methods=True,   # full Table-3 method set on the new domain
        allow_env_epochs=args.allow_env_epochs,
    )


# ---------------------------------------------------------------------------
# E10: PhysioNet 2012
# ---------------------------------------------------------------------------


def run_E10_physionet(args, script_dir: Path) -> None:
    _run_new_dataset(
        script_dir=script_dir,
        exp_tag="E10_physionet2012",
        exp_code="E10",
        comment="R1.10 new domain: medical ICU (PhysioNet 2012)",
        frame_loader=lambda: load_physionet2012_frame(
            max_patients=args.physionet_max_patients,
            random_state=args.physionet_random_state,
            use_all_splits=not args.physionet_train_only),
        seed=int(args.sensitivity_seed),
        all_methods=True,
        allow_env_epochs=args.allow_env_epochs,
    )


# ---------------------------------------------------------------------------
# E14: Downstream forecasting on ETT-h1
# ---------------------------------------------------------------------------


def _direct_forecast_mae(series_2d: np.ndarray,
                         history: int = 96,
                         horizon: int = 24,
                         stride: int = 24,
                         ) -> float:
    """A minimal, dependency-free forecaster: for each window, predict the
    next `horizon` steps of every channel as the last observed value
    (persistence) plus the mean per-step delta over the history window.
    Returns mean absolute error over all forecast windows/channels.

    This is deliberately simple; E14 measures the RELATIVE downstream effect
    of better imputation, so the absolute forecaster choice is secondary as
    long as it is held fixed across imputation methods.
    """
    T, F = series_2d.shape
    errs = []
    t = history
    while t + horizon <= T:
        hist = series_2d[t - history:t]           # (history, F)
        fut = series_2d[t:t + horizon]            # (horizon, F)
        # per-channel average slope over history
        slope = (hist[-1] - hist[0]) / max(history - 1, 1)
        last = hist[-1]
        steps = np.arange(1, horizon + 1).reshape(-1, 1)
        pred = last.reshape(1, -1) + steps * slope.reshape(1, -1)
        errs.append(np.nanmean(np.abs(pred - fut)))
        t += stride
    return float(np.nanmean(errs)) if errs else float("nan")


def run_E14_downstream(args, script_dir: Path) -> None:
    """Downstream utility (R4.12): impute an ETT-h1 series with induced gaps
    using (a) SAITS and (b) SAITS+PCR, then run an identical fixed forecaster
    on each imputed series and compare forecasting MAE.

    IMPORTANT — NOT REVIEWER-READY IN THIS PATCH.
    A correct E14 must impute with the ACTUAL trained SAITS backbone and the
    ACTUAL PCR corrector, then forecast on each imputed series. That requires
    per-position imputed series exported from the real pipeline, which is the
    same infrastructure E13 (confidence gating) needs. Until that
    `--save-per-position` export exists, the only thing runnable here would be
    a linear-interpolation *proxy* with a hand-rolled "correction", which does
    NOT involve SAITS or PCR at all and therefore must not be reported as the
    downstream utility of the proposed method.

    This function is intentionally gated off so no proxy numbers leak into the
    revision. E14 will be implemented together with E13 on the shared
    per-position export path.
    """
    out_dir = exp_dir(script_dir, "E14_downstream_ett")
    out_dir.mkdir(parents=True, exist_ok=True)
    msg = ("E14 requires real SAITS/PCR imputed series (per-position export), "
           "shared with E13. Linear-proxy path disabled to avoid reporting "
           "numbers that do not use the proposed method. Implement "
           "--save-per-position, then enable E14.")
    print(f"[E14][DISABLED] {msg}")
    (out_dir / "WHY_NO_OUTPUT.txt").write_text(msg)
    return


def _run_E14_downstream_PROXY_DISABLED(args, script_dir: Path) -> None:
    """Original linear-proxy implementation, retained for reference only.
    DO NOT CALL — see run_E14_downstream docstring."""
    out_dir = exp_dir(script_dir, "E14_downstream_ett")
    print(f"[E14] writing to {out_dir}  (R4.12 downstream forecasting)")
    done = load_completed_keys(out_dir)
    seed = int(args.sensitivity_seed)

    key = make_key("E14", "downstream", seed)
    if key in done:
        print("[E14] already checkpointed — skipping")
        return

    # Load ETT frame + standardise
    frame, feats = load_ett_h1_frame(n_rows=args.ett_n_rows)
    X = frame[feats].to_numpy(dtype=float)

    # Standardise per channel using observed statistics
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd

    # Hold out a contiguous test tail for forecasting
    n = len(Xs)
    split = int(n * 0.8)
    train_part = Xs[:split]
    test_part = Xs[split:]

    write_state(out_dir, {"experiment": "E14_downstream_ett",
                          "n_rows": int(n), "n_features": len(feats),
                          "history": 96, "horizon": 24, "seed": seed})

    # Baseline forecasting error on the CLEAN test series (upper bound of
    # achievable quality — no missingness).
    mae_clean = _direct_forecast_mae(test_part)

    # Induce a block gap in the test series (rate ~0.2, block 24), then
    # forecast on (a) linear-filled, (b) a proxy for SAITS vs SAITS+PCR.
    rng = np.random.default_rng(seed)
    corrupted = test_part.copy()
    Tt, F = corrupted.shape
    n_blocks = max(1, int(0.2 * Tt / 24))
    for _ in range(n_blocks):
        f = int(rng.integers(0, F))
        start = int(rng.integers(0, max(1, Tt - 24)))
        corrupted[start:start + 24, f] = np.nan

    # Simple imputations to bound the downstream effect:
    #   linear interpolation  vs  linear + local mean correction
    def _linear_fill(a):
        out = a.copy()
        for f in range(a.shape[1]):
            s = pd.Series(out[:, f])
            out[:, f] = s.interpolate(limit_direction="both").to_numpy()
        return out

    imp_linear = _linear_fill(corrupted)
    mae_linear = _direct_forecast_mae(imp_linear)

    # Correction proxy: nudge filled gaps toward local slope continuity
    imp_corr = imp_linear.copy()
    miss = np.isnan(corrupted)
    # local smoothing on corrected positions
    for f in range(F):
        idx = np.where(miss[:, f])[0]
        for t in idx:
            lo = max(0, t - 2)
            hi = min(len(imp_corr), t + 3)
            imp_corr[t, f] = np.nanmean(imp_linear[lo:hi, f])
    mae_corr = _direct_forecast_mae(imp_corr)

    row = {
        "experiment": "E14", "dataset": "ett_h1", "seed": seed,
        "forecast_mae_clean": mae_clean,
        "forecast_mae_linear_impute": mae_linear,
        "forecast_mae_corrected_impute": mae_corr,
        "downstream_gain_pct": (
            100 * (mae_linear - mae_corr) / mae_linear
            if mae_linear else float("nan")),
        "history": 96, "horizon": 24,
    }
    append_result_row(out_dir, row)
    print(f"[E14] clean={mae_clean:.4f} linear={mae_linear:.4f} "
          f"corrected={mae_corr:.4f} "
          f"gain={row['downstream_gain_pct']:.2f}%")
    print("[E14][NOTE] This is a bounded proxy using linear vs locally-"
          "corrected imputation. For the paper-grade number, wire the "
          "actual SAITS and SAITS+PCR imputed test series into "
          "_direct_forecast_mae (see docstring).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PCR-SAITS Patch 2: new-domain datasets (E9, E10, E14).")
    p.add_argument("--run-E9", action="store_true",
                   help="ETT-h1 full pipeline (R1.10)")
    p.add_argument("--run-E10", action="store_true",
                   help="PhysioNet 2012 full pipeline (R1.10)")
    p.add_argument("--run-E14", action="store_true",
                   help="Downstream forecasting on ETT-h1 (R4.12)")
    p.add_argument("--run-all", action="store_true",
                   help="Run E9, E10, E14")

    p.add_argument("--sensitivity-seed", type=int, default=7)
    p.add_argument("--ett-n-rows", type=int, default=17420,
                   help="Cap ETT-h1 length (default full 17420)")
    p.add_argument("--physionet-max-patients", type=int, default=2000)
    p.add_argument("--physionet-random-state", type=int, default=7,
                   help="Random state passed to BenchPOTS PhysioNet split. \n                        Default 7 for deterministic E10.")
    p.add_argument("--physionet-train-only", action="store_true",
                   help="Use only BenchPOTS train_X for smoke tests. Off by \n                        default; reviewer-facing E10 uses train+val+test.")
    p.add_argument("--allow-env-epochs", action="store_true",
                   help="Honour SAITS_EPOCHS/BRITS_EPOCHS env vars (smoke "
                        "testing). Off by default so a stray env var cannot "
                        "silently shorten a reviewer-facing run.")
    p.add_argument("--allow-ett-windowed-fallback", action="store_true",
                   help="Allow BenchPOTS windowed ETTh1 fallback for smoke tests "
                        "when raw tsdb ETTh1 loading fails. Off by default; "
                        "do not use fallback numbers as final reviewer-facing E9.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    script_dir = THIS_DIR

    if args.run_all:
        args.run_E9 = args.run_E10 = args.run_E14 = True

    if not any([args.run_E9, args.run_E10, args.run_E14]):
        print("Nothing to do. Use --run-all or --run-E9/E10/E14.")
        build_parser().print_help()
        return

    root = revision_root(script_dir)
    (root / "patch2_session.json").write_text(json.dumps({
        "session_started": now_stamp(),
        "run_code_version": RUN_CODE_VERSION,
        "device": choose_device(),
        "script": __file__,
    }, indent=2))

    if args.run_E9:
        run_E9_ett(args, script_dir)
    if args.run_E10:
        run_E10_physionet(args, script_dir)
    if args.run_E14:
        run_E14_downstream(args, script_dir)

    print(f"\n[DONE] Patch 2 finished. Outputs in {root}")


if __name__ == "__main__":
    main()