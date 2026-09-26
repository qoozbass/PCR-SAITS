#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bridge script: generate ETTh1.csv / physionet2012.csv for the E13/E14
per-position exporter (pcr_saits_v7_1b_per_position.py) using the EXACT SAME
tsdb/benchpots loading functions and parameters that produced the verified
E9 (+6.41%) and E10 (+8.42%) results in
1783952488458_pcr_saits_new_datasets.py.

WHY THIS SCRIPT EXISTS
----------------------
pcr_saits_v7_1b_per_position.py's _load_ett_dataset() / _load_physionet_dataset()
read LOCAL CSV files. But E9/E10's real, already-verified numbers were produced
by loading ETT-h1 and PhysioNet-2012 via `tsdb`/`benchpots` (auto-download),
NOT from local CSV files. Those two pipelines had never been connected before
this script — running the exporter directly against hand-made or missing local
CSVs risked either a crash, or worse, silently using a DIFFERENT data instance
than E9/E10, breaking cross-experiment consistency in the paper.

This script is copy-pasted logic (not just re-imported) from the uploaded
patch file, deliberately WITHOUT importing that file directly — the uploaded
file monkey-patches `pcr_saits_v1_5_main_extensions.py` at import time
(module-level code, lines ~94-100), which would require that file to also be
present. The functions below have no dependency on that monkey-patch chain.

Parameters below are hard-set to match E9/E10's actual reported numbers:
  - ETT:       n_rows=17420, allow_windowed_fallback=False
  - PhysioNet: max_patients=2000, random_state=7, use_all_splits=True
Do NOT change these unless you intend to regenerate E9/E10 with different
data (in which case E9/E10's existing reported numbers would need rerunning too).

Usage
-----
python make_ett_physionet_csv_bridge.py --out-dir D:\\GAP-SAITS

Requires: tsdb, benchpots, pypots, pygrinder, scikit-learn, pandas, numpy
  pip install pypots benchpots tsdb pygrinder
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _safe_feature_names(raw, n: int, prefix: str) -> List[str]:
    """Coerce a possibly-ndarray / pandas.Index / None feature-name object
    into a plain Python list of length n."""
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
    try:
        import benchpots.datasets as bd  # noqa
        return bd
    except Exception as exc:
        raise RuntimeError(
            "benchpots is required for ETT/PhysioNet loading. "
            "Install with:  pip install pypots benchpots tsdb pygrinder\n"
            f"Underlying import error: {exc}"
        )


def load_ett_h1_frame(n_rows: Optional[int] = None,
                       allow_windowed_fallback: bool = False
                       ) -> Tuple[pd.DataFrame, List[str]]:
    """Identical logic to load_ett_h1_frame() in
    1783952488458_pcr_saits_new_datasets.py (the function that produced E9's
    verified +6.41% result). Loads the raw long ETTh1 series via tsdb."""
    try:
        import tsdb
        data = tsdb.load("electricity_transformer_temperature")
        df = None
        for key in ("ETTh1", "etth1", "ETT_h1"):
            if isinstance(data, dict) and key in data:
                df = data[key]
                break
        if df is None and isinstance(data, dict):
            dfs = data.get("dataframes") or {}
            for key in ("ETTh1", "etth1"):
                if key in dfs:
                    df = dfs[key]
                    break
        if df is None:
            raise KeyError("ETTh1 not found in tsdb payload")
        # Bug fix: only reset_index() when there is NOT already an explicit
        # date/timestamp column. Unconditional reset_index() inserts a new
        # "index" column at position 0; if df also already has a "date"
        # column (redundant with a DatetimeIndex), the later time-column
        # detection loop would iterate columns in order and match "index"
        # before "date" — silently using the row-number as a fake timestamp
        # (parses to 1970-epoch nanoseconds) instead of ETTh1's real hourly
        # timestamps. This looks like it worked (right row/feature count)
        # while being wrong, which is worse than an outright crash.
        _lower_cols = {str(c).lower() for c in df.columns}
        if _lower_cols & {"date", "timestamp", "datetime"}:
            df = df.copy()
        else:
            df = df.reset_index()
    except Exception as exc:
        if not allow_windowed_fallback:
            raise RuntimeError(
                "Could not load raw ETTh1 via tsdb. Refusing to use the "
                "BenchPOTS windowed fallback because it can reshape sliding "
                "windows into a non-raw time series (would NOT match E9's "
                "verified numbers). Install/fix tsdb raw ETTh1 loading. "
                f"Underlying error: {exc}"
            )
        print("[BRIDGE][WARN] using BenchPOTS windowed fallback. This will "
              "NOT match E9's original verified data instance.")
        bd = _import_benchpots()
        prep = bd.preprocess_ett(subset="ETTh1", rate=0.0, pattern="point", n_steps=48)
        X = prep["train_X"]
        if hasattr(X, "numpy"):
            X = X.numpy()
        X = np.asarray(X, dtype=float)
        feats = _safe_feature_names(prep.get("feature_names"), X.shape[-1], "ETT")
        flat = X.reshape(-1, X.shape[-1])
        df = pd.DataFrame(flat, columns=feats)
        df.insert(0, "timestamp", pd.date_range("2016-07-01", periods=len(df), freq="h"))

    # Bug fix: prefer an explicit real time column over a reset_index-
    # generated "index" column. Only fall back to "index" if it actually
    # looks like a datetime (>95% parse successfully and it isn't already
    # a plain numeric RangeIndex), rather than matching it unconditionally.
    time_col = None
    for key in ("timestamp", "date", "datetime"):
        for c in df.columns:
            if str(c).lower() == key:
                time_col = c
                break
        if time_col is not None:
            break
    if time_col is None:
        for c in df.columns:
            if str(c).lower() == "index":
                _parsed = pd.to_datetime(df[c], errors="coerce")
                if (not pd.api.types.is_numeric_dtype(df[c])) and _parsed.notna().mean() > 0.95:
                    time_col = c
                break

    if time_col is None:
        df.insert(0, "timestamp", pd.date_range("2016-07-01", periods=len(df), freq="h"))
    else:
        df = df.rename(columns={time_col: "timestamp"})
    # Drop any leftover non-feature time/index columns (e.g. a spurious
    # numeric "index" column that wasn't chosen as the timestamp).
    _drop_cols = [c for c in df.columns
                  if c != "timestamp" and str(c).lower() in {"index", "date", "datetime"}]
    if _drop_cols:
        df = df.drop(columns=_drop_cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Guard 1: validate the parse actually succeeded, rather than trusting
    # errors="coerce" silently. If timestamp detection breaks again in the
    # future (e.g. tsdb changes its payload format), this should crash loudly
    # instead of writing a CSV with mostly-garbage timestamps.
    valid_ts_rate = float(df["timestamp"].notna().mean())
    if valid_ts_rate < 0.95:
        raise RuntimeError(
            f"ETTh1 timestamp parse failed: only {valid_ts_rate:.1%} valid timestamps. "
            "Check tsdb payload/time-column detection before writing ETTh1.csv."
        )
    bad_ts = int(df["timestamp"].isna().sum())
    if bad_ts:
        print(f"[BRIDGE][ETT][WARN] dropping {bad_ts:,} rows with invalid timestamp")
        df = df.dropna(subset=["timestamp"]).copy()
    dup_ts = int(df["timestamp"].duplicated().sum())
    if dup_ts:
        print(f"[BRIDGE][ETT][WARN] dropping {dup_ts:,} duplicate timestamps")
        df = df.drop_duplicates(subset=["timestamp"], keep="first").copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Bug fix (robustness): force numeric rather than silently dropping a
    # feature just because tsdb happened to return it as a string dtype.
    candidate_feats = [c for c in df.columns if c != "timestamp"]
    feats = []
    for c in candidate_feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().all():
            print(f"[BRIDGE][ETT][WARN] dropping all-NaN/non-numeric feature: {c}")
            continue
        feats.append(c)
    df = df[["timestamp"] + feats].copy()

    # Guard 2: sanity-check against ETTh1's known feature set. Catches a
    # tsdb payload-format change silently producing an incomplete/wrong
    # feature set instead of the expected 7 channels.
    expected_ett = {"HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"}
    missing_expected = expected_ett - set(feats)
    if missing_expected:
        print(f"[BRIDGE][ETT][WARN] missing expected ETTh1 features: "
              f"{sorted(missing_expected)}; got {feats}")
    if len(feats) < 7:
        raise RuntimeError(
            f"ETTh1 has only {len(feats)} numeric features ({feats}). "
            "Expected around 7 features; check tsdb payload before writing CSV."
        )

    # Guard 3: this bridge exists specifically to reproduce E9's exact data
    # instance (n_rows=17420). Truncating when there's MORE data than needed
    # is fine, but getting FEWER rows than expected means the tsdb payload
    # is incomplete/different from what produced E9's verified numbers —
    # that must not be allowed to silently write a shorter ETTh1.csv.
    if n_rows is not None:
        if len(df) > n_rows:
            df = df.iloc[:n_rows].reset_index(drop=True)
        elif len(df) < n_rows:
            raise RuntimeError(
                f"ETTh1 expected {n_rows:,} rows to match E9, got {len(df):,}. "
                "Check tsdb raw ETTh1 payload before writing ETTh1.csv."
            )

    print(f"[BRIDGE][ETT] loaded: {df.shape[0]} rows x {len(feats)} feats ({feats})")
    return df, feats


def _manual_preprocess_physionet2012_seta(random_state: int = 7) -> Dict[str, Any]:
    """Identical logic to _manual_preprocess_physionet2012_seta() in the
    patch file — fallback for the known BenchPOTS RecordID KeyError."""
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
        raise KeyError(f"PhysioNet set-a must contain RecordID and Time columns; got {list(df.columns)}")

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
    X = X.drop(columns=["RecordID"], errors="ignore").reset_index()
    X = X.drop(columns=["level_1"], errors="ignore")
    if "RecordID" not in X.columns:
        raise KeyError("Manual fallback could not recover RecordID after groupby/reset_index")
    X = X.drop(columns=["ICUType"], errors="ignore")

    feature_names = [c for c in X.columns if c not in ("RecordID", "Time")]
    feature_names = [c for c in feature_names if pd.api.types.is_numeric_dtype(X[c])]
    if not feature_names:
        raise RuntimeError("No numeric PhysioNet features found after preprocessing")

    all_ids = np.array(sorted(pd.Series(X["RecordID"].unique()).dropna().tolist()))

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
            raise RuntimeError(f"Expected {expected} rows for {len(ids)} stays, got {arr.shape[0]}")
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
    """Identical logic to load_physionet2012_frame() in the patch file (the
    function that produced E10's verified +8.42% result)."""
    bd = _import_benchpots()
    try:
        prep = bd.preprocess_physionet2012(subset="set-a", rate=0.0,
                                            pattern="point", random_state=int(random_state))
        prep_source = "benchpots_preprocess_physionet2012"
    except KeyError as exc:
        if "RecordID" not in str(exc):
            raise
        print(f"[BRIDGE][WARN] BenchPOTS preprocess_physionet2012 failed with "
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
        split_desc = f"train+val+test ({train_X.shape[0]}+{val_X.shape[0]}+{test_X.shape[0]})"
    else:
        X = train_X
        split_desc = f"train_only ({train_X.shape[0]})"

    n_pat = min(int(max_patients), X.shape[0])
    X = X[:n_pat]
    n, steps, feats_n = X.shape
    feat_names = _safe_feature_names(prep.get("feature_names"), feats_n, "vital")

    with np.errstate(all="ignore"):
        global_median = np.nanmedian(X.reshape(n * steps, feats_n), axis=0)
    global_median = np.where(np.isfinite(global_median), global_median, 0.0)

    cleaned = np.empty_like(X)
    for p in range(n):
        stay = pd.DataFrame(X[p], columns=feat_names)
        stay = stay.interpolate(method="linear", limit_direction="both", axis=0)
        for j, c in enumerate(feat_names):
            if stay[c].isna().any():
                stay[c] = stay[c].fillna(global_median[j])
        cleaned[p] = stay.to_numpy()
    cleaned = np.nan_to_num(cleaned, nan=0.0)

    natural_missing = float(np.isnan(X).mean())
    flat = cleaned.reshape(n * steps, feats_n)
    df = pd.DataFrame(flat, columns=feat_names)
    df.insert(0, "timestamp", pd.date_range("2012-01-01", periods=len(df), freq="h"))

    print(f"[BRIDGE][PhysioNet] loaded via {prep_source}: {split_desc}; "
          f"using {n} patients -> {df.shape[0]} rows x {feats_n} feats "
          f"| random_state={int(random_state)} "
          f"| natural missing cleaned: {natural_missing:.1%}")
    return df, feat_names


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=str, required=True,
                    help="Directory to write ETTh1.csv / physionet2012.csv into (e.g. D:\\GAP-SAITS)")
    ap.add_argument("--skip-ett", action="store_true")
    ap.add_argument("--skip-physionet", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_ett:
        print("\n" + "=" * 70)
        print("Generating ETTh1.csv (matching E9's verified parameters)")
        print("=" * 70)
        ett_df, ett_feats = load_ett_h1_frame(n_rows=17420, allow_windowed_fallback=False)
        ett_path = out_dir / "ETTh1.csv"
        if ett_path.exists():
            print(f"[BRIDGE][WARN] overwriting existing file: {ett_path}")
        ett_df.to_csv(ett_path, index=False)
        print(f"[BRIDGE] wrote {ett_path} ({len(ett_df):,} rows, {len(ett_feats)} features)")

    if not args.skip_physionet:
        print("\n" + "=" * 70)
        print("Generating physionet2012.csv (matching E10's verified parameters)")
        print("=" * 70)
        phys_df, phys_feats = load_physionet2012_frame(
            max_patients=2000, random_state=7, use_all_splits=True)
        phys_path = out_dir / "physionet2012.csv"

        # Guard: this bridge exists to reproduce E10's exact data instance
        # (2000 patients x 48 steps = 96,000 rows). Checked BEFORE to_csv()
        # so a mismatched file is never written to disk in the first place.
        if len(phys_df) != 96000:
            raise RuntimeError(
                f"PhysioNet2012 expected 96,000 rows (2000 patients x 48 steps) "
                f"to match E10, got {len(phys_df):,}. Check max_patients / "
                "tsdb / benchpots data availability before writing physionet2012.csv."
            )

        if phys_path.exists():
            print(f"[BRIDGE][WARN] overwriting existing file: {phys_path}")
        phys_df.to_csv(phys_path, index=False)
        print(f"[BRIDGE] wrote {phys_path} ({len(phys_df):,} rows, {len(phys_feats)} features)")

    print("\n[BRIDGE][DONE] CSV files ready for pcr_saits_v7_1b_per_position.py's "
          "--datasets etth1,physionet2012")


if __name__ == "__main__":
    main()
