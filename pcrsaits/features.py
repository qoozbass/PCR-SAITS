from __future__ import annotations

import warnings
import numpy as np

# Legacy v7.1 paper-suite feature grouping.
POLLUTANT_VARS = {
    "CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"
}
METEO_VARS = {"T", "RH", "AH"}
BEIJING_POLLUTANT_VARS = {"PM2.5", "PM10", "SO2", "NO2", "CO", "O3"}
BEIJING_METEO_VARS = {"TEMP", "PRES", "DEWP", "RAIN", "WSPM"}

_UNKNOWN_GROUP_WARNED = set()


def infer_feature_group(feature_name: str) -> str:
    """Legacy paper-suite inference, including unknown->sensor fallback."""
    if feature_name in METEO_VARS or feature_name in BEIJING_METEO_VARS:
        return "meteorological"
    if feature_name.startswith("PT08."):
        return "sensor"
    if feature_name in POLLUTANT_VARS or feature_name in BEIJING_POLLUTANT_VARS:
        return "pollutant"

    if feature_name not in _UNKNOWN_GROUP_WARNED:
        # Legacy used print(). Keep the observable warning intent but do not
        # alter the returned group.
        print(
            f"[WARN] Unknown feature group for '{feature_name}'. "
            "Falling back to 'sensor'."
        )
        _UNKNOWN_GROUP_WARNED.add(feature_name)
    return "sensor"


def group_id(feature_name: str) -> int:
    g = infer_feature_group(feature_name)
    return 0 if g == "pollutant" else 1 if g == "sensor" else 2


def _precompute_local_estimates(full_series: np.ndarray):
    """Exact legacy v7.1 nearest-anchor interpolation/reference logic."""
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
                weight = (t - li) / float(ri - li)
                local_vals[t, f] = float(
                    (1.0 - weight) * col[li] + weight * col[ri]
                )
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


def seasonal_estimate(
    full_series: np.ndarray,
    t: int,
    f: int,
    lags=(24, 48, 168),
):
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


def build_example_arrays(
    *,
    original_values: np.ndarray,
    masked_values: np.ndarray,
    base_imputed_values: np.ndarray,
    target_mask: np.ndarray,
    gap_len: np.ndarray,
    feature_names,
    use_local_branch: bool,
    use_seasonal_branch: bool,
    use_domain_tags: bool,
):
    """Training-side extraction of legacy `_build_examples`.

    The implementation intentionally mirrors the original row construction
    rather than sharing code with deployment correction before Phase 4.
    """
    input_dim = 10 if use_domain_tags else 8
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
            seasonal_val, seasonal_ok = seasonal_estimate(
                masked_values, t, f
            )

            if not use_local_branch:
                local_val, local_ok = 0.0, 0.0
            if not use_seasonal_branch:
                seasonal_val, seasonal_ok = 0.0, 0.0

            gl = float(gap_len[t, f])
            gln = min(gl / 48.0, 1.0)
            feat_group = group_id(feature_names[f]) if use_domain_tags else -1

            row = [
                base,
                local_val,
                seasonal_val,
                base - local_val,
                base - seasonal_val,
                local_ok,
                seasonal_ok,
                gln,
            ]
            if use_domain_tags:
                row.extend(
                    [
                        float(feat_group == 0),
                        float(feat_group == 1),
                    ]
                )

            feats.append(np.array(row, dtype=np.float32))
            tgts.append(np.float32(true - base))
            base_errs.append(np.float32(abs(true - base)))
            easy_mask.append(np.float32(1.0 if gl <= 1.0 else 0.0))

    if not feats:
        return (
            np.zeros((0, input_dim), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    return (
        np.stack(feats),
        np.asarray(tgts, dtype=np.float32),
        np.asarray(base_errs, dtype=np.float32),
        np.asarray(easy_mask, dtype=np.float32),
    )


def build_correction_rows(
    *,
    original_masked_values: np.ndarray,
    base_imputed_values: np.ndarray,
    correction_mask: np.ndarray,
    gap_len: np.ndarray,
    feature_names,
    use_local_branch: bool,
    use_seasonal_branch: bool,
    use_domain_tags: bool,
):
    """Deployment-side extraction of the row construction in `correct`.

    Deliberately not deduplicated with `build_example_arrays` before Phase 4.
    """
    rows, coords = [], []
    n_t, n_f = base_imputed_values.shape
    local_vals_all, local_ok_all = _precompute_local_estimates(
        original_masked_values
    )

    for t in range(n_t):
        for f in range(n_f):
            if not correction_mask[t, f]:
                continue

            base = float(base_imputed_values[t, f])
            local_val = float(local_vals_all[t, f])
            local_ok = float(local_ok_all[t, f])
            seasonal_val, seasonal_ok = seasonal_estimate(
                original_masked_values, t, f
            )

            if not use_local_branch:
                local_val, local_ok = 0.0, 0.0
            if not use_seasonal_branch:
                seasonal_val, seasonal_ok = 0.0, 0.0

            gl = float(gap_len[t, f])
            gln = min(gl / 48.0, 1.0)
            feat_group = group_id(feature_names[f]) if use_domain_tags else -1

            row = [
                base,
                local_val,
                seasonal_val,
                base - local_val,
                base - seasonal_val,
                local_ok,
                seasonal_ok,
                gln,
            ]
            if use_domain_tags:
                row.extend(
                    [
                        float(feat_group == 0),
                        float(feat_group == 1),
                    ]
                )

            rows.append(np.array(row, dtype=np.float32))
            coords.append((t, f))

    return rows, coords
