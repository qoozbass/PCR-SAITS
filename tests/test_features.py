import numpy as np

from pcrsaits.features import (
    _precompute_local_estimates,
    build_correction_rows,
    build_example_arrays,
    group_id,
    infer_feature_group,
    seasonal_estimate,
)


def test_local_interpolation_matches_legacy_anchor_rule():
    x = np.array([[0.0], [np.nan], [np.nan], [3.0]])
    vals, ok = _precompute_local_estimates(x)
    np.testing.assert_allclose(vals[1:3, 0], [1.0, 2.0])
    np.testing.assert_allclose(ok[1:3, 0], [1.0, 1.0])


def test_local_edge_uses_single_available_anchor():
    x = np.array([[np.nan], [np.nan], [2.0]])
    vals, ok = _precompute_local_estimates(x)
    np.testing.assert_allclose(vals[:2, 0], [2.0, 2.0])
    np.testing.assert_allclose(ok[:2, 0], [1.0, 1.0])


def test_seasonal_uses_both_sides_and_multiple_lags():
    x = np.full((60, 1), np.nan)
    x[0, 0] = 2.0
    x[48, 0] = 6.0
    val, ok = seasonal_estimate(x, 24, 0)
    assert ok == 1.0
    assert val == 4.0


def test_legacy_feature_groups_include_uci_and_beijing():
    assert infer_feature_group("CO(GT)") == "pollutant"
    assert infer_feature_group("PT08.S1(CO)") == "sensor"
    assert infer_feature_group("T") == "meteorological"
    assert infer_feature_group("PM2.5") == "pollutant"
    assert infer_feature_group("TEMP") == "meteorological"
    assert group_id("CO(GT)") == 0
    assert group_id("PT08.S1(CO)") == 1
    assert group_id("T") == 2


def test_no_seasonal_training_row_exact_order_and_dtype():
    original = np.array([[0.0], [1.0], [2.0]])
    masked = np.array([[0.0], [np.nan], [2.0]])
    base = np.array([[0.0], [1.5], [2.0]])
    target = np.array([[False], [True], [False]])
    gap = np.zeros_like(original)
    gap[1, 0] = 6.0

    X, y, base_err, easy = build_example_arrays(
        original_values=original,
        masked_values=masked,
        base_imputed_values=base,
        target_mask=target,
        gap_len=gap,
        feature_names=["CO(GT)"],
        use_local_branch=True,
        use_seasonal_branch=False,
        use_domain_tags=True,
    )

    expected = np.array(
        [[1.5, 1.0, 0.0, 0.5, 1.5, 1.0, 0.0, 0.125, 1.0, 0.0]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(X, expected)
    np.testing.assert_allclose(y, [-0.5])
    np.testing.assert_allclose(base_err, [0.5])
    np.testing.assert_allclose(easy, [0.0])
    assert X.dtype == np.float32
    assert y.dtype == np.float32


def test_pointwise_gap_is_easy_example():
    original = np.array([[0.0], [1.0], [2.0]])
    masked = np.array([[0.0], [np.nan], [2.0]])
    base = np.array([[0.0], [1.0], [2.0]])
    target = np.array([[False], [True], [False]])
    gap = np.zeros_like(original)
    gap[1, 0] = 1.0

    _, _, _, easy = build_example_arrays(
        original_values=original,
        masked_values=masked,
        base_imputed_values=base,
        target_mask=target,
        gap_len=gap,
        feature_names=["T"],
        use_local_branch=True,
        use_seasonal_branch=False,
        use_domain_tags=True,
    )
    np.testing.assert_array_equal(easy, [1.0])


def test_correction_row_matches_training_feature_semantics():
    masked = np.array([[0.0], [np.nan], [2.0]])
    base = np.array([[0.0], [1.5], [2.0]])
    scope = np.array([[False], [True], [False]])
    gap = np.zeros_like(base)
    gap[1, 0] = 6.0

    rows, coords = build_correction_rows(
        original_masked_values=masked,
        base_imputed_values=base,
        correction_mask=scope,
        gap_len=gap,
        feature_names=["CO(GT)"],
        use_local_branch=True,
        use_seasonal_branch=False,
        use_domain_tags=True,
    )

    assert coords == [(1, 0)]
    expected = np.array(
        [1.5, 1.0, 0.0, 0.5, 1.5, 1.0, 0.0, 0.125, 1.0, 0.0],
        dtype=np.float32,
    )
    np.testing.assert_allclose(rows[0], expected)
    assert rows[0].dtype == np.float32
