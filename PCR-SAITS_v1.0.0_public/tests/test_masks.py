import numpy as np
import pytest

from pcrsaits.masks import (
    apply_mask,
    build_correction_scope,
    make_holdout_train_mask,
)


def test_apply_mask_only_masks_requested_cells():
    x = np.arange(6, dtype=float).reshape(3, 2)
    m = np.zeros_like(x, dtype=bool)
    m[1, 0] = True
    out = apply_mask(x, m)
    assert np.isnan(out[1, 0])
    np.testing.assert_array_equal(out[~m], x[~m])


def test_correction_scope_covers_all_missing_not_only_eval_targets():
    x = np.array(
        [[1.0], [np.nan], [np.nan], [4.0], [np.nan], [6.0]]
    )
    eval_mask = np.zeros_like(x, dtype=bool)
    eval_mask[1, 0] = True
    eval_gap = np.ones_like(x, dtype=float)
    eval_gap[1, 0] = 24.0

    scope, gaps = build_correction_scope(x, eval_mask, eval_gap)

    np.testing.assert_array_equal(
        scope[:, 0], [False, True, True, False, True, False]
    )
    assert gaps[1, 0] == 24.0
    assert gaps[2, 0] == 2.0
    assert gaps[4, 0] == 1.0


def test_correction_gap_is_capped():
    x = np.full((60, 1), np.nan)
    _, gaps = build_correction_scope(x, gap_cap=48)
    assert np.all(gaps == 48.0)


def test_eval_target_must_be_missing():
    x = np.ones((3, 1))
    m = np.zeros_like(x, dtype=bool)
    m[0, 0] = True
    with pytest.raises(ValueError, match="must be missing"):
        build_correction_scope(x, m, np.ones_like(x))


def test_holdout_is_deterministic_and_never_selects_natural_nan():
    x = np.arange(240, dtype=float).reshape(120, 2)
    x[10:14, 0] = np.nan

    m1, g1 = make_holdout_train_mask(x, seed=107)
    m2, g2 = make_holdout_train_mask(x, seed=107)

    np.testing.assert_array_equal(m1, m2)
    np.testing.assert_array_equal(g1, g2)
    assert not np.any(m1 & ~np.isfinite(x))
    assert np.all(g1[m1] > 0)
    assert np.all(g1[~m1] == 0)


def test_holdout_gap_labels_are_legacy_values():
    x = np.arange(400, dtype=float).reshape(200, 2)
    m, gaps = make_holdout_train_mask(x, seed=123)
    labels = set(np.unique(gaps[m]).tolist())
    assert labels.issubset({1.0, 6.0, 12.0, 24.0, 48.0})

def test_holdout_exact_v71_golden_fixture():
    """Freeze authoritative v7.1 RNG call order and holdout policy."""
    x = np.arange(48, dtype=float).reshape(24, 2)
    x[3, 0] = np.nan
    x[10, 1] = np.nan

    mask, gaps = make_holdout_train_mask(
        x,
        seed=107,
        holdout_ratio=0.25,
        pointwise_fraction=0.5,
        block_patterns=(3, 6),
        block_buffer=1,
    )

    expected_coords = np.array(
        [
            [2, 0],
            [4, 0],
            [5, 0],
            [6, 0],
            [11, 1],
            [12, 0],
            [12, 1],
            [13, 0],
            [14, 0],
            [14, 1],
            [19, 0],
            [20, 1],
        ],
        dtype=int,
    )
    expected_labels = np.array(
        [1.0, 3.0, 3.0, 3.0, 1.0, 3.0, 1.0, 3.0, 3.0, 1.0, 1.0, 1.0],
        dtype=float,
    )

    np.testing.assert_array_equal(np.argwhere(mask), expected_coords)
    np.testing.assert_array_equal(gaps[mask], expected_labels)

    assert not mask[3, 0]
    assert not mask[10, 1]

