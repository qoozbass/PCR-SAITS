import numpy as np

from pcrsaits.windows import build_windows, reconstruct_from_windows


def test_build_windows_appends_terminal_window():
    x = np.arange(11, dtype=float)[:, None]
    windows, starts = build_windows(x, window=4, stride=3)
    np.testing.assert_array_equal(starts, [0, 3, 6, 7])
    assert windows.shape == (4, 4, 1)
    np.testing.assert_array_equal(windows[-1, :, 0], [7, 8, 9, 10])


def test_build_windows_pads_short_series_with_nan():
    x = np.array([[1.0], [2.0]])
    windows, starts = build_windows(x, window=4, stride=2)
    np.testing.assert_array_equal(starts, [0])
    assert windows.shape == (1, 4, 1)
    np.testing.assert_allclose(windows[0, :2, 0], [1.0, 2.0])
    assert np.isnan(windows[0, 2:, 0]).all()


def test_reconstruction_overlap_average_and_nan_ignore():
    windows = np.array(
        [
            [[0.0], [1.0], [2.0], [3.0]],
            [[2.0], [np.nan], [4.0], [5.0]],
        ]
    )
    starts = np.array([0, 2])
    out = reconstruct_from_windows(windows, starts, total_length=6)
    np.testing.assert_allclose(out[:, 0], [0, 1, 2, 3, 4, 5])


def test_window_roundtrip_for_consistent_windows():
    x = np.arange(17, dtype=float).reshape(-1, 1)
    windows, starts = build_windows(x, 6, 4)
    out = reconstruct_from_windows(windows, starts, len(x))
    np.testing.assert_allclose(out, x)
