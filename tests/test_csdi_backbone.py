import numpy as np
import pytest
import torch

import pcrsaits.backbones.csdi as csdi_module
from pcrsaits.backbones.csdi import CSDIBackbone


class _FakeCSDI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_args = None
        self.saved = None
        self.loaded = None

    def fit(self, train_set, val_set=None):
        self.fit_args = (train_set, val_set)

    def predict(self, test_set, n_sampling_times=1):
        x = np.asarray(test_set["X"], dtype=float)
        out = np.repeat(
            np.nan_to_num(x, nan=0.0)[:, None, :, :],
            n_sampling_times,
            axis=1,
        )
        missing = ~np.isfinite(x)
        for s in range(n_sampling_times):
            noise = torch.randn(int(missing.sum())).cpu().numpy()
            sample = out[:, s]
            sample[missing] = noise + float(s)
        return {"imputation": out}

    def save(self, path):
        self.saved = str(path)

    def load(self, path):
        self.loaded = str(path)


@pytest.fixture(autouse=True)
def _fake_csdi(monkeypatch):
    monkeypatch.setattr(csdi_module, "CSDI", _FakeCSDI)


def _make_adapter(**overrides):
    kwargs = dict(
        n_steps=4,
        n_features=2,
        epochs=1,
        batch_size=2,
        patience=1,
        n_layers=1,
        n_heads=1,
        n_channels=8,
        d_time_embedding=8,
        d_feature_embedding=4,
        d_diffusion_embedding=8,
        n_diffusion_steps=5,
        n_sampling_times=3,
        aggregation="median",
        sampling_seed=7,
        verbose=False,
    )
    kwargs.update(overrides)
    return CSDIBackbone(**kwargs)


def _masked_windows():
    x = np.arange(16, dtype=float).reshape(2, 4, 2)
    x[0, 1, 0] = np.nan
    x[1, 3, 1] = np.nan
    return x


def test_sample_shape_and_observed_preservation():
    adapter = _make_adapter()
    x = _masked_windows()
    samples = adapter.sample(x)
    assert samples.shape == (2, 3, 4, 2)
    observed = np.isfinite(x)
    for s in range(3):
        assert np.array_equal(samples[:, s][observed], x[observed])


def test_repeated_seeded_sampling_is_deterministic():
    adapter = _make_adapter(sampling_seed=123)
    x = _masked_windows()
    a = adapter.sample(x)
    b = adapter.sample(x)
    assert np.array_equal(a, b)


def test_seeded_sampling_restores_external_cpu_rng_state():
    adapter = _make_adapter(sampling_seed=123)
    x = _masked_windows()
    torch.manual_seed(2026)
    before = torch.get_rng_state().clone()
    adapter.sample(x)
    after = torch.get_rng_state().clone()
    assert torch.equal(before, after)


@pytest.mark.skipif(
    (not torch.cuda.is_available()) or torch.cuda.device_count() < 2,
    reason="multi-GPU RNG restoration requires at least two CUDA devices",
)
def test_seeded_sampling_restores_all_cuda_rng_states():
    adapter = _make_adapter(sampling_seed=123)
    x = _masked_windows()
    torch.cuda.manual_seed_all(2026)
    before = [
        torch.cuda.get_rng_state(i).clone()
        for i in range(torch.cuda.device_count())
    ]
    adapter.sample(x)
    after = [
        torch.cuda.get_rng_state(i).clone()
        for i in range(torch.cuda.device_count())
    ]
    assert all(torch.equal(a, b) for a, b in zip(before, after))


def test_sampling_seed_none_is_stochastic():
    adapter = _make_adapter(sampling_seed=None)
    x = _masked_windows()
    a = adapter.sample(x)
    b = adapter.sample(x)
    missing = ~np.isfinite(x)
    assert not np.array_equal(a, b)


def test_n_sampling_times_override_and_single_sample_axis():
    adapter = _make_adapter(n_sampling_times=3)
    x = _masked_windows()
    assert adapter.sample(x, n_sampling_times=5).shape == (2, 5, 4, 2)
    assert adapter.sample(x, n_sampling_times=1).shape == (2, 1, 4, 2)


def test_impute_is_sample_median_and_preserves_observed():
    adapter = _make_adapter()
    x = _masked_windows()
    samples = adapter.sample(x)
    out = adapter.impute(x)
    expected = np.median(samples, axis=1)
    observed = np.isfinite(x)
    expected[observed] = x[observed]
    assert np.array_equal(out, expected)
    assert np.array_equal(out[observed], x[observed])


def test_mean_aggregation_is_supported():
    adapter = _make_adapter(aggregation="mean", n_sampling_times=2)
    x = _masked_windows()
    samples = adapter.sample(x)
    out = adapter.impute(x)
    expected = np.mean(samples, axis=1)
    observed = np.isfinite(x)
    expected[observed] = x[observed]
    assert np.array_equal(out, expected)


def test_fit_passes_pypots_train_and_validation_contract():
    adapter = _make_adapter()
    train = np.ones((3, 4, 2), dtype=float)
    val_ori = np.ones((2, 4, 2), dtype=float)
    val = val_ori.copy()
    val[0, 1, 0] = np.nan
    adapter.fit(train, val, val_ori)
    train_set, val_set = adapter.model.fit_args
    assert train_set["X"].shape == (3, 4, 2)
    assert val_set["X"].shape == (2, 4, 2)
    assert val_set["X_ori"].shape == (2, 4, 2)


@pytest.mark.parametrize(
    "val,val_ori",
    [
        (np.ones((2, 4, 2)), None),
        (None, np.ones((2, 4, 2))),
    ],
)
def test_one_sided_validation_arguments_are_rejected(val, val_ori):
    adapter = _make_adapter()
    train = np.ones((3, 4, 2), dtype=float)
    with pytest.raises(ValueError, match="requires both"):
        adapter.fit(train, val, val_ori)


@pytest.mark.parametrize(
    "name,value",
    [
        ("n_steps", 0),
        ("n_features", 0),
        ("epochs", 0),
        ("batch_size", 0),
        ("n_layers", 0),
        ("n_heads", 0),
        ("n_channels", 0),
        ("d_time_embedding", 0),
        ("d_feature_embedding", 0),
        ("d_diffusion_embedding", 0),
        ("n_diffusion_steps", 0),
        ("n_sampling_times", 0),
        ("patience", 0),
    ],
)
def test_nonpositive_integer_parameters_rejected(name, value):
    with pytest.raises(ValueError):
        _make_adapter(**{name: value})


@pytest.mark.parametrize(
    "overrides",
    [
        {"aggregation": "bad"},
        {"target_strategy": "bad"},
        {"schedule": "bad"},
        {"beta_start": 0.0},
        {"beta_start": 0.6, "beta_end": 0.5},
        {"beta_end": 1.0},
        {"sampling_seed": 1.5},
        {"is_unconditional": "no"},
        {"verbose": 1},
        {"n_heads": 3, "n_channels": 8},
    ],
)
def test_invalid_constructor_values_rejected(overrides):
    with pytest.raises(ValueError):
        _make_adapter(**overrides)


def test_window_shape_contract_is_enforced():
    adapter = _make_adapter()
    with pytest.raises(ValueError):
        adapter.impute(np.ones((4, 2)))
    with pytest.raises(ValueError):
        adapter.impute(np.ones((1, 5, 2)))
    with pytest.raises(ValueError):
        adapter.impute(np.ones((1, 4, 3)))


def test_save_and_load_delegate_to_pypots(tmp_path):
    adapter = _make_adapter()
    path = tmp_path / "csdi.pypots"
    adapter.save(path)
    assert adapter.model.saved == str(path)

    restored = CSDIBackbone.load_from_checkpoint(
        path,
        n_steps=4,
        n_features=2,
        epochs=1,
        batch_size=2,
        patience=1,
        n_layers=1,
        n_heads=1,
        n_channels=8,
        d_time_embedding=8,
        d_feature_embedding=4,
        d_diffusion_embedding=8,
        n_diffusion_steps=5,
        n_sampling_times=3,
        aggregation="median",
        sampling_seed=7,
        verbose=False,
    )
    assert restored.model.loaded == str(path)
