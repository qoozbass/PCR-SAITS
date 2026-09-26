import math
from pathlib import Path

import numpy as np
import torch

import pcrsaits.corrector as corr_mod
from pcrsaits.corrector import PCRCorrector
from pcrsaits.masks import build_correction_scope


class FillZeroBackbone:
    def __init__(self):
        self.calls = []
    def impute(self, windows):
        self.calls.append(np.array(windows, copy=True))
        return np.nan_to_num(windows, nan=0.0)


def make(monkeypatch, **overrides):
    monkeypatch.setattr(corr_mod, "choose_device", lambda: "cpu")
    kwargs = dict(
        variant="pcrsaitsv14_no_seasonal_branch",
        n_steps=4,
        learning_rate=1e-3,
        weight_decay=1e-4,
        epochs=1,
        batch_size=4,
        patience=1,
        preserve_loss_weight=1.0,
        rel_loss_weight=1.0,
        sparse_loss_weight=1.0,
        base_model=FillZeroBackbone(),
        feature_names=["CO(GT)"],
        base_impute_stride=2,
        verbose=False,
    )
    kwargs.update(overrides)
    return PCRCorrector(**kwargs)


def test_proposed_variant_flags(monkeypatch):
    obj = make(monkeypatch)
    assert obj.use_seasonal_branch is False
    assert obj.use_local_branch is True
    assert obj.use_domain_tags is True
    assert obj.direct_residual is True
    assert obj.input_dim == 10
    assert obj.base_impute_stride == 2


def test_legacy_stride_fallback_is_n_steps(monkeypatch):
    obj = make(monkeypatch, base_impute_stride=None)
    assert obj.base_impute_stride == obj.n_steps == 4


def test_base_imputation_uses_configured_stride(monkeypatch):
    obj = make(monkeypatch)
    x = np.arange(7, dtype=float)[:, None]
    x[3, 0] = np.nan
    out = obj._impute_full_series_with_base(x)
    # window=4, stride=2, length=7 -> starts [0,2,3], hence 3 calls in one batch
    assert obj.base_model.calls[-1].shape == (3, 4, 1)
    assert out.shape == x.shape


def test_correct_adds_residual_only_to_scope_and_restores_observed(monkeypatch):
    obj = make(
        monkeypatch,
        n_steps=3,
        base_impute_stride=3,
        feature_names=["CO(GT)"],
    )
    with torch.no_grad():
        obj.model.delta_head.weight.zero_()
        # 4*tanh(bias)=1
        obj.model.delta_head.bias.fill_(math.atanh(0.25))

    masked = np.array([[1.0], [np.nan], [3.0]])
    base = np.array([[100.0], [10.0], [300.0]])
    scope, gaps = build_correction_scope(masked)

    out = obj.correct(masked, base, scope, gaps)

    # observed positions must be exact original input values, not base predictions
    assert out[0, 0] == 1.0
    assert out[2, 0] == 3.0
    # only missing target receives +1 bounded residual
    np.testing.assert_allclose(out[1, 0], 11.0, rtol=1e-6, atol=1e-6)


def test_fit_smoke_with_explicit_masks(monkeypatch):
    torch.manual_seed(7)
    obj = make(monkeypatch)

    train = np.arange(8, dtype=float)[:, None] / 10.0
    val = np.arange(8, dtype=float)[:, None] / 20.0

    train_mask = np.zeros_like(train, dtype=bool)
    val_mask = np.zeros_like(val, dtype=bool)
    train_mask[[1, 4, 6], 0] = True
    val_mask[[2, 5], 0] = True

    train_gap = np.zeros_like(train)
    val_gap = np.zeros_like(val)
    train_gap[train_mask] = 1.0
    val_gap[val_mask] = 1.0

    obj.fit(
        train,
        train_mask,
        train_gap,
        val,
        val_mask,
        val_gap,
    )

    # A valid best state must have been loaded; with non-zero residual targets,
    # the zero-initialized output head should receive an update.
    changed = (
        torch.count_nonzero(obj.model.delta_head.weight).item() > 0
        or torch.count_nonzero(obj.model.delta_head.bias).item() > 0
    )
    assert changed


def test_checkpoint_roundtrip(monkeypatch, tmp_path):
    obj = make(monkeypatch)
    with torch.no_grad():
        obj.model.delta_head.bias.fill_(0.123)

    path = tmp_path / "pcr.pt"
    obj.save(path)

    kwargs = dict(
        variant="pcrsaitsv14_no_seasonal_branch",
        n_steps=4,
        learning_rate=1e-3,
        weight_decay=1e-4,
        epochs=1,
        batch_size=4,
        patience=1,
        preserve_loss_weight=1.0,
        rel_loss_weight=1.0,
        sparse_loss_weight=1.0,
        base_model=FillZeroBackbone(),
        feature_names=["CO(GT)"],
        base_impute_stride=2,
        verbose=False,
    )
    loaded = PCRCorrector.load_from_checkpoint(path, **kwargs)
    torch.testing.assert_close(
        loaded.model.delta_head.bias,
        obj.model.delta_head.bias,
    )
    assert loaded.model.training is False
