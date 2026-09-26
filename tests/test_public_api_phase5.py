import numpy as np
import pytest
import torch

from pcrsaits import PCRBRITS, PCRSAITS
from pcrsaits.metadata import resolve_core_feature_names


class DummyBackbone:
    def __init__(self, fill=0.25):
        self.fill = float(fill)

    def impute(self, windows):
        x = np.asarray(windows, dtype=float)
        return np.nan_to_num(x, nan=self.fill)


@pytest.mark.parametrize("cls", [PCRSAITS, PCRBRITS])
def test_public_wrappers_use_audited_proposed_core_config(cls):
    obj = cls(
        backbone=DummyBackbone(),
        feature_names=["a", "b", "c"],
        feature_groups=["pollutant", "sensor", "meteorological"],
        epochs=1,
        batch_size=4,
        patience=1,
        verbose=False,
    )
    core = obj.core
    assert core.variant == "pcrsaitsv14_no_seasonal_branch"
    assert core.use_seasonal_branch is False
    assert core.use_local_branch is True
    assert core.use_domain_tags is True
    assert core.direct_residual is True
    assert core.input_dim == 10


def test_explicit_metadata_is_default_and_requires_groups():
    with pytest.raises(ValueError, match="feature_groups is required"):
        PCRSAITS(
            backbone=DummyBackbone(),
            feature_names=["x", "y"],
            epochs=1,
            batch_size=4,
            patience=1,
            verbose=False,
        )


def test_explicit_groups_encode_to_legacy_recognized_core_names():
    names, groups = resolve_core_feature_names(
        ["custom_pollutant", "custom_sensor", "custom_weather"],
        metadata_mode="explicit",
        feature_groups=["pollutant", "sensor", "meteorological"],
    )
    assert groups == ["pollutant", "sensor", "meteorological"]
    assert names[0] == "CO(GT)"
    assert names[1].startswith("PT08.")
    assert names[2] == "T"


def test_paper_legacy_inference_preserves_feature_names():
    original = ["CO(GT)", "PT08.S1(CO)", "T"]
    names, groups = resolve_core_feature_names(
        original,
        metadata_mode="paper_legacy_inference",
    )
    assert names == original
    assert groups is None


def test_paper_legacy_mode_rejects_explicit_groups():
    with pytest.raises(ValueError, match="must be omitted"):
        resolve_core_feature_names(
            ["CO(GT)"],
            metadata_mode="paper_legacy_inference",
            feature_groups=["pollutant"],
        )


def test_invalid_explicit_group_rejected():
    with pytest.raises(ValueError, match="Unknown feature_groups"):
        resolve_core_feature_names(
            ["x"],
            metadata_mode="explicit",
            feature_groups=["other"],
        )


def test_impute_restores_observed_values_exactly():
    torch.manual_seed(123)
    obj = PCRSAITS(
        backbone=DummyBackbone(fill=0.5),
        feature_names=["a", "b"],
        feature_groups=["sensor", "meteorological"],
        n_steps=4,
        base_impute_stride=2,
        epochs=1,
        batch_size=4,
        patience=1,
        verbose=False,
    )
    x = np.arange(16, dtype=float).reshape(8, 2)
    x[2:4, 0] = np.nan
    x[6, 1] = np.nan

    out = obj.impute(x)
    observed = np.isfinite(x)
    np.testing.assert_array_equal(out[observed], x[observed])
    assert np.isfinite(out[~observed]).all()


def test_public_checkpoint_roundtrip_exact(tmp_path):
    torch.manual_seed(321)
    backbone = DummyBackbone(fill=0.4)
    obj = PCRBRITS(
        backbone=backbone,
        feature_names=["x", "y"],
        feature_groups=["pollutant", "sensor"],
        n_steps=4,
        base_impute_stride=2,
        epochs=1,
        batch_size=4,
        patience=1,
        verbose=False,
    )

    path = tmp_path / "pcrbrits.pt"
    obj.save(path)
    restored = PCRBRITS.load(path, backbone=backbone)

    assert restored.feature_names == obj.feature_names
    assert restored.feature_groups == obj.feature_groups
    assert restored.metadata_mode == obj.metadata_mode
    assert restored.core.variant == obj.core.variant

    for key, value in obj.core.model.state_dict().items():
        assert torch.equal(value.cpu(), restored.core.model.state_dict()[key].cpu())


def test_checkpoint_class_mismatch_rejected(tmp_path):
    obj = PCRSAITS(
        backbone=DummyBackbone(),
        feature_names=["x"],
        feature_groups=["sensor"],
        epochs=1,
        batch_size=4,
        patience=1,
        verbose=False,
    )
    path = tmp_path / "pcrsaits.pt"
    obj.save(path)
    with pytest.raises(ValueError, match="not 'PCRBRITS'"):
        PCRBRITS.load(path, backbone=DummyBackbone())


def test_feature_count_validation():
    obj = PCRSAITS(
        backbone=DummyBackbone(),
        feature_names=["x", "y"],
        feature_groups=["sensor", "sensor"],
        epochs=1,
        batch_size=4,
        patience=1,
        verbose=False,
    )
    with pytest.raises(ValueError, match="expects 2"):
        obj.impute(np.ones((8, 3)))
