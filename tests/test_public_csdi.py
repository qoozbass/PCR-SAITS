import numpy as np
import torch

from pcrsaits import CSDIBackbone, PCRCSDI


class _DummyBackbone:
    def impute(self, windows):
        return np.nan_to_num(np.asarray(windows, dtype=float), nan=0.0)


def _model():
    return PCRCSDI(
        backbone=_DummyBackbone(),
        feature_names=["sensor_1", "sensor_2"],
        feature_groups=["sensor", "sensor"],
        n_steps=4,
        base_impute_stride=2,
        epochs=1,
        batch_size=2,
        patience=1,
        verbose=False,
    )


def test_public_exports_exist():
    assert CSDIBackbone.__name__ == "CSDIBackbone"
    assert PCRCSDI.__name__ == "PCRCSDI"


def test_pcrcsdi_reuses_unchanged_public_pcr_core():
    model = _model()
    assert model.public_name == "PCRCSDI"
    assert model.core.base_model is model.backbone
    assert model.core.variant == "pcrsaitsv14_no_seasonal_branch"
    assert model.core.use_seasonal_branch is False
    assert model.core.use_domain_tags is True
    assert model.core.direct_residual is True


def test_pcrcsdi_public_checkpoint_roundtrip_metadata(tmp_path):
    model = _model()
    path = tmp_path / "pcrcsdi.pt"
    model.save(path)
    restored = PCRCSDI.load(path, backbone=_DummyBackbone())
    assert restored.public_name == "PCRCSDI"
    assert restored.feature_names == model.feature_names
    for key, value in model.core.model.state_dict().items():
        assert torch.equal(value, restored.core.model.state_dict()[key])
