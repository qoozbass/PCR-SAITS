import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pypots") is None,
    reason="real PyPOTS integration requires installed pypots",
)

from pcrsaits import CSDIBackbone, PCRCSDI, build_windows


def _kwargs():
    return dict(
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
        n_diffusion_steps=2,
        n_sampling_times=2,
        aggregation="median",
        sampling_seed=7,
        verbose=False,
    )


def test_real_pypots_csdi_and_pcrcsdi_end_to_end(tmp_path):
    rng = np.random.default_rng(7)
    train = rng.normal(size=(48, 2))
    val = rng.normal(size=(24, 2))
    test = rng.normal(size=(16, 2))

    train_w, _ = build_windows(train, 4, stride=4)
    val_ori_w, _ = build_windows(val, 4, stride=4)
    val_masked = val.copy()
    val_masked[3:5, 0] = np.nan
    val_w, _ = build_windows(val_masked, 4, stride=4)

    backbone = CSDIBackbone(**_kwargs())
    backbone.fit(train_w, val_w, val_ori_w)

    masked = test.copy()
    masked[4:7, 0] = np.nan
    masked_w, _ = build_windows(masked, 4, stride=4)

    samples1 = backbone.sample(masked_w)
    samples2 = backbone.sample(masked_w)
    assert samples1.shape == (4, 2, 4, 2)
    assert np.array_equal(samples1, samples2)

    observed_w = np.isfinite(masked_w)
    assert np.array_equal(
        backbone.impute(masked_w)[observed_w],
        masked_w[observed_w],
    )

    csdi_ckpt = tmp_path / "csdi.pypots"
    backbone.save(csdi_ckpt)
    restored_backbone = CSDIBackbone.load_from_checkpoint(
        csdi_ckpt,
        **_kwargs(),
    )
    restored_samples = restored_backbone.sample(masked_w)
    assert np.array_equal(samples1, restored_samples)

    pcr = PCRCSDI(
        backbone=restored_backbone,
        feature_names=["sensor_1", "sensor_2"],
        feature_groups=["sensor", "sensor"],
        n_steps=4,
        base_impute_stride=4,
        epochs=2,
        batch_size=16,
        patience=1,
        verbose=False,
    )
    pcr.fit(
        train,
        val,
        seed=7,
        holdout_ratio=0.15,
        pointwise_fraction=0.5,
        block_patterns=(1, 2, 3),
        block_buffer=1,
    )
    out1 = pcr.impute(masked)
    observed = np.isfinite(masked)
    assert out1.shape == masked.shape
    assert np.isfinite(out1[~observed]).all()
    assert np.array_equal(out1[observed], masked[observed])

    pcr_ckpt = tmp_path / "pcrcsdi.pt"
    pcr.save(pcr_ckpt)
    restored_pcr = PCRCSDI.load(
        pcr_ckpt,
        backbone=restored_backbone,
    )
    out2 = restored_pcr.impute(masked)
    assert np.array_equal(out1, out2)
