# PCR-SAITS

**PCR-SAITS** is a lightweight residual-correction package for multivariate time-series imputation.  
The public package supports the same PCR correction core over three backbone adapters:

- **SAITS**
- **BRITS**
- **CSDI**

Version **1.1.0** adds CSDI support while retaining the existing PCR correction mechanism.

## Associated paper

**PCR-SAITS: A Lightweight Disagreement-Based Residual Corrector for SAITS-Based Multivariate Time Series Imputation**  
Sawet Somnugpong, *Expert Systems with Applications* (2026)  
DOI: https://doi.org/10.1016/j.eswa.2026.134510

## Current release

```text
PCR-SAITS v1.1.0
```

### What's new in v1.1.0

- Added `CSDIBackbone`
- Added public `PCRCSDI`
- Added probabilistic CSDI sampling through `CSDIBackbone.sample()`
- Added deterministic CSDI point imputation through `CSDIBackbone.impute()`
- Added CSDI checkpoint save/load support
- Added real PyPOTS CSDI integration tests
- Preserved exact restoration of originally observed values
- Kept the PCR correction core unchanged
- Updated the minimum PyPOTS dependency to `pypots>=1.5`

The experimental PCR-CSDI-UQ path is **not** part of the public v1.1.0 API.

## Installation

### PyPI

```bash
python -m pip install pcrsaits==1.1.0
```

Requirements:

- Python `>=3.9`
- `numpy`
- `torch`
- `pypots>=1.5`

### From source

```bash
python -m pip install .
```

## Public API

```python
from pcrsaits import (
    SAITSBackbone,
    BRITSBackbone,
    CSDIBackbone,
    PCRSAITS,
    PCRBRITS,
    PCRCSDI,
)
```

Lower-level reusable utilities are also exposed, including:

```python
from pcrsaits import (
    BackboneAdapter,
    PCRCorrector,
    PCRResidualNet,
    build_windows,
    reconstruct_from_windows,
)
```

## Supported backbone / PCR pairs

| Backbone | PCR wrapper | Public behavior |
|---|---|---|
| SAITS | `PCRSAITS` | deterministic backbone imputation + PCR correction |
| BRITS | `PCRBRITS` | deterministic backbone imputation + PCR correction |
| CSDI | `PCRCSDI` | diffusion samples aggregated to deterministic point imputation + PCR correction |

The CSDI integration does **not** change PCR feature construction, residual network, loss, masking logic, or window reconstruction logic.

## Input format

PCR wrappers operate on 2-D arrays:

```text
[time, features]
```

Missing cells are represented by `NaN`.

Example:

```python
train_values.shape
# (n_train_time_steps, n_features)

val_values.shape
# (n_val_time_steps, n_features)

masked_values.shape
# (n_test_time_steps, n_features)
```

The supplied backbone must already be trained before `PCRSAITS.fit()`, `PCRBRITS.fit()`, or `PCRCSDI.fit()` is called.

## Feature metadata

For new datasets, explicit feature metadata is recommended.

```python
feature_names = ["PM2.5", "TEMP", "WSPM"]

feature_groups = [
    "pollutant",
    "meteorological",
    "meteorological",
]
```

Allowed public groups are:

- `pollutant`
- `sensor`
- `meteorological`

Example:

```python
model = PCRSAITS(
    backbone=trained_saits,
    feature_names=feature_names,
    feature_groups=feature_groups,
)
```

The same metadata interface is used by `PCRBRITS` and `PCRCSDI`.

Legacy paper-suite name inference remains available through:

```python
metadata_mode="paper_legacy_inference"
```

For new datasets, explicit `feature_groups` are preferable.

## PCR configuration

The public wrappers share the same PCR constructor surface:

```python
model = PCRSAITS(
    backbone=trained_saits,
    feature_names=feature_names,
    feature_groups=feature_groups,
    metadata_mode="explicit",
    n_steps=48,
    base_impute_stride=24,
    learning_rate=1e-3,
    weight_decay=1e-4,
    epochs=100,
    batch_size=256,
    patience=10,
    verbose=True,
)
```

The same PCR arguments can be used with `PCRBRITS` and `PCRCSDI`.

## Fitting the PCR corrector

After training the backbone:

```python
model.fit(
    train_values,
    val_values,
    seed=7,
    holdout_ratio=0.15,
    pointwise_fraction=0.5,
    block_patterns=(6, 12, 24, 48),
    block_buffer=1,
)
```

For a fixed seed, PCR holdout-mask construction is deterministic.

The backbone is treated as an already-trained base imputer; PCR fitting trains the residual corrector rather than retraining the backbone.

## Imputation

```python
imputed = model.impute(masked_values)
```

PCR correction is applied to input-missing cells. Originally observed cells are restored exactly in the returned array.

# SAITS

## SAITS backbone

```python
import numpy as np
from pcrsaits import SAITSBackbone, PCRSAITS, build_windows

train = np.load("train.npy")
val = np.load("val.npy")
test_masked = np.load("test_masked.npy")

feature_names = ["PM2.5", "TEMP", "WSPM"]
feature_groups = ["pollutant", "meteorological", "meteorological"]

backbone = SAITSBackbone(
    n_steps=48,
    n_features=train.shape[1],
    epochs=100,
    batch_size=128,
    patience=10,
    d_model=64,
    d_ffn=128,
    n_heads=4,
    n_layers=2,
    dropout=0.1,
    verbose=True,
)

train_windows, _ = build_windows(train, 48, stride=48)
val_windows, _ = build_windows(val, 48, stride=48)

backbone.fit(train_windows, val_windows)

model = PCRSAITS(
    backbone=backbone,
    feature_names=feature_names,
    feature_groups=feature_groups,
    n_steps=48,
    base_impute_stride=24,
)

model.fit(train, val, seed=7)

imputed = model.impute(test_masked)
np.save("test_imputed.npy", imputed)
```

This mirrors the repository example:

```text
examples/quickstart_pcrsaits.py
```

# BRITS

## BRITS backbone

```python
import numpy as np
from pcrsaits import BRITSBackbone, PCRBRITS, build_windows

train = np.load("train.npy")
val = np.load("val.npy")
test_masked = np.load("test_masked.npy")

feature_names = ["PM2.5", "TEMP", "WSPM"]
feature_groups = ["pollutant", "meteorological", "meteorological"]

backbone = BRITSBackbone(
    n_steps=48,
    n_features=train.shape[1],
    epochs=100,
    batch_size=128,
    patience=10,
    rnn_hidden_size=64,
    verbose=True,
)

train_windows, _ = build_windows(train, 48, stride=48)
val_windows, _ = build_windows(val, 48, stride=48)

backbone.fit(train_windows, val_windows, val_windows)

model = PCRBRITS(
    backbone=backbone,
    feature_names=feature_names,
    feature_groups=feature_groups,
    n_steps=48,
    base_impute_stride=24,
)

model.fit(train, val, seed=7)

imputed = model.impute(test_masked)
np.save("test_imputed.npy", imputed)
```

This mirrors:

```text
examples/quickstart_pcrbrits.py
```

# CSDI

## CSDI backbone behavior

`CSDIBackbone` preserves the probabilistic CSDI diffusion ensemble while providing a deterministic point-imputation interface required by the unchanged PCR core.

For windowed input:

```python
samples = trained_csdi.sample(masked_windows)
# shape: [N, S, L, F]

point = trained_csdi.impute(masked_windows)
# shape: [N, L, F]
```

where:

- `N` = number of windows
- `S` = number of diffusion samples
- `L` = sequence length
- `F` = number of features

By default, `impute()` aggregates diffusion samples using the **median**.  
The adapter also supports `aggregation="mean"`.

When `sampling_seed` is an integer, repeated sampling for a fixed trained model and fixed input is deterministic under the adapter's seeded-sampling behavior.

For stochastic inference:

```python
sampling_seed=None
```

Both raw CSDI samples and deterministic CSDI point imputations restore originally observed values exactly at the adapter boundary.

## PCR-CSDI quick start

A small real PyPOTS CSDI smoke example is included at:

```text
examples/quickstart_pcrcsdi.py
```

The core workflow is:

```python
import numpy as np
import torch

from pcrsaits import CSDIBackbone, PCRCSDI, build_windows

SEED = 7
np.random.seed(SEED)
torch.manual_seed(SEED)

rng = np.random.default_rng(SEED)

train = rng.normal(size=(48, 2))
val = rng.normal(size=(24, 2))
test = rng.normal(size=(16, 2))

train_w, _ = build_windows(train, 4, stride=4)

val_ori_w, _ = build_windows(val, 4, stride=4)
val_masked = val.copy()
val_masked[3:5, 0] = np.nan
val_w, _ = build_windows(val_masked, 4, stride=4)

backbone = CSDIBackbone(
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
    sampling_seed=SEED,
    verbose=False,
)

backbone.fit(train_w, val_w, val_ori_w)

masked = test.copy()
masked[4:7, 0] = np.nan

model = PCRCSDI(
    backbone=backbone,
    feature_names=["sensor_1", "sensor_2"],
    feature_groups=["sensor", "sensor"],
    n_steps=4,
    base_impute_stride=4,
    epochs=2,
    batch_size=16,
    patience=1,
    verbose=False,
)

model.fit(
    train,
    val,
    seed=SEED,
    holdout_ratio=0.15,
    pointwise_fraction=0.5,
    block_patterns=(1, 2, 3),
    block_buffer=1,
)

out = model.impute(masked)
```

A real-integration verifier is provided at:

```text
scripts/verify_pcrcsdi_real_integration.py
```

# Save / load

Backbone checkpoints and PCR checkpoints are intentionally separate.

PCR checkpoint files contain PCR model state and public metadata; they do not bundle backbone weights.

## SAITS

Save:

```python
trained_saits.save("saits_backbone.pypots")
model.save("pcrsaits.pt")
```

Restore:

```python
restored_saits = SAITSBackbone.load_from_checkpoint(
    "saits_backbone.pypots",
    **saits_config,
)

model = PCRSAITS.load(
    "pcrsaits.pt",
    backbone=restored_saits,
)
```

## BRITS

Save:

```python
trained_brits.save("brits_backbone.pypots")
model.save("pcrbrits.pt")
```

Restore:

```python
restored_brits = BRITSBackbone.load_from_checkpoint(
    "brits_backbone.pypots",
    **brits_config,
)

model = PCRBRITS.load(
    "pcrbrits.pt",
    backbone=restored_brits,
)
```

## CSDI

Save:

```python
trained_csdi.save("csdi_backbone.pypots")
model.save("pcrcsdi.pt")
```

Restore:

```python
restored_csdi = CSDIBackbone.load_from_checkpoint(
    "csdi_backbone.pypots",
    **csdi_config,
)

model = PCRCSDI.load(
    "pcrcsdi.pt",
    backbone=restored_csdi,
)
```

# Examples

Repository examples:

```text
examples/quickstart_pcrsaits.py
examples/quickstart_pcrbrits.py
examples/quickstart_pcrcsdi.py
```

# Paper reproduction

Historical experimental programs supplied for the published study are preserved separately under:

```text
paper_reproduction/legacy_scripts/
```

with their original SHA256 manifest.

The reusable `pcrsaits` package and historical paper-reproduction scripts are intentionally separated so package development is not silently presented as a change to the published experimental protocol.

See:

- `REPRODUCIBILITY.md`
- `paper_reproduction/README.md`
- `paper_reproduction/PHYSIONET_PROTOCOL.md`

To verify preserved paper-reproduction sources:

```bash
python paper_reproduction/verify_sources.py
```

# Qualified v1.1.0 release artifacts

```text
pcrsaits-1.1.0-py3-none-any.whl
SHA256: 45b0454f61c255d458b1d1734d97f176a4fc18e7070036e9e7181da4cf3d9dbb

pcrsaits-1.1.0.tar.gz
SHA256: e013c276666e9bb1b54cd6e23c60818e4705bad1d7537bd9b4c28d157d0d06cb
```

The qualified wheel was also verified after public PyPI installation:

```text
pcrsaits version = 1.1.0
PUBLIC API PASS
```

# Citation

If you use PCR-SAITS in research, cite the associated paper and the software version used.

## Paper

Somnugpong, S. (2026).  
**PCR-SAITS: A Lightweight Disagreement-Based Residual Corrector for SAITS-Based Multivariate Time Series Imputation.**  
*Expert Systems with Applications.*  
https://doi.org/10.1016/j.eswa.2026.134510

## Zenodo software records

All-version / concept DOI:

```text
10.5281/zenodo.22973878
```

https://doi.org/10.5281/zenodo.22973878

PCR-SAITS v1.1.0 version DOI:

```text
10.5281/zenodo.23000374
```

https://doi.org/10.5281/zenodo.23000374

PCR-SAITS v1.0.0 version DOI:

```text
10.5281/zenodo.22973879
```

https://doi.org/10.5281/zenodo.22973879

For exact reproducibility, use the version-specific DOI corresponding to the software release used.

# Links

- Repository: https://github.com/qoozbass/PCR-SAITS
- PyPI: https://pypi.org/project/pcrsaits/
- Paper: https://doi.org/10.1016/j.eswa.2026.134510
- Zenodo concept DOI: https://doi.org/10.5281/zenodo.22973878
- Zenodo v1.1.0 DOI: https://doi.org/10.5281/zenodo.23000374

# License

MIT License. See `LICENSE`.
