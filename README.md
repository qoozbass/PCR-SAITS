# PCR-SAITS

**PCR-SAITS** is a lightweight residual-correction package for multivariate time-series imputation.  
Version **1.1.0** exposes the same PCR correction core over three backbone adapters:

- **SAITS**
- **BRITS**
- **CSDI**

The PCR layer is trained on top of an already-trained imputation backbone. It does **not** replace or retrain the backbone; instead, it learns a residual correction for cells that are missing in the supplied input while preserving originally observed values exactly.

## Associated paper

**PCR-SAITS: A Lightweight Disagreement-Based Residual Corrector for SAITS-Based Multivariate Time Series Imputation**  
Sawet Somnugpong, *Expert Systems with Applications* (2026)  
DOI: https://doi.org/10.1016/j.eswa.2026.134510

## Release

Current public release:

```text
PCR-SAITS v1.1.0
```

Version 1.1.0 adds **CSDI** as the third supported backbone while retaining the existing PCR correction mechanism used by PCR-SAITS and PCR-BRITS.

### Highlights in v1.1.0

- Added `CSDIBackbone`
- Added public `PCRCSDI`
- Added access to raw CSDI diffusion samples through `CSDIBackbone.sample()`
- Added deterministic point imputation from CSDI samples through `CSDIBackbone.impute()`
- Added CSDI checkpoint save/load support
- Added end-to-end CSDI/PCR-CSDI integration tests
- Preserved the existing PCR residual-correction core
- Preserved exact restoration of originally observed values
- Updated the minimum PyPOTS requirement to `pypots>=1.5`

The experimental PCR-CSDI-UQ path is **not** part of the public v1.1.0 API.

## Installation

### From PyPI

```bash
python -m pip install pcrsaits==1.1.0
```

Core dependencies are:

- `numpy`
- `torch`
- `pypots>=1.5`

Python requirement:

```text
Python >= 3.9
```

### From source

Clone or download the repository, then run:

```bash
python -m pip install .
```

## Public API

The main public classes are:

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

The package also exposes lower-level utilities such as:

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

| Backbone | PCR wrapper | Backbone output used by PCR |
|---|---|---|
| SAITS | `PCRSAITS` | deterministic imputation |
| BRITS | `PCRBRITS` | deterministic imputation |
| CSDI | `PCRCSDI` | deterministic aggregation of diffusion samples |

All three PCR wrappers use the same public PCR correction layer. Adding CSDI does not change the PCR feature construction, residual network, loss, masking logic, or window reconstruction logic.

## Data format

The public PCR wrappers operate on 2-D arrays:

```text
[time, features]
```

For example:

```python
train_values.shape
# (n_train_time_steps, n_features)

val_values.shape
# (n_val_time_steps, n_features)

masked_values.shape
# (n_test_time_steps, n_features)
```

Missing values are represented by `NaN`.

The supplied backbone must already be trained before calling the PCR wrapper's `fit()` method.

## Feature metadata

For new datasets, **explicit feature metadata** is the default.

Example:

```python
feature_names = ["PM2.5", "TEMP", "WSPM"]

feature_groups = [
    "pollutant",
    "meteorological",
    "meteorological",
]
```

Allowed public feature groups are:

- `pollutant`
- `sensor`
- `meteorological`

A PCR wrapper can then be constructed as:

```python
model = PCRSAITS(
    backbone=trained_saits,
    feature_names=feature_names,
    feature_groups=feature_groups,
)
```

The same interface is used for BRITS and CSDI:

```python
model_brits = PCRBRITS(
    backbone=trained_brits,
    feature_names=feature_names,
    feature_groups=feature_groups,
)

model_csdi = PCRCSDI(
    backbone=trained_csdi,
    feature_names=feature_names,
    feature_groups=feature_groups,
)
```

### Legacy paper-suite metadata inference

For reproduction of the historical paper-suite naming convention, legacy name inference remains available:

```python
model = PCRSAITS(
    backbone=trained_saits,
    feature_names=["CO(GT)", "PT08.S1(CO)", "T"],
    metadata_mode="paper_legacy_inference",
)
```

For new datasets, explicit `feature_groups` are recommended.

## PCR wrapper configuration

The three public wrappers share the same constructor surface.

Typical configuration:

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

The same arguments can be used with `PCRBRITS` and `PCRCSDI`.

## Fitting the PCR corrector

After the backbone has been trained:

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

The PCR layer uses deterministic holdout-mask construction for a fixed seed.

The backbone is supplied to PCR as an already-trained model; PCR fitting trains the residual corrector rather than retraining the backbone.

## Imputation

Given a time series containing missing values:

```python
imputed = model.impute(masked_values)
```

PCR correction is applied to cells that are missing in the supplied input.

Originally observed cells are restored exactly in the returned array.

## SAITS quick start

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

## BRITS quick start

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

## CSDI behavior

`CSDIBackbone` preserves access to CSDI's probabilistic diffusion samples while also providing a deterministic point-imputation interface for the PCR core.

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

The raw diffusion ensemble remains available through `sample()`.

When `sampling_seed` is an integer, repeated sampling for a fixed trained model and fixed input is deterministic under the adapter's seeded-sampling behavior. Set:

```python
sampling_seed=None
```

when stochastic inference is desired.

CSDI samples and deterministic point imputations restore observed values exactly at the adapter boundary.

## PCR-CSDI quick start

A minimal example is included in:

```text
examples/quickstart_pcrcsdi.py
```

The basic workflow is:

```python
from pcrsaits import CSDIBackbone, PCRCSDI, build_windows

backbone = CSDIBackbone(
    n_steps=48,
    n_features=n_features,
    epochs=100,
    batch_size=32,
    patience=10,
    n_sampling_times=10,
    aggregation="median",
    sampling_seed=7,
)

train_windows, _ = build_windows(train_values, 48, stride=48)

backbone.fit(train_windows)

model = PCRCSDI(
    backbone=backbone,
    feature_names=feature_names,
    feature_groups=feature_groups,
    n_steps=48,
    base_impute_stride=24,
)

model.fit(train_values, val_values, seed=7)

imputed = model.impute(masked_values)
```

For an executable small end-to-end example using real PyPOTS CSDI, see:

```text
examples/quickstart_pcrcsdi.py
```

A dedicated real-integration verifier is also provided:

```text
scripts/verify_pcrcsdi_real_integration.py
```

## Save and load

Backbone checkpoints and PCR checkpoints are intentionally separate.

This design keeps the pretrained imputation backbone independent from the lightweight PCR correction layer.

### SAITS

Save:

```python
trained_saits.save("saits_backbone.pypots")
model.save("pcrsaits.pt")
```

After restoring the SAITS backbone:

```python
model = PCRSAITS.load(
    "pcrsaits.pt",
    backbone=restored_saits,
)
```

### BRITS

Save:

```python
trained_brits.save("brits_backbone.pypots")
model.save("pcrbrits.pt")
```

After restoring the BRITS backbone:

```python
model = PCRBRITS.load(
    "pcrbrits.pt",
    backbone=restored_brits,
)
```

### CSDI

Save:

```python
trained_csdi.save("csdi_backbone.pypots")
model.save("pcrcsdi.pt")
```

Restore the CSDI backbone using the same backbone configuration used when it was created:

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

PCR checkpoint files contain the PCR model state and public metadata; they do not bundle the backbone weights.

## Examples

The repository includes:

```text
examples/quickstart_pcrsaits.py
examples/quickstart_pcrbrits.py
examples/quickstart_pcrcsdi.py
```

These examples demonstrate the intended public API rather than dataset-specific preprocessing.


## Research ablations

The stable v1.1.0 public API supports explicit feature groups and the legacy
paper-suite inference mode. A **true no-feature-group configuration is not a
public API mode**.

For research analysis, the repository includes:

```text
experiments/no_feature_group_ablation.py
```

This experiment removes only the two PCR domain-tag inputs while keeping the
remaining PCR configuration unchanged. It compares, for each supported
backbone:

```text
Backbone
PCR-Group
PCR-NoGroup
```

across SAITS, BRITS, and CSDI.

`feature_groups=None` under `metadata_mode="explicit"` should not be interpreted
as the no-group ablation: explicit mode requires one supported group for each
feature. Likewise, `metadata_mode="paper_legacy_inference"` still infers feature
groups from feature names.

See:

```text
experiments/README.md
```

for the protocol, commands, outputs, and interpretation. This experiment is
research-only and does not change the stable public v1.1.0 API.

## Paper reproduction

Historical experiment programs supplied by the author are preserved separately under:

```text
paper_reproduction/legacy_scripts/
```

with their original SHA256 manifest.

The reusable `pcrsaits` package and the historical paper-reproduction scripts are intentionally separated so that changes to the reusable package are not silently presented as changes to the published experimental protocol.

To verify the preserved paper-reproduction sources:

```bash
python paper_reproduction/verify_sources.py
```

Additional reproduction information is available in:

- `REPRODUCIBILITY.md`
- `paper_reproduction/README.md`
- `paper_reproduction/PHYSIONET_PROTOCOL.md`

## Release artifacts

Qualified v1.1.0 release distributions:

```text
pcrsaits-1.1.0-py3-none-any.whl
SHA256: 45b0454f61c255d458b1d1734d97f176a4fc18e7070036e9e7181da4cf3d9dbb

pcrsaits-1.1.0.tar.gz
SHA256: e013c276666e9bb1b54cd6e23c60818e4705bad1d7537bd9b4c28d157d0d06cb
```

The same qualified distributions were used for the public v1.1.0 release.

## Citation

If you use PCR-SAITS in research, please cite the associated paper and the appropriate software record.

### Paper

Somnugpong, S. (2026).  
**PCR-SAITS: A Lightweight Disagreement-Based Residual Corrector for SAITS-Based Multivariate Time Series Imputation.**  
*Expert Systems with Applications.*  
https://doi.org/10.1016/j.eswa.2026.134510

### Zenodo

All-version / concept DOI:

```text
10.5281/zenodo.22973878
```

https://doi.org/10.5281/zenodo.22973878

Version-specific DOI for PCR-SAITS v1.1.0:

```text
10.5281/zenodo.23000374
```

https://doi.org/10.5281/zenodo.23000374

For exact reproducibility, cite the version-specific DOI corresponding to the software version used.

The original v1.0.0 software record remains:

```text
10.5281/zenodo.22973879
```

## Links

- Repository: https://github.com/qoozbass/PCR-SAITS
- PyPI: https://pypi.org/project/pcrsaits/
- Paper: https://doi.org/10.1016/j.eswa.2026.134510
- Zenodo concept DOI: https://doi.org/10.5281/zenodo.22973878
- Zenodo v1.1.0 DOI: https://doi.org/10.5281/zenodo.23000374

## License

MIT License. See `LICENSE`.
