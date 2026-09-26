# PCR-SAITS

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22973879.svg)](https://doi.org/10.5281/zenodo.22973879)

PCR-SAITS is a lightweight residual-correction layer for multivariate
time-series imputation. The public package exposes the same audited PCR core
with SAITS and BRITS backbones through a small user-facing API.

Paper:

**PCR-SAITS: A Lightweight Disagreement-Based Residual Corrector for
SAITS-Based Multivariate Time Series Imputation**  
Sawet Somnugpong, *Expert Systems with Applications* (2026)  
DOI: https://doi.org/10.1016/j.eswa.2026.134510

## Install

From this source tree:

```bash
python -m pip install .
```

Core dependencies are `numpy`, `torch`, and `pypots`.

## Public API

```python
from pcrsaits import PCRSAITS, PCRBRITS
```

Both are thin wrappers around the legacy-equivalence-tested `PCRCorrector`.
A trained SAITS/BRITS backbone is supplied to the PCR wrapper; the PCR layer
does not retrain or own the backbone checkpoint.

For a new dataset, explicit feature metadata is the default:

```python
model = PCRSAITS(
    backbone=trained_saits,
    feature_names=["PM2.5", "TEMP", "WSPM"],
    feature_groups=["pollutant", "meteorological", "meteorological"],
)
model.fit(train_values, val_values, seed=7)
imputed = model.impute(masked_values)
```

Allowed public feature groups are:

- `pollutant`
- `sensor`
- `meteorological`

To reproduce legacy paper-suite name inference, use:

```python
model = PCRSAITS(
    backbone=trained_saits,
    feature_names=["CO(GT)", "PT08.S1(CO)", "T"],
    metadata_mode="paper_legacy_inference",
)
```

## Important inference behavior

PCR correction is applied to every cell that is missing in the supplied input.
Originally observed cells are restored exactly.

## Save / load

Backbone and PCR checkpoints are intentionally separate:

```python
trained_saits.save("saits_backbone.pypots")
model.save("pcrsaits.pt")
```

After restoring the backbone:

```python
model = PCRSAITS.load("pcrsaits.pt", backbone=restored_saits)
```

## Examples

- `examples/quickstart_pcrsaits.py`
- `examples/quickstart_pcrbrits.py`

## Paper reproduction

Historical experiment programs supplied by the author are preserved under
`paper_reproduction/legacy_scripts/` with a SHA256 manifest.

```bash
python paper_reproduction/verify_sources.py
```

Final reviewer-specific Table 17 and patient-aware PhysioNet source identities
are recorded separately so that older protocols are not silently presented as
the final paper protocol.

See:

- `REPRODUCIBILITY.md`
- `paper_reproduction/README.md`
- `paper_reproduction/PHYSIONET_PROTOCOL.md`

## Citation

See `CITATION.cff`.

Software archive DOI: https://doi.org/10.5281/zenodo.22973879

## Release status

This tree is the `v1.0.0` release payload prepared for final pre-tag audit.
The Git tag and archival deposit should be created only after the clean ZIP,
wheel, and sdist hashes are independently verified.


## License

MIT License. See `LICENSE`.
