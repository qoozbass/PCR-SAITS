# PCR-SAITS

PCR-SAITS is a lightweight residual-correction layer for multivariate
time-series imputation. The development tree now exposes the same audited PCR
core over three backbone adapters: **SAITS, BRITS, and CSDI**.

The published paper remains:

**PCR-SAITS: A Lightweight Disagreement-Based Residual Corrector for
SAITS-Based Multivariate Time Series Imputation**  
Sawet Somnugpong, *Expert Systems with Applications* (2026)  
DOI: https://doi.org/10.1016/j.eswa.2026.134510

## Release status

This source tree is the finalized `v1.1.0` source candidate produced from a cryptographically bound Phase-3 qualification PASS. Published `v1.0.0` artifacts and tags remain immutable. Phase 3.7 final artifact build/install/hash qualification must still pass before publication.

## Install from source

```bash
python -m pip install .
```

Core dependencies are `numpy`, `torch`, and `pypots`.

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

All three PCR wrappers reuse the same audited `PCRCorrector`. Adding CSDI does
not change the PCR feature construction, residual network, loss, mask logic, or
window logic.

For a new dataset, explicit feature metadata remains the default:

```python
model = PCRCSDI(
    backbone=trained_csdi,
    feature_names=["PM2.5", "TEMP", "WSPM"],
    feature_groups=["pollutant", "meteorological", "meteorological"],
)
model.fit(train_values, val_values, seed=7)
imputed = model.impute(masked_values)
```

Allowed public feature groups are `pollutant`, `sensor`, and
`meteorological`. Legacy paper-suite name inference is still available through
`metadata_mode="paper_legacy_inference"`.

## CSDI backbone behavior

`CSDIBackbone` preserves CSDI's probabilistic samples while providing the
deterministic adapter surface required by the unchanged PCR core:

```python
samples = trained_csdi.sample(masked_windows)  # [N, S, L, F]
point = trained_csdi.impute(masked_windows)    # [N, L, F]
```

`impute()` uses the median across diffusion samples by default. The raw sample
ensemble remains available through `sample()`. Phase 2 does **not** add the
experimental PCR-CSDI-UQ feature path; Phase 3 preserves that exclusion and CSDI enters only as the third backbone
adapter.

When `sampling_seed` is an integer, repeated inference is deterministic for a
fixed trained model/input and external CPU/CUDA RNG state is restored. Set
`sampling_seed=None` for stochastic inference.

## Important inference behavior

PCR correction is applied to every cell that is missing in the supplied input.
Originally observed cells are restored exactly. CSDI samples and CSDI point
imputations also restore observed values exactly at the adapter boundary.

## Save / load

Backbone and PCR checkpoints remain separate:

```python
trained_csdi.save("csdi_backbone.pypots")
model.save("pcrcsdi.pt")
```

Restore the backbone first, then the PCR wrapper:

```python
restored_csdi = CSDIBackbone.load_from_checkpoint(
    "csdi_backbone.pypots",
    # same constructor configuration used for the backbone
    **csdi_config,
)
model = PCRCSDI.load("pcrcsdi.pt", backbone=restored_csdi)
```

## Examples

- `examples/quickstart_pcrsaits.py`
- `examples/quickstart_pcrbrits.py`
- `examples/quickstart_pcrcsdi.py`

A real PyPOTS integration verifier is provided at:

```text
scripts/verify_pcrcsdi_real_integration.py
```

## Paper reproduction

Historical experiment programs supplied by the author remain preserved under
`paper_reproduction/legacy_scripts/` with their original SHA256 manifest. The
Phase-2 integration does not alter those paper-reproduction sources.

See `REPRODUCIBILITY.md` for the separation between reusable package code and
historical research scripts.

## Citation

`CITATION.cff` describes software release candidate `v1.1.0`. The DOI below identifies the existing Zenodo software record; archival metadata is updated only when publication occurs.

Software archive DOI:
https://doi.org/10.5281/zenodo.22973879

## License

MIT License. See `LICENSE`.
