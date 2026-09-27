# Research experiments

This directory contains research-only experiments that are intentionally kept
outside the stable public API.

## No-feature-group ablation

Files:

```text
no_feature_group_ablation.py
quickstart_no_feature_group.py
```

Start with the small functional example:

```bash
python experiments/quickstart_no_feature_group.py
```

Then use `no_feature_group_ablation.py` for the larger paired experiment.

Purpose:

- compare the trained backbone alone,
- compare the normal public PCR configuration with explicit feature groups,
- compare an experimental PCR configuration with the two domain-tag inputs
  removed.

The experiment covers:

- SAITS / PCR-SAITS
- BRITS / PCR-BRITS
- CSDI / PCR-CSDI

and evaluates:

- pointwise missingness
- `block_12`
- `block_48`

### Why this is not a public API example

PCR-SAITS v1.1.0 intentionally requires explicit feature groups when
`metadata_mode="explicit"` is used.

This means:

```python
PCRSAITS(
    backbone=trained_saits,
    feature_names=feature_names,
    feature_groups=None,
    metadata_mode="explicit",
)
```

is **not** the supported way to request a no-group model.

Likewise:

```python
metadata_mode="paper_legacy_inference"
```

is not group-free because it still infers pollutant / sensor /
meteorological membership from feature names.

The no-group experiment therefore disables only the two domain-tag inputs
inside the PCR corrector while preserving the remaining public PCR
configuration. It is an ablation for research analysis, not a supported
deployment mode.

### Run

Install the released package first:

```bash
python -m pip install pcrsaits==1.1.0 pandas
```

Run one seed:

```bash
python experiments/no_feature_group_ablation.py \
  --seeds 7 \
  --output-dir pcr_no_feature_group_results \
  --reset
```

Continue additional seeds without `--reset`:

```bash
python experiments/no_feature_group_ablation.py \
  --seeds 21 42 84 168 \
  --output-dir pcr_no_feature_group_results
```

On Windows PowerShell:

```powershell
python .\experiments\no_feature_group_ablation.py `
  --seeds 7 `
  --output-dir pcr_no_feature_group_results `
  --reset
```

then:

```powershell
python .\experiments\no_feature_group_ablation.py `
  --seeds 21 42 84 168 `
  --output-dir pcr_no_feature_group_results
```

### Outputs

```text
pcr_no_feature_group_results/
    raw_results.csv
    group_ablation.csv
    group_ablation_summary.csv
    config.json
```

The main ablation quantities are:

```text
group_benefit_rmse_pct
group_benefit_mae_pct
```

Interpretation:

- positive: explicit feature groups improve PCR
- near zero: feature groups have little effect
- negative: the no-group ablation performs better

This experiment uses internal PCR components and is therefore not covered by
the stable public API compatibility promise.
