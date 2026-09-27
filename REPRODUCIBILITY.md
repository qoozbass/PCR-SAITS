# Reproducibility notes

## Reusable algorithm

The `pcrsaits/` package is the recommended entry point. Its PCR numerical core
was frozen and legacy-equivalence tested before the public API layer was added.
Phase 2 adds CSDI only through a backbone adapter and public wrapper; it does
not alter PCR correction mathematics.

## CSDI stochasticity

`CSDIBackbone.sample()` exposes the diffusion ensemble. For deterministic PCR
integration, `impute()` aggregates the ensemble (median by default). With an
integer `sampling_seed`, the adapter forks/restores caller RNG state and repeats
the same stochastic sampling path for a fixed model/input. Setting
`sampling_seed=None` retains stochastic inference.

## Research scripts

`paper_reproduction/legacy_scripts/` preserves the author's v1.0.0 supplied
program archive byte-for-byte. The Phase-2 integration does not rewrite those
historical scripts and they are not imported by the reusable library package.

## Environment

Phase 3.1–3.5 runtime qualification passed with PyPOTS `1.5`. The v1.1.0 release candidate declares `pypots>=1.5` as its selected minimum dependency. This records the release decision; it does not by itself claim that every older PyPOTS version was tested and found incompatible.

## Data and checkpoints

Large datasets, benchmark caches, trained checkpoints, and generated result
tables are not included in this development source tree.

## PhysioNet

The final paper used a patient-aware, originally-observed-target protocol. The
older script that concatenates patient stays is historical evidence only and
must not be represented as the final Table 18 protocol. See
`paper_reproduction/PHYSIONET_PROTOCOL.md`.

## Licensing

The reusable software is released under the MIT License.
