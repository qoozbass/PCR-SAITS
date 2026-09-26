# Reproducibility notes

## Reusable algorithm

The `pcrsaits/` package is the recommended entry point. Its numerical core was
frozen and legacy-equivalence tested before the public API layer was added.

## Research scripts

`paper_reproduction/legacy_scripts/` preserves the author's supplied program
archive byte-for-byte. The scripts are intentionally not imported by the
library package.

## Environment

`requirements.txt` lists reusable-library dependencies.
`requirements-reproduction.txt` lists additional dependencies seen in the
research scripts.

These dependency files are not an exact historical environment lock because a
complete paper-run package/version freeze was not present in the supplied
source archive. Do not interpret successful installation today as proof that
every historical benchmark is bitwise reproducible.

## Data and checkpoints

Large datasets, downloaded benchmark caches, trained checkpoints, and generated
result tables are not included in this source release candidate.

## PhysioNet

The final paper used a patient-aware, originally-observed-target protocol. The
older script that concatenates patient stays is historical evidence only and
must not be represented as the final Table 18 protocol. See
`paper_reproduction/PHYSIONET_PROTOCOL.md`.


## Licensing

The reusable software is released under the MIT License.
