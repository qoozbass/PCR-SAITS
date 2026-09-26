# Paper reproduction layer

This directory intentionally stays separate from the reusable `pcrsaits/`
package.

## What is bundled

`legacy_scripts/` contains byte-for-byte copies of all 11 programs from the
project archive supplied by the author for this release cycle.

The source manifest records every file size and SHA256. Verify them with:

```bash
python paper_reproduction/verify_sources.py
```

The scripts cover the historical/main paper suite, revision experiments,
post-hoc analyses, additional datasets, per-position exports, figure
generation, and the v7.2 BRITS/CSDI extension path.

Important: these monolithic scripts are preserved for reproducibility and
provenance. They are not the recommended public API. New users should import
`PCRSAITS` or `PCRBRITS` from `pcrsaits`.

## Final-paper reviewer-specific scripts

Some final reviewer-response runners live as separate authoritative source
objects outside the 11-file archive. They are listed in
`FINAL_PAPER_SOURCE_REFERENCES.json`.

This release candidate does not silently substitute older flattened-PhysioNet
logic for the final patient-aware protocol.

## Dataset/environment boundary

Large datasets and trained checkpoints are not bundled. Historical dependency
versions were not available as a complete lockfile, so
`requirements-reproduction.txt` is a compatibility dependency list, not a
claim of an exact historical environment.

Dataset-specific scripts should be run only after reading their module
docstrings and output/provenance safeguards.
