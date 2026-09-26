# Final PhysioNet protocol boundary

The final paper's PhysioNet result must not be reproduced with the earlier
flatten-and-fill path in `pcr_saits_new_datasets.py`.

The final reviewer-corrected chain is:

```text
patient-aware full v8
    -> patient-aware diagnostic v7
        -> validity-corrected full v6
            -> v7.1 core
```

Protocol invariants recorded for the final chain:

- 48 steps per patient stay;
- artificial evaluation targets are originally observed cells only;
- natural missing cells are not evaluation targets;
- PCR train/validation holdouts do not select natural missing cells;
- local references and temporal blocks must not cross patient boundaries.

The exact authoritative objects for the v8/v7/v6 scripts are recorded in
`FINAL_PAPER_SOURCE_REFERENCES.json`.

Those objects are not raw-byte attachments in the active release workspace, so
this release candidate does not fabricate copies from snippets. If full
one-command reproduction of final Table 18 is required for v1.0.0, export and
add those exact scripts (or a regression-tested flattened replacement) before
the final tag.
