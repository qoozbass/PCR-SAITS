# Final v1.0.0 release gates

## Resolved

- Phase 6 scientific/software integrity audit: PASS WITH REQUIRED CHANGES.
- Final licensing choice: MIT.
- Package metadata updated to SPDX `MIT`.
- Exact final R4.2/R4.3 scripts remain a documented non-blocking limitation;
  this release does not claim complete one-command final-paper reproduction.

## Still required before publishing/tagging

1. Independently verify the actual clean `PCR-SAITS_v1.0.0_public.zip`.
2. Verify the built wheel and sdist hashes.
3. Create the Git tag `v1.0.0` from exactly the verified clean release tree.
4. Upload the same source release payload to the archival service.
5. Verify the archival DOI metadata and deposited bytes after upload.

Do not change source files after the final SHA256 manifest is generated.
