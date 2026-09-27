# v1.1.0 release qualification gates

## Closed before Phase 3

- [x] CSDI adapter integrated as the third backbone.
- [x] `PCRCSDI` reuses the unchanged PCR numerical core.
- [x] Legacy SAITS/BRITS regression passed.
- [x] Qualification source remained development-versioned through Phase 3.1–3.5.
- [x] Published v1.0.0 artifacts/tags remain immutable.

## Phase 3 mandatory gates

- [x] 3.1 Real PyPOTS CSDI/PCRCSDI runtime + checkpoint evidence PASS.
- [x] 3.2 Tested PyPOTS `1.5`; selected release minimum `pypots>=1.5`.
- [x] 3.3 Fresh `1.1.0.dev0` wheel/sdist built from the qualified source tree.
- [x] 3.4 Development wheel installed in a clean virtual environment; `pip check` PASS.
- [x] 3.5 Installed-wheel public API + real CSDI/PCRCSDI checkpoint smoke PASS.
- [x] 3.6 Final metadata transitioned to `1.1.0` from the bound qualification source.
- [ ] 3.7 Build final artifacts, clean-install the final wheel, verify final source, and freeze SHA256 hashes.

Publication remains NO-GO until Phase 3.7 passes. Multi-GPU RNG testing is environment-dependent and is not a blocker on a host with fewer than two CUDA devices.
