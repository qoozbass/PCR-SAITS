# Changelog

## 1.1.0 — 2026-09-27

- Added `CSDIBackbone` as the third supported backbone adapter.
- Added public `PCRCSDI` wrapper over the unchanged audited `PCRCorrector`.
- Added `CSDIBackbone.sample()` for probabilistic diffusion samples with shape
  `[N, S, L, F]`.
- Added deterministic `CSDIBackbone.impute()` using median aggregation by
  default for PCR compatibility.
- Added deterministic sampling controls, external RNG-state restoration,
  validation/parameter guards, observed-value preservation, and CSDI
  checkpoint delegation.
- Added CSDI unit/integration tests and a real-PyPOTS verification script.
- Did not add the experimental PCR-CSDI-UQ feature path.
- Preserved the PCR numerical core and existing SAITS/BRITS adapter logic.

## 1.0.0 — 2026-09-26

- Adopted the MIT License.

- Added public `PCRSAITS` and `PCRBRITS` convenience APIs.
- Added explicit feature metadata and paper-legacy inference compatibility.
- Added PCR checkpoint save/load support.
- Preserved the Phase-4-audited numerical core unchanged through Phase 5.
- Added a separate paper-reproduction layer with SHA256-verified historical
  research scripts.
- Added final-paper source references for deployment, Table 17, and the
  patient-aware PhysioNet chain without substituting older protocols.
- Added citation metadata, reproducibility notes, release manifesting, and a
  clean public release candidate.
