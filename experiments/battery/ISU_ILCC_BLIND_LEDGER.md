# ISU-ILCC v2 confirmation attempt ledger

Date: 2026-08-20 UTC

## Prelabel state

- Candidate method: frozen APACE-Cal v2 in `METHOD_FREEZE_V2.md`.
- Official archive: `ISU_ILCC.zip`, 10,063,843,011 bytes, MD5
  `98c0561ff25eb68538572c54aeb279ea`; full ZIP CRC test passed.
- Label membership only: 240 top-level keys were lexically read while every
  value was skipped.
- Frozen manifest SHA-256:
  `59a67a67a2a24f730df13464047357d9f54bf51aa6adbaf8a7d4cd9c14de2a23`.
- Active settings frozen before labels: H10/K3, H20/K3, H50/K3.
- Unlabeled diagnostics: `D_p=0.675387`; rho H10/H20/H50 =
  `0.423029/0.443599/0.473347`, routing all three active settings to the
  evidence-coupled `w0.5_bw0.5` kernel.

## One-shot outcome

The frozen evaluator opened the official label mapping only after all chain
hashes passed. It then stopped before any prediction, error, p-value, or result
JSON was produced because the official label for `ISU-ILCC_G40C3.pkl` is 16,
while that pickle supplies features beyond H=20. This violates the frozen
evaluator invariant `life >= visible horizon` and raised:

`RuntimeError: Label life 16.0 shorter than visible horizon 20: ISU-ILCC_G40C3.pkl`

The redirected result log is empty (SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`),
and no blind-evaluation JSON exists.

Post-failure inspection found exactly zero labels below H10 and exactly this
one label below H20 and H50. These counts were obtained only after the failed
label opening and are not prelabel evidence.

## Evidence classification

- This is a disclosed **technical failure after label opening**, not a method
  success or method-performance failure.
- ISU-ILCC is no longer an untouched blind domain.
- Silently dropping the cell, changing its EOL, changing the cohort per H, or
  modifying the evaluator and calling the rerun blind is prohibited.
- A fixed-manifest diagnostic recovery may be run later, but must be labelled
  post-open/non-confirmatory.
- Independent confirmation still requires a different untouched target domain.
