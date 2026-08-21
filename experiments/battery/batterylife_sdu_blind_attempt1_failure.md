# SDU blind evaluation attempt 1: technical pre-outcome failure

- Date: 2026-08-20 UTC
- Manifest: `batterylife_sdu_prelabel_manifest.json`
- Manifest SHA-256: `deee0f975db8f32142e9d08eb8efc44dce9914a1fd21a6494ff942cb0f27f08e`
- Evaluator SHA-256: `1c885807d59ec61f7969d62d35a7fd848effb802ea89cb3eb9ffe5ca056e9a20`
- Failure stage: the evaluator loaded the official label mapping, then stopped at
  the manifest/evaluator identity-set assertion before entering any episode
  prediction loop or computing any error statistic.
- Cause: 16 of the 86 pickle identities had no official life-label key.  The
  original opaque-hash-only prelabel script could not detect this coverage gap.
- Outcome exposure: no label value, prediction, per-cell error, aggregate metric,
  or method-versus-baseline comparison was printed or written.  Only the 16
  missing identity names were emitted by the assertion.
- Repair rule: a new manifest may use top-level label keys as sample-availability
  metadata while lexically skipping all JSON values.  Algorithm, thresholds,
  weights, bandwidth, H/K, seeds, and evaluation formulas remain unchanged.
- Evidentiary consequence: the repaired run must be described as an
  outcome-blind technical rerun, not as an uninterrupted pristine one-shot run.
