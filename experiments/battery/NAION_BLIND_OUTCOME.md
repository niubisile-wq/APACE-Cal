# NA-ion independent blind outcome

NA-ion was evaluated as an independent external domain using the frozen APACE-Cal v2 chain. The archive and label membership were frozen before label values were opened; the one-shot evaluator verified the archive hash, method freeze, evaluator hash, and development-result hash before decoding labels.

- 34 cells were eligible at each of H=10, 20, and 50.
- 12 settings were evaluated (H ∈ {10,20,50}, K ∈ {1,3,5,10}, 100 episodes each).
- Protocol dispersion was exactly zero at every horizon, so the precommitted method route was `fallback_zero_protocol_dispersion` for all 1,200 episodes.
- Consequently, method and baseline were identical in every setting: 0 improved / 34 same / 0 worse, with paired p=1.
- This is a safety/generalization confirmation of the fallback rule, not evidence of active calibration gain on NA-ion.

Artifacts: `batterylife_naion_prelabel_manifest_v2.json`, `batterylife_naion_blind_eval_v2.json`, and their logs.
