# E7 route stability completion outcome

The full route audit now includes all 18 development-domain/horizon cells, four K values, 100 deterministic 1% input perturbations per row, and the five best/five worst H50/K3 per-cell changes.

The result is intentionally mixed. CALB's low-dispersion active branch is stable for K=3, but zero/medium-dispersion domains can cross a hard route boundary under tiny perturbations; HNEI, CALCE and MICH show 100% route flips in the corresponding affected K branches. This is a genuine limitation of hard threshold routing and is retained as a red-line result rather than hidden. It motivates reporting the router as a safety gate under exact metadata, not as a perturbation-invariant classifier.

The largest H50/K3 improvements are concentrated in CALB cells, while the largest adverse cell-level changes include MICH_EXP and SNL cells. Full values and route phase data are in `batterylife_apace_route_stability_full.json`.
