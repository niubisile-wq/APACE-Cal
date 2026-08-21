# E4/E5 completion outcome

The missing preregistered development-domain ablations were run with the same six domains, H/K grid, deterministic episode seeds, target splits, and outer LODO baseline choices. The frozen v2 source and result files were not modified.

## E4 missing single-factor variants

The primary H50/K3 macro MAPE results are:

| variant | baseline → variant | improved/same/worse domains | interpretation |
|---|---:|---:|---|
| A3 fixed concat | 48.535 → 22.838 | 5/0/1 | fixed fusion can improve the mean but loses the safety guarantee |
| A7 all median | 48.535 → 65.641 | 0/3/3 | median is not a universal high-dispersion replacement |
| A8 mismatched evidence | 48.535 → 30.503 | 2/3/1 | acquisition and prediction geometry must be coupled |
| A9 fixed predictor | 59.426 → 23.031 | 3/3/0 | outer predictor selection materially changes the matched baseline |
| A10 no robust scaling | 48.223 → 42.860 | 2/2/2 | scale choice affects both gain and risk |

The full H/K/domain rows are in `batterylife_apace_e4_missing_ablation.json`.

## E5 sensitivity grid

The 27-point sensitivity grid was run on the preregistered primary endpoint H50/K3 with 100 episodes per domain. The frozen point (`low=.30`, `high=.60`, `rho=.35`, bandwidth=.5, predictor distance weight 2) reproduced 48.5352→22.9422% exactly.

Stable regions:

- low threshold 0.20–0.40: identical primary result and 3/3/0 domain safety;
- high threshold 0.55–0.60: identical result and 3/3/0 safety;
- rho threshold 0.20–0.45: identical result and 3/3/0 safety;
- bandwidth .25–.5 and distance weights .5–2 remain positive and safe on the primary endpoint.

Unstable regions are explicitly retained: high threshold .65–.70 reduces the gain; rho=.50 creates one worse domain; bandwidth 2 and weight 0 create worse domains. This supports a plateau rather than an isolated tuned spike, while preserving the original frozen constants.
