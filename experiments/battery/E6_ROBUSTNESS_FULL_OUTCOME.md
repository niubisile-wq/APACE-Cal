# E6 full robustness outcome

Nine pre-registered unlabeled-input perturbation levels were run with the frozen v2 gates and 100 episodes: curve missing 10/20/30%, curve noise 0.5/1/2%, and protocol metadata missing 10/25/50%. H50/K3 macro MAPE is summarized below.

| condition | baseline → method | worse domains |
|---|---:|---:|
| curve missing 10% | 46.572 → 22.554 | 0 |
| curve missing 20% | 43.304 → 22.213 | 0 |
| curve missing 30% | 46.410 → 25.398 | 0 |
| curve noise 0.5% | 46.821 → 24.549 | 1 |
| curve noise 1% | 47.549 → 22.893 | 0 |
| curve noise 2% | 47.268 → 22.803 | 0 |
| protocol missing 10% | 48.605 → 47.525 | 0 |
| protocol missing 25% | 48.568 → 47.403 | 0 |
| protocol missing 50% | 48.701 → 48.701 | 0 |

The method remains directionally useful under all nine aggregate conditions, but the active gain becomes small or disappears under severe protocol loss. The one worse-domain case at 0.5% curve noise and the previously observed H10/H20 stress failures remain disclosed. Label-noise and reduced-acquisition/small-queue tests are tracked separately because they alter the support-label or episode sampling process rather than only the unlabeled geometry.

Those additional H50/K3 tests (30 perturbation seeds × 100 episodes) produced relative macro reductions of 23.17%, 23.03%, and 22.61% for 2%, 5%, and 10% support-label noise; 21.03%, 22.14%, and 23.52% for 50%, 60%, and 80% acquisition pools; and 16.05%, 21.92%, and 23.21% for queue sizes 15, 30, and 60. The smallest queue is measurably harder but remains directionally positive.
