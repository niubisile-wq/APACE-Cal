# UConn-MathWorks external audit outcome

This archive was frozen using only first-life cycling members. The manifest was
hashed before opening labels; the subsequent run is a one-shot external audit,
not a development run. The archive contains 27 eligible cells, so this result is
treated as a mechanism/transfer confirmation rather than a large-sample claim.

| horizon | K | baseline MAPE | v3 MAPE | reduction | improved/tied/worse | paired p |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 3 | 24.572 | 15.302 | 37.72% | 24/0/3 | 1.14e-05 |
| 20 | 3 | 25.892 | 15.946 | 38.41% | 23/0/4 | 6.41e-07 |
| 50 | 3 | 28.725 | 14.941 | 47.98% | 21/0/6 | 2.21e-05 |

K=1,5,10 are protocol-frozen fallback branches and therefore have exact
baseline equality (0/27/0). The label definition is the maximum first-life
cycle observed across first-life cycling members. This is an external audit;
the dataset is not used to tune thresholds or the predictor freeze.
