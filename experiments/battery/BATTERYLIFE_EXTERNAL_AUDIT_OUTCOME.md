# BatteryLife standardized external audit

## Prelabel screen

The official processed archives were read without opening the Life labels
archive. HUST (77 cells), Tongji (119 eligible cells), and XJTU (23 cells)
yielded 219 eligible cells. A pooled HUST+Tongji+XJTU manifest was frozen with
K=3 active routes at H=10,20,50. The remaining K settings were precommitted
fallbacks.

## Label-contract result

The official Tongji label file contains 108 keys for 130 processed members; 22
processed members therefore have no official label. The frozen 219-cell pooled
manifest cannot be evaluated without changing episode identities after label
opening, so it is recorded as a data-contract failure, not silently filtered.

The complete-label HUST+XJTU audit contains 100 cells. Its protocol dispersion
falls below the active gate, so all 12 settings safely fallback and are exactly
equal to baseline (0/100/0, paired p=1). This is a large-sample safety audit,
not an active-gain claim.

Stanford prelabel screening found 41 eligible cells with zero protocol
dispersion; labels were not opened and no active claim is made.

SDU prelabel screening found 86 eligible cells, also with zero protocol
dispersion; its labels were not opened and no active claim is made.

MATR contains 169 eligible cells and was evaluated after the prelabel freeze.
Its protocol dispersion is 0.4337, which falls in the precommitted medium-
dispersion abstention band. All 12 settings therefore fallback exactly:

- H10/K3: 41.9544% → 41.9544%, 0/169/0, p=1;
- H20/K3: 41.1856% → 41.1856%, 0/169/0, p=1;
- H50/K3: 39.9781% → 39.9781%, 0/169/0, p=1.

This is the largest standardized external safety audit in the current package,
not an active-gain claim. The active external evidence remains the frozen SNU
and 27-cell MathWorks audits.

The failed pooled audit is retained because it demonstrates why label coverage
must be checked before freezing an external episode manifest.
