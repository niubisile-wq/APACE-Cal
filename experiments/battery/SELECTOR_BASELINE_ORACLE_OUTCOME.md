# Selector baseline and oracle supplement

The existing E2 fixed-pool rankboard is the formal selector-baseline table:
matched random-K, nested calibration, and ordinary active selection are all
evaluated under the same fixed total-label protocol. It is retained as the
strong-selector comparison rather than adding more prediction backbones.

For an explicit upper-bound diagnostic, a test-aware Monte-Carlo oracle was run
on the 100-cell complete-label HUST+XJTU cohort. For every episode and budget,
500 random K-subsets of the acquisition pool were scored using held-out test
labels, and the best subset was retained. This uses test labels and is
therefore unattainable; it is not a method or a claim.

At H50/K3, matched baseline MAPE was 85.252% and the test-aware oracle-search
upper bound was 16.182% (81.02% reduction). The large gap shows that the
remaining limitation is support selection under label scarcity, not an
intrinsic absence of predictive signal. The oracle is deliberately excluded
from the main comparison table and appears only as a ceiling diagnostic.
