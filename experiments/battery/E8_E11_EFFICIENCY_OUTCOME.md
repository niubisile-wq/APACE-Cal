# E8/E11 efficiency audit

The existing EOL-cycle cost Pareto audit is retained as the only cost claim; it does not claim energy or guaranteed parallel wall-clock savings. A separate CPU scaling audit now measures the unlabeled geometry and facility-selection implementation at n=30, 60, 120, 240, and 500. At n=500, both distance matrices occupy 4.0 MB, geometry takes 4.18 s on the current CPU, and selecting K=10 supports takes 0.040 s. This is consistent with the declared O(n²d) geometry, O(K n_a n) facility selection, and O(n²) distance memory. No GPU or learned model is required by the frozen method.

Artifact: `batterylife_apace_efficiency_audit.json`.
