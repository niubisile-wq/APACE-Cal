"""E11 CPU scaling audit for APACE's unlabeled geometry and support selection."""
from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import numpy as np

from batterylife_curve_aware_support import robust_scale
from batterylife_transductive_pool_acquisition import distance_matrix, select_facilities

HERE = Path(__file__).parent
OUT = HERE / "batterylife_apace_efficiency_audit.json"


def main():
    rng = np.random.default_rng(20260820)
    rows = []
    for n in (30, 60, 120, 240, 500):
        protocol = rng.normal(size=(n, 5))
        curve = rng.normal(size=(n, 56))
        pscale, cscale = robust_scale(protocol), robust_scale(curve)
        start = time.perf_counter()
        dp = distance_matrix(protocol, pscale, 1e9)
        dc = distance_matrix(curve, cscale, 1e9)
        geometry_seconds = time.perf_counter() - start
        start = time.perf_counter()
        combined = np.sqrt(dp * dp + 2.0 * dc * dc)
        tie = np.arange(n, dtype=int)
        support = select_facilities(combined, np.arange(max(n - 2, 1)), np.arange(n), min(10, n - 2), tie)
        selection_seconds = time.perf_counter() - start
        rows.append({
            "n": n,
            "geometry_seconds": geometry_seconds,
            "selection_seconds": selection_seconds,
            "distance_matrix_bytes_each": int(dp.nbytes),
            "distance_matrix_bytes_both": int(dp.nbytes + dc.nbytes),
            "rss_kb_after": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "support_size": len(support),
        })
    output = {
        "theory": {"geometry": "O(n^2 d)", "facility_selection": "O(K n_a n)", "distance_memory": "O(n^2)"},
        "implementation": "NumPy CPU geometry; no learned model or GPU dependency",
        "rows": rows,
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
