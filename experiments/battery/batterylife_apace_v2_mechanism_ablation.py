"""Clean fixed-pool v2 mechanism ablation on the six development domains.

The frozen v2 result is never overwritten.  This run removes only the rho
router by setting its threshold to zero (all positive-rho high-dispersion K3
settings use the evidence-coupled kernel); all seeds, data, and outer baseline
selection remain unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import batterylife_asymmetric_cohort_router_v2 as v2


HERE = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=HERE / "batterylife_apace_v2_no_rho_ablation.json")
    parser.add_argument("--seeds", type=int, default=100)
    args = parser.parse_args()
    original = v2.CONCORDANCE_THRESHOLD
    v2.CONCORDANCE_THRESHOLD = 0.0
    try:
        v2.run((10, 20, 50), (1, 3, 5, 10), args.seeds, args.output)
    finally:
        v2.CONCORDANCE_THRESHOLD = original
    data = json.loads(args.output.read_text())
    data["status"] = "DEVELOPMENT ABLATION; rho router removed; not frozen method"
    data["ablation"] = {
        "changed": "CONCORDANCE_THRESHOLD=0.0; all positive-rho high-dispersion K3 use w0.5/bw0.5 kernel",
        "unchanged": "same six domains, H/K, 70/30 episodes, 100 seeds, outer baseline selection",
    }
    args.output.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
