"""One-command integrity and red-line verification for APACE-Cal v2 evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
EXPECTED = {
    ROOT / "METHOD_FREEZE_V2.md": "2afc722a5a77472b779a112304f616a53df26779d4cfb1151f7089064243d1ba",
    ROOT / "CONFIRMATION_FREEZE_SNU.md": "def4183ec017c12bb0c3bf2f8b1100f0c0961d3b8abb5e485e8316075f69a4c5",
    HERE / "batterylife_asymmetric_cohort_router_v2.py": "438a0eeff5091e7bc65c2ed79eafafc9100e007c23800630c36adf90f6f9549b",
    HERE / "batterylife_asymmetric_cohort_router_v2.json": "101c0d9b3161fb32c3ac80df8efc2a08089caddea82facfaba68bfbe323bbd65",
    HERE / "batterylife_apace_v2_postblind_audit.json": "5b83cf7bde0c135aff10c544be909e7a30babf275eca2548440696acfe93daaf",
    HERE / "batterylife_snu_dynamic_blind.py": "b195ddadb59fe8fe0f57232a9f30c941929149db166e2e1ed02b4784beda8e99",
    HERE / "snu_dynamic_dataset1_frozen_prelabel_manifest.json": "a5115c810617f069a6245a2dcfa44c27ddc0e7c558138f2b81edd1813d783095",
    HERE / "snu_dynamic_dataset1_blind_eval.json": "655fd2f8bbc829d738635cc6159a66d039069a84dbd8f3507c7ec1f25fa1af7d",
    HERE / "snu_dynamic_dataset1_blind_stats.json": "72d52bd10fbe458dc5e7b45d1f67b8ad5e931f8d2e0624911c9791eb01ca9557",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    checks = []
    for path, expected in EXPECTED.items():
        actual = sha256(path)
        checks.append({"check": f"sha256:{path.relative_to(ROOT)}",
                       "passed": actual == expected,
                       "expected": expected, "actual": actual})
    manifest = json.loads((HERE / "snu_dynamic_dataset1_frozen_prelabel_manifest.json").read_text())
    result = json.loads((HERE / "snu_dynamic_dataset1_blind_eval.json").read_text())
    active = {(x["horizon"], x["label_budget_k"])
              for x in result["results"] if any(
                  r.startswith("active_") for r in x["route_counts"])}
    checks.append({"check": "active setting equals frozen H50/K3",
                   "passed": active == {(50, 3)}, "actual": sorted(active)})
    target = next(x for x in result["results"]
                  if x["horizon"] == 50 and x["label_budget_k"] == 3)
    checks.append({"check": "active relative improvement >=10%",
                   "passed": target["relative_mape_reduction_percent"] >= 10.0,
                   "actual": target["relative_mape_reduction_percent"]})
    fallback_exact = all(
        all(abs(x["baseline_ape"] - x["method_ape"]) <= 1e-12 and
            abs(x["baseline_ae"] - x["method_ae"]) <= 1e-12
            for x in row["per_cell"])
        for row in result["results"]
        if (row["horizon"], row["label_budget_k"]) != (50, 3)
    )
    checks.append({"check": "all 11 fallback settings bit-identical",
                   "passed": fallback_exact})
    checks.append({"check": "manifest declares hard early-information barrier",
                   "passed": "TotCycle > 50" in manifest["information_barrier"]})
    output = {"passed": all(x["passed"] for x in checks), "checks": checks}
    print(json.dumps(output, indent=2))
    raise SystemExit(0 if output["passed"] else 1)


if __name__ == "__main__":
    main()
