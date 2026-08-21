"""Audit all currently available external candidates before label claims.

The audit is intentionally conservative: a candidate is eligible for a new
strict active confirmation only if its prelabel manifest is frozen, its label
membership is valid, its first-H input contract is satisfied, and at least one
frozen setting routes to active. Existing labels/results are not reopened.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).parent
OUT = HERE / "batterylife_external_candidate_contract_audit.json"


def read(name):
    return json.loads((HERE / name).read_text())


def route_summary(settings):
    counts = {}
    for setting in settings:
        for route, count in setting.get("route_counts", {}).items():
            counts[route] = counts.get(route, 0) + int(count)
    return counts


def main() -> None:
    candidates = []
    for name in (
        "batterylife_ulpur_prelabel_manifest.json",
        "batterylife_sdu_prelabel_manifest.json",
        "batterylife_sdu_membership_corrected_prelabel_manifest.json",
        "snu_dynamic_dataset2_dryrun_prelabel.json",
    ):
        d = read(name)
        routes = route_summary(d.get("settings", []))
        active = any("active" in r for r in routes)
        corrected = "corrected" in name
        candidates.append({
            "manifest": name,
            "dataset": d.get("dataset", d.get("subset", "SNU-dataset2")),
            "phase": d.get("phase"),
            "label_membership_count": d.get("label_membership_count"),
            "routes": routes,
            "prelabel_active_route": active,
            "contract_valid": not (name == "batterylife_sdu_prelabel_manifest.json"),
            "decision": (
                "not eligible: superseded by corrected label-membership manifest"
                if name == "batterylife_sdu_prelabel_manifest.json"
                else "not eligible for active confirmation: no active route"
                if not active
                else "not eligible: corrected contract routes to abstention"
                if corrected
                else "candidate requires separate label-open decision"
            ),
        })
    candidates.extend([
        {
            "manifest": "batterylife_external_rwth_frozen_adaptive.json",
            "dataset": "RWTH",
            "phase": "legacy_external_result",
            "prelabel_active_route": None,
            "contract_valid": False,
            "decision": "excluded: legacy adaptive result is not frozen APACE-Cal v2 blind chain",
        },
        {
            "manifest": "batterylife_luh_blank_external_audit.json",
            "dataset": "Luh-Blank",
            "phase": "technical_audit",
            "prelabel_active_route": None,
            "contract_valid": False,
            "decision": "excluded: no first-50-cycle curve and no common cell-level EOL label",
        },
    ])
    output = {
        "protocol": "candidate screening before any new label opening; no label values read or changed",
        "source_hashes": {
            name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
            for name in (
                "batterylife_ulpur_prelabel_manifest.json",
                "batterylife_sdu_prelabel_manifest.json",
                "batterylife_sdu_membership_corrected_prelabel_manifest.json",
                "snu_dynamic_dataset2_dryrun_prelabel.json",
                "batterylife_external_rwth_frozen_adaptive.json",
                "batterylife_luh_blank_external_audit.json",
            )
        },
        "candidates": candidates,
        "eligible_new_strict_active_candidates": [
            c["dataset"] for c in candidates
            if c.get("contract_valid") and c.get("prelabel_active_route")
        ],
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
