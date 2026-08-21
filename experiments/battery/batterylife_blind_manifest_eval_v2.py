"""One-shot label-opening evaluator for a frozen APACE-Cal v2 manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import batterylife_asymmetric_cohort_router_v2 as v2
import batterylife_blind_manifest_eval as common_eval
import batterylife_blind_prelabel_manifest as common_prelabel


HERE = Path(__file__).parent
FREEZE_RECORD = HERE.parents[1] / "METHOD_FREEZE_V2.md"
DEV_RESULT = HERE / "batterylife_asymmetric_cohort_router_v2.json"


def verify(manifest, manifest_path, label_file):
    if manifest["phase"] != "PRELABEL_FROZEN_MANIFEST_V2":
        raise RuntimeError("Not a v2 prelabel frozen manifest")
    archive = Path(manifest["archive"])
    checks = [
        (archive.stat().st_size == manifest["archive_size"], "archive size"),
        (common_prelabel.md5(archive) == manifest["archive_md5"], "archive MD5"),
        (common_prelabel.sha256(label_file) == manifest["label_file_opaque_sha256"],
         "label hash"),
        (common_prelabel.sha256(FREEZE_RECORD) == manifest["method_freeze_sha256"],
         "freeze record"),
        (common_prelabel.sha256(Path(__file__)) == manifest["evaluator_script_sha256"],
         "evaluator"),
        (common_prelabel.sha256(
            HERE / "batterylife_blind_prelabel_manifest_v2.py") ==
         manifest["prelabel_script_sha256"], "prelabel builder"),
        (common_prelabel.sha256(
            HERE / "batterylife_blind_prelabel_manifest.py") ==
         manifest["common_prelabel_dependency_sha256"], "prelabel dependency"),
        (common_prelabel.sha256(
            HERE / "batterylife_asymmetric_cohort_router_v2.py") ==
         manifest["method_script_sha256"], "method"),
        (common_prelabel.sha256(
            HERE / "batterylife_asymmetric_cohort_router.py") ==
         manifest["v1_dependency_sha256"], "v1 dependency"),
        (common_prelabel.sha256(DEV_RESULT) == manifest["development_result_sha256"],
         "development result"),
    ]
    failed = [name for ok, name in checks if not ok]
    if failed:
        raise RuntimeError("Frozen-chain verification failed: " + ", ".join(failed))
    return archive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--label-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    archive = verify(manifest, args.manifest, args.label_file)

    # All hashes and frozen artifacts are verified before this first semantic
    # decoding of label values.
    labels = json.loads(args.label_file.read_text())
    common_eval.predict = v2.predict
    loaded = {h: common_eval.load_cells_archive(archive, labels, h)
              for h in sorted({x["horizon"] for x in manifest["settings"]})}
    results = [common_eval.evaluate_setting(loaded[x["horizon"]], x)
               for x in manifest["settings"]]
    output = {
        "phase": "ONE_SHOT_LABEL_OPENED_EVALUATION_V2",
        "manifest": str(args.manifest),
        "manifest_sha256": common_prelabel.sha256(args.manifest),
        "dataset": manifest["dataset"],
        "label_file_sha256": common_prelabel.sha256(args.label_file),
        "active_settings_frozen_prelabel": manifest["active_settings"],
        "results": results,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({f"h{x['horizon']}_k{x['label_budget_k']}": {
        "mape": [x["baseline"]["mape"], x["method"]["mape"]],
        "relative_reduction_percent": x["relative_mape_reduction_percent"],
        "cells": x["improved_same_worse_cells"], "p": x["paired_wilcoxon_p"],
        "route_counts": x["route_counts"]} for x in results}, indent=2))


if __name__ == "__main__":
    main()
