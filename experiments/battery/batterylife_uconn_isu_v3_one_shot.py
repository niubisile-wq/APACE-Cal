"""One-shot evaluator for the frozen UConn-ISU-ILCC v3 manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import batterylife_asymmetric_cohort_router_v2 as v2
import batterylife_blind_manifest_eval as common_eval
import batterylife_uconn_isu_prelabel as pre


HERE = Path(__file__).parent
DEV = HERE / "batterylife_asymmetric_cohort_router_v2.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def derive_labels(root: Path, members: dict[str, list[tuple[Path, str]]]) -> dict[str, float]:
    """Open RPT only after all manifest hashes are verified."""
    labels = {}
    usecols = ["RPT Number", "Life", "Num Cycles"]
    for cid, parts in sorted(members.items()):
        rpt_max = {}
        for archive, member in parts:
            if "rpt_" not in member.lower():
                continue
            proc = subprocess.Popen(["unzip", "-p", str(archive), member],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            assert proc.stdout is not None
            try:
                for chunk in pd.read_csv(proc.stdout, usecols=usecols, chunksize=300_000):
                    life_mask = chunk["Life"].astype(str).str.contains("1st", case=False, na=False)
                    part = chunk.loc[life_mask, ["RPT Number", "Num Cycles"]].copy()
                    part["RPT Number"] = pd.to_numeric(part["RPT Number"], errors="coerce")
                    part["Num Cycles"] = pd.to_numeric(part["Num Cycles"], errors="coerce")
                    for rpt, group in part.dropna().groupby("RPT Number"):
                        value = float(group["Num Cycles"].max())
                        if np.isfinite(value):
                            rpt_max[int(rpt)] = max(rpt_max.get(int(rpt), 0.0), value)
            finally:
                proc.stdout.close()
                stderr = proc.stderr.read() if proc.stderr is not None else b""
                code = proc.wait()
                if code not in (0, -13, 13, 141):
                    raise RuntimeError(f"RPT unzip failed for {member}: {stderr[-500:]!r}")
        life = float(sum(value for rpt, value in rpt_max.items() if rpt > 0))
        if life < 50:
            raise RuntimeError(f"Invalid derived EOL for cell {cid}: {life}")
        labels[f"UConn_ISU_cell_{cid}"] = life
    return labels


def load_early(root: Path, params: dict, members: dict[str, list[tuple[Path, str]]]):
    cache = {cid: pre.read_cell(members[cid], 50) for cid in sorted(params)}
    loaded = {}
    for h in (10, 20, 50):
        loaded[h] = []
        for cid in sorted(params):
            curve = pre.curve_features(cache[cid], h)
            if curve is not None:
                loaded[h].append({"name": f"UConn_ISU_cell_{cid}",
                                  "life": np.nan,
                                  "protocol": params[cid]["protocol"],
                                  "curve": curve})
    return loaded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/autodl-fs/data/battery_external_uconn_isu"))
    ap.add_argument("--manifest", type=Path, default=Path("/autodl-fs/data/battery_external_uconn_isu/uconn_isu_manifest_v3.json"))
    ap.add_argument("--output", type=Path, default=HERE / "batterylife_uconn_isu_v3_one_shot.json")
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("phase") != "UCONN_ISU_V3_PRELABEL_FROZEN" or manifest.get("label_read") is not False:
        raise RuntimeError("invalid or already-opened UConn manifest")
    frozen = {
        "manifest_sha256": sha256(args.manifest),
        "candidate_sha256": sha256(HERE / "batterylife_apace_stability_gate_candidate.py"),
        "builder_sha256": sha256(HERE / "batterylife_uconn_isu_manifest_v3.py"),
        "prelabel_sha256": sha256(HERE / "batterylife_uconn_isu_prelabel.py"),
        "development_result_sha256": sha256(DEV),
    }
    expected = {
        "manifest_sha256": "a8b14771d838a5f9a4e2849853b530965e8ba22558e5b63d823c725b98e7ce95",
        "candidate_sha256": manifest["method_candidate_sha256"],
        "builder_sha256": manifest["manifest_builder_sha256"],
        "prelabel_sha256": manifest["prelabel_reader_sha256"],
        "development_result_sha256": manifest["development_result_sha256"],
    }
    if frozen != expected:
        raise RuntimeError(f"frozen hash mismatch: {frozen} != {expected}")

    params = pre.params(args.root / "cycling_parameters.csv")
    cycling_members = pre.member_map([args.root / "cycling_part1.zip", args.root / "cycling_part2.zip"])
    rpt_members = pre.member_map([args.root / "rpt_data.zip"])
    # Prefix collision is impossible in member_map; retain only RPT entries.
    labels = derive_labels(args.root, rpt_members)
    loaded = load_early(args.root, params, cycling_members)
    for horizon in loaded:
        for cell in loaded[horizon]:
            cell["life"] = labels[cell["name"]]
    settings = manifest["settings"]
    common_eval.predict = v2.predict
    results = [common_eval.evaluate_setting(loaded[s["horizon"]], s) for s in settings]
    output = {
        "phase": "UCONN_ISU_V3_ONE_SHOT_LABEL_OPENED",
        "dataset": "UConn-ISU-ILCC LFP/Gr", "manifest_sha256": frozen["manifest_sha256"],
        "label_definition": "sum of maximum Num Cycles per first-life RPT number; official EOL target is 80% SOH",
        "label_count": len(labels), "active_settings_frozen_prelabel": manifest["active_settings"],
        "frozen_hashes": frozen, "results": results,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({f"h{x['horizon']}_k{x['label_budget_k']}": {
        "mape": [x["baseline"]["mape"], x["method"]["mape"]],
        "relative_reduction_percent": x["relative_mape_reduction_percent"],
        "cells": x["improved_same_worse_cells"], "p": x["paired_wilcoxon_p"],
        "route_counts": x["route_counts"]} for x in results}, indent=2))


if __name__ == "__main__":
    main()
