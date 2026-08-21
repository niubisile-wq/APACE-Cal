"""Label-blind v3 manifest for the pristine UConn-ILCC NMC/Gr candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import batterylife_apace_stability_gate_candidate as gate
import batterylife_blind_prelabel_manifest as common
import batterylife_uconn_isu_prelabel as pre
import batterylife_uconn_isu_manifest_v3 as generic


HERE = Path(__file__).parent
DEV = HERE / "batterylife_asymmetric_cohort_router_v2.json"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_params(path):
    import csv
    out = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cid = f"{int(row['Cell ID']):02d}"
            out[cid] = {"group": int(row["Group Number"]), "protocol": pre.np.asarray([
                22.0, float(row["DoD 1"]) * 100.0, 0.0,
                float(row["Chg C-Rate 1"]), float(row["DChg C-Rate 1"])
            ], dtype=float)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/autodl-fs/data/battery_external_uconn_ilcc_nmc"))
    ap.add_argument("--output", type=Path, default=Path("/autodl-fs/data/battery_external_uconn_ilcc_nmc/uconn_nmc_manifest_v3.json"))
    ap.add_argument("--seeds", type=int, default=100)
    args = ap.parse_args()
    root = args.root
    par = read_params(root / "cycling_parameters.csv")
    members = pre.member_map([root / "cycling_data.zip"])
    cache = {cid: pre.read_cell(members[cid], 50) for cid in sorted(par)}
    loaded = {}
    for h in (10, 20, 50):
        loaded[h] = [{"name": f"UConn_NMC_cell_{cid}", "protocol": par[cid]["protocol"],
                      "curve": pre.curve_features(cache[cid], h)} for cid in sorted(par)
                     if pre.curve_features(cache[cid], h) is not None]
    dev = json.loads(DEV.read_text())
    settings, predictor_freeze, active = [], {}, []
    for h in (10, 20, 50):
        stability = gate.stability(loaded[h], h, "UConn_NMC", (1, 3, 5, 10))[1]
        for k in (1, 3, 5, 10):
            choice, ranking = common.external_predictor(dev, h, k)
            predictor_freeze[f"h{h}_k{k}"] = {"selected": choice, "ranking": ranking}
            row = generic.setting(loaded[h], h, k, args.seeds, choice["predictor"], stability[k])
            settings.append(row)
            if any(route.startswith("active_") for route in row["route_counts"]):
                active.append(f"h{h}_k{k}")
    output = {
        "phase": "UCONN_NMC_V3_PRELABEL_FROZEN", "dataset": "UConn-ILCC NMC/Gr",
        "label_read": False, "archive_root": str(root), "active_settings": active,
        "predictor_freeze": predictor_freeze, "settings": settings,
        "method_candidate_sha256": sha256(HERE / "batterylife_apace_stability_gate_candidate.py"),
        "manifest_builder_sha256": sha256(Path(__file__)),
        "prelabel_reader_sha256": sha256(HERE / "batterylife_uconn_isu_prelabel.py"),
        "development_result_sha256": sha256(DEV),
        "explicit_non_access_statement": "Only cycling data and protocol parameters were read; RPT archive was not opened.",
    }
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"cells": {h: len(loaded[h]) for h in loaded}, "active_settings": active}, indent=2))


if __name__ == "__main__":
    main()
