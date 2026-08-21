"""Descriptive H50/K3 stratification by lab/domain, temperature, chemistry tags."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SRC = HERE / "batterylife_asymmetric_cohort_router_v2.json"
OUT = HERE / "batterylife_apace_stratified_audit.json"


def add(group, row):
    group[row["held_out"]].append(100.0 * (row["baseline_ape"] - row["method_ape"]) / max(row["baseline_ape"], 1e-12))


def main():
    groups = defaultdict(list)
    for r in json.loads(SRC.read_text())["results"]:
        if r["horizon"] != 50 or r["label_budget_k"] != 3:
            continue
        for c in r["per_cell"]:
            name = c["held_out"]
            add(groups, {**c, "held_out": f"{r['target']}::{name}"})
            groups[f"domain:{r['target']}"].append(100.0 * (c["baseline_ape"]-c["method_ape"]) / max(c["baseline_ape"],1e-12))
            temp = re.search(r"(\d{1,2})C", name)
            groups[f"temperature:{temp.group(1)}C" if temp else "temperature:unknown"].append(100.0 * (c["baseline_ape"]-c["method_ape"]) / max(c["baseline_ape"],1e-12))
            chem = next((x for x in ("LFP","NMC","NCA","LMO") if x in name), "unknown")
            groups[f"chemistry:{chem}"].append(100.0 * (c["baseline_ape"]-c["method_ape"]) / max(c["baseline_ape"],1e-12))
    output = {"protocol":"descriptive stratification of frozen H50/K3 cell-level audit; no post-hoc method tuning", "groups": {k:{"n":len(v),"mean_relative_reduction_percent":float(np.mean(v)),"median_relative_reduction_percent":float(np.median(v)),"positive_fraction":float(np.mean(np.asarray(v)>0))} for k,v in sorted(groups.items()) if v}}
    OUT.write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps(output,indent=2))


if __name__=='__main__': main()
