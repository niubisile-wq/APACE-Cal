"""Exploratory (non-blind) 21700 Expt4 evaluation at a dataset-local 90% EOL.

The prelabel manifest was frozen before performance summaries were opened, but
the summaries were later inspected for the input-contract audit. Consequently
this run is explicitly non-blind and cannot be used as independent confirmation.
It is retained only to determine whether this source merits a future clean
re-download and prelabel freeze.
"""
from __future__ import annotations

import csv
import io
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import batterylife_asymmetric_cohort_router as v1
import batterylife_asymmetric_cohort_router_v2 as v2
import batterylife_zenodo_21700_prelabel as pre
from batterylife_curve_aware_support import robust_scale
from batterylife_transductive_pool_acquisition import distance_matrix


HERE = Path(__file__).parent
MANIFEST = HERE / "batterylife_zenodo_21700_expt4_prelabel.json"
LABEL_AUDIT = HERE / "batterylife_zenodo_21700_postlabel_audit.json"
OUT = HERE / "batterylife_zenodo_21700_exploratory_90eol.json"
SEEDS = 100


def get_life90() -> dict[str, float]:
    size, entries = pre.central_directory(pre.API)
    result = {}
    for cell in "ABCDEFGH":
        name = next(n for n in entries if "Performance Summary" in n and f"cell {cell} (" in n and n.endswith("Processed Data.csv"))
        rows = csv.DictReader(io.StringIO(pre.member_bytes(pre.API, entries[name]).decode("utf-8", "replace")))
        candidates = [
            float(row["Ageing Cycles"])
            for row in rows
            if row.get("SoH") not in (None, "")
            and row.get("Ageing Cycles") not in (None, "")
            and float(row["SoH"]) <= 0.90
        ]
        if not candidates:
            raise RuntimeError(f"No 90% EOL for cell {cell}")
        result[f"21700_expt4_cell_{cell}"] = min(candidates)
    return result


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    life = get_life90()
    v1.predict = v2.predict
    v1.routed_support = v2.routed_support
    v1.predictor_names = v2.predictor_names
    results = []
    for horizon in (10, 20, 50):
        curve_key = "curve_h50_capacity_Ah"
        source_cells = [
            {
                "name": c["name"],
                "life": life[c["name"]],
                "protocol": c["protocol"],
                "curve": c[curve_key][:horizon],
            }
            for c in manifest["cells"]
        ]
        names = [c["name"] for c in source_cells]
        truth = np.asarray([c["life"] for c in source_cells], float)
        protocol = np.asarray([c["protocol"] for c in source_cells], float)
        curve = np.asarray([c["curve"] for c in source_cells], float)
        dp = distance_matrix(protocol, robust_scale(protocol), 1e9)
        dc = distance_matrix(curve, robust_scale(curve), 1e9)
        distances = {w: dc if math.isinf(w) else np.sqrt(dp**2 + w * dc**2) for w in v1.WEIGHTS}
        spread = v1.protocol_dispersion(dp)
        for budget in (1, 3, 5, 10):
            baseline_predictor = "logmean" if budget == 1 else "w2_bw0.5"
            method_predictor = baseline_predictor if budget == 1 else "w2_bw0.5"
            store = {"baseline": defaultdict(list), "method": defaultdict(list)}
            routes = defaultdict(int)
            for seed in range(1, SEEDS + 1):
                rng = np.random.default_rng(60_000_000 * horizon + 10_000 * budget + seed)
                perm = rng.permutation(len(source_cells)); pool_n = min(len(source_cells)-2, max(int(math.ceil(.7*len(source_cells))),budget))
                acquisition, test = np.sort(perm[:pool_n]), np.sort(perm[pool_n:]); k=min(budget,len(acquisition))
                rank_order=rng.permutation(len(source_cells)); tie_rank=np.empty(len(source_cells),int);tie_rank[rank_order]=np.arange(len(source_cells))
                random_support=np.sort(rng.choice(acquisition,size=k,replace=False))
                router_support,route=v1.routed_support(spread,k,random_support,acquisition,np.arange(len(source_cells)),distances,tie_rank);routes[route]+=1
                for arm,predictor,support in (("baseline",baseline_predictor,random_support),("method",method_predictor,router_support)):
                    pred=v2.predict(predictor,distances,truth,support,test)
                    for pos,idx in enumerate(test):
                        e=abs(float(pred[pos])-truth[idx]);store[arm][names[idx]].append((e,100*e/max(truth[idx],1)))
            summary={}
            for arm in ("baseline","method"):
                rows=[]
                for name in names:
                    a=np.asarray(store[arm][name]);rows.append((float(a[:,0].mean()),float(a[:,1].mean())))
                summary[arm]={"mae":float(np.mean([x[0] for x in rows])),"mape":float(np.mean([x[1] for x in rows]))}
            results.append({"horizon":horizon,"label_budget_k":budget,"protocol_dispersion":spread,"route_counts":dict(routes),"summary":summary,"relative_mape_reduction_percent":100*(summary["baseline"]["mape"]-summary["method"]["mape"])/summary["baseline"]["mape"]})
    output={"phase":"EXPLORATORY_NONBLIND_90_PERCENT_EOL","manifest":MANIFEST.name,"label_audit":LABEL_AUDIT.name,"eol_definition":"first performance-summary Ageing Cycles with SoH <= 0.90","results":results,"warning":"Not an independent blind confirmation; 90% EOL is dataset-local and not unified with development labels."}
    OUT.write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps({f"h{r['horizon']}_k{r['label_budget_k']}":r["relative_mape_reduction_percent"] for r in results},indent=2))


if __name__ == "__main__":
    main()
