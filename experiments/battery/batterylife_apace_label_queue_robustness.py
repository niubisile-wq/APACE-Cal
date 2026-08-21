"""E6 support-label noise, acquisition-fraction, and small-queue audit."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import batterylife_asymmetric_cohort_router_v2 as v2
from batterylife_curve_aware_support import DATASETS, load_cells, robust_scale
from batterylife_transductive_pool_acquisition import WEIGHTS, distance_matrix, select_facilities

HERE = Path(__file__).parent
OUT = HERE / "batterylife_apace_label_queue_robustness.json"


def eval_condition(cells, target, condition, perturb_seed, episodes=100):
    n = len(cells); truth = np.asarray([c["life"] for c in cells], float)
    p = np.asarray([c["protocol"] for c in cells], float); q = np.asarray([c["curve"] for c in cells], float)
    dp = distance_matrix(p, robust_scale(p), 1e9); dc = distance_matrix(q, robust_scale(q), 1e9)
    distances = {w: dc if math.isinf(w) else np.sqrt(dp*dp+w*dc*dc) for w in WEIGHTS}
    spread = float(np.median(dp[np.triu_indices(n,1)])) if n > 1 else 0.
    rho = v2.distance_concordance(cells)
    bpred = next(r["selected_matched_predictor"]["predictor"] for r in json.loads((HERE/"batterylife_asymmetric_cohort_router_v2.json").read_text())["results"] if r["target"]==target and r["horizon"]==50 and r["label_budget_k"]==3)
    bvals={c["name"]:[] for c in cells}; mvals={c["name"]:[] for c in cells}
    for ep in range(episodes):
        seed = 90_000_000 + perturb_seed*10_000 + ep
        rng=np.random.default_rng(seed)
        pool=np.arange(n)
        frac=condition.get("acquisition_fraction",.70)
        if "queue_n" in condition:
            keep=np.sort(rng.choice(pool,size=min(condition["queue_n"],n),replace=False)); cells2=[cells[i] for i in keep]
            # Recurse only for the selected queue with a stable one-episode path.
            # Rebuilding matrices is intentional: queue size is the factor under test.
            return eval_condition(cells2,target,{k:v for k,v in condition.items() if k!="queue_n"},perturb_seed,episodes)
        perm=rng.permutation(n); acq_n=min(n-2,max(int(math.ceil(frac*n)),3)); acq=np.sort(perm[:acq_n]); test=np.sort(perm[acq_n:]); shuffled=rng.permutation(n); tie=np.empty(n,int); tie[shuffled]=np.arange(n); random=np.sort(rng.choice(acq,size=3,replace=False))
        if spread<=1e-12 or .30<=spread<.60:
            support=random
        elif spread>=.60 and 3>=5:
            support=random
        else:
            aw=2.0 if spread<.30 else .5; support=select_facilities(distances[aw],acq,np.arange(n),3,tie)
        local_truth=truth.copy()
        sigma=condition.get("label_noise",0.)
        if sigma:
            local_truth[support] *= 1+rng.normal(0,sigma,size=len(support))
        bp=v2.predict(bpred,distances,local_truth,random,test)
        if spread<=1e-12 or .30<=spread<.60 or spread>=.60 and 3>=5:
            mp=bp
        elif spread>=.60 and rho<.35:
            mp=v2.predict("support_median",distances,local_truth,support,test)
        elif spread>=.60:
            mp=v2.predict("w0.5_bw0.5",distances,local_truth,support,test)
        else:
            mp=v2.predict("w2_bw0.5",distances,local_truth,support,test)
        for j,i in enumerate(test):
            name=cells[i]["name"]; denom=max(truth[i],1.); bvals[name].append(100*abs(bp[j]-truth[i])/denom); mvals[name].append(100*abs(mp[j]-truth[i])/denom)
    b=float(np.mean([np.mean(x) for x in bvals.values() if x])); m=float(np.mean([np.mean(x) for x in mvals.values() if x]))
    return {"target":target,"condition":condition,"baseline_mape":b,"method_mape":m,"relative_reduction_percent":100*(b-m)/max(b,1e-12)}


def main():
    conditions=[{"name":f"label_noise_{x:g}","label_noise":x} for x in (.02,.05,.10)] + [{"name":f"acquisition_{int(x*100)}pct","acquisition_fraction":x} for x in (.50,.60,.80)] + [{"name":f"queue_{x}","queue_n":x} for x in (15,30,60)]
    output={"protocol":"H50/K3; 30 perturbation seeds x 100 episodes; life truth held out for scoring","rows":[]}
    loaded={d:load_cells(d,50) for d in DATASETS}
    for c in conditions:
        print(c["name"],flush=True)
        vals=[]
        for s in range(30):
            for d in DATASETS: vals.append(eval_condition(loaded[d],d,c,s))
        output["rows"].append({"condition":c,"macro_baseline_mape":float(np.mean([x["baseline_mape"] for x in vals])),"macro_method_mape":float(np.mean([x["method_mape"] for x in vals])),"macro_relative_reduction_percent":float(np.mean([x["relative_reduction_percent"] for x in vals])),"per_target":vals})
    OUT.write_text(json.dumps(output,indent=2)+"\n")


if __name__=='__main__': main()
