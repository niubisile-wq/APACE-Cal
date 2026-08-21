"""Fast exact-seed E4/E5 runner using only the selected predictor per cell.

It avoids evaluating the unused 40-predictor panel for every variant while
preserving the frozen target split, acquisition RNG, distance construction,
and prediction formulas.
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np

import batterylife_asymmetric_cohort_router as v1
import batterylife_asymmetric_cohort_router_v2 as v2
from batterylife_curve_aware_support import DATASETS, load_cells, robust_scale
from batterylife_transductive_pool_acquisition import WEIGHTS, distance_matrix, key_number

HERE = Path(__file__).parent
FROZEN = json.loads((HERE / "batterylife_asymmetric_cohort_router_v2.json").read_text())
HORIZONS, BUDGETS = (10, 20, 50), (1, 3, 5, 10)


def standard_scale(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        s = np.nanstd(np.asarray(x, dtype=float), axis=0)
    return np.where(np.isfinite(s) & (s > 1e-12), s, 1.0)


def specs():
    out = [
        {"name":"A3_fixed_concat","family":"E4","fixed_weight":1.0,"disable_fallback":True},
        {"name":"A7_all_median","family":"E4","all_median":True},
        {"name":"A8_mismatched_evidence","family":"E4","acq_weight":0.0,"pred_weight":2.0},
        {"name":"A9_fixed_predictor","family":"E4","fixed_predictor":"logmean"},
        {"name":"A10_no_robust_scaling","family":"E4","no_robust":True},
    ]
    out += [{"name":f"E5_low_{x:g}","family":"E5","low":x} for x in (.20,.25,.30,.35,.40)]
    out += [{"name":f"E5_high_{x:g}","family":"E5","high":x} for x in (.50,.55,.60,.65,.70)]
    out += [{"name":f"E5_rho_{x:g}","family":"E5","rho":x} for x in (.20,.25,.30,.35,.40,.45,.50)]
    out += [{"name":f"E5_bw_{x:g}","family":"E5","bw":x} for x in (.25,.5,1.,2.)]
    out += [{"name":f"E5_weight_{('inf' if math.isinf(x) else f'{x:g}')}","family":"E5","pred_weight":x} for x in (0.,.125,.5,1.,2.,math.inf)]
    return out


def frozen_baseline(target, h, k):
    for row in FROZEN["results"]:
        if row["target"] == target and row["horizon"] == h and row["label_budget_k"] == k:
            return row["selected_matched_predictor"]["predictor"]
    raise KeyError((target,h,k))


def one_domain(cells, target, h, k, seed_count, spec):
    names = [c["name"] for c in cells]; n = len(cells)
    truth = np.asarray([c["life"] for c in cells], dtype=float)
    protocol = np.asarray([c["protocol"] for c in cells], dtype=float)
    curve = np.asarray([c["curve"] for c in cells], dtype=float)
    scale_fn = standard_scale if spec.get("no_robust") else robust_scale
    dp = distance_matrix(protocol, scale_fn(protocol), 1e9)
    dc = distance_matrix(curve, scale_fn(curve), 1e9)
    distances = {w: dc if math.isinf(w) else np.sqrt(dp*dp + w*dc*dc) for w in WEIGHTS}
    spread = float(np.median(dp[np.triu_indices(n,1)])) if n > 1 else 0.
    rho = v2.distance_concordance(cells)
    low, high = float(spec.get("low",.30)), float(spec.get("high",.60))
    acq_weight = spec.get("acq_weight")
    method_preds = []; base_preds = []; route_counts = {}
    bpred = spec.get("fixed_predictor") or frozen_baseline(target,h,k)
    for seed in range(1, seed_count+1):
        rng = np.random.default_rng(60_000_000*h + 10_000*k + seed)
        permutation = rng.permutation(n)
        acq_n = min(n-2, max(int(np.ceil(.7*n)), k))
        acquisition = np.sort(permutation[:acq_n]); test = np.sort(permutation[acq_n:])
        kk = min(k, len(acquisition)); shuffled = rng.permutation(n); tie = np.empty(n,dtype=int); tie[shuffled]=np.arange(n)
        random_support = np.sort(rng.choice(acquisition,size=kk,replace=False))
        if spec.get("disable_fallback"):
            route = f"active_w{key_number(spec['fixed_weight'])}"
            support = v1.select_facilities(distances[spec['fixed_weight']], acquisition, np.arange(n), kk, tie)
        elif spread <= 1e-12:
            route, support = "fallback_zero_protocol_dispersion", random_support
        elif low <= spread < high:
            route, support = "fallback_medium_protocol_dispersion", random_support
        elif kk == 1:
            route, support = "fallback_one_label_unidentifiable", random_support
        elif kk >= 5 and spread >= high:
            route, support = "fallback_large_budget_high_protocol_dispersion", random_support
        else:
            aw = acq_weight if acq_weight is not None else (2. if spread < low else .5)
            route = f"active_w{key_number(aw)}"
            support = v1.select_facilities(distances[aw], acquisition, np.arange(n), kk, tie)
        route_counts[route] = route_counts.get(route,0)+1
        base_preds.append(v2.predict(bpred, distances, truth, random_support, test))
        if not route.startswith("active_"):
            mpred = bpred
        elif spec.get("all_median"):
            mpred = "support_median"
        elif spec.get("pred_weight") is not None:
            w=spec["pred_weight"]; wt="inf" if math.isinf(w) else key_number(w); mpred=f"w{wt}_bw{key_number(spec.get('bw',.5))}"
        elif spec.get("acq_weight") is not None:
            mpred = f"w{key_number(spec['pred_weight'])}_bw{key_number(spec.get('bw',.5))}" if spec.get('pred_weight') is not None else "w2_bw0.5"
        elif spread >= high and rho < float(spec.get("rho",.35)):
            mpred = "support_median"
        elif spread >= high:
            mpred = f"w0.5_bw{key_number(spec.get('bw',.5))}"
        else:
            mpred = f"w2_bw{key_number(spec.get('bw',.5))}"
        method_preds.append(v2.predict(mpred, distances, truth, support, test))
    # Re-run aggregation deterministically to preserve one row per test cell.
    # Store per-cell episode errors directly using the same RNG sequence.
    bsum={name:[] for name in names}; msum={name:[] for name in names}
    # The compact path above only retained predictions; repeat split generation
    # (cheap compared with feature loading) to attach cell identities.
    for seed in range(1, seed_count+1):
        rng=np.random.default_rng(60_000_000*h+10_000*k+seed); perm=rng.permutation(n); acq_n=min(n-2,max(int(np.ceil(.7*n)),k)); acq=np.sort(perm[:acq_n]); test=np.sort(perm[acq_n:]); kk=min(k,len(acq)); rng.permutation(n); random=np.sort(rng.choice(acq,size=kk,replace=False))
        # use prediction arrays from stored sequence; align with sorted test
        bp=base_preds[seed-1]; mp=method_preds[seed-1]
        for j,idx in enumerate(test):
            bsum[names[idx]].append(100*abs(bp[j]-truth[idx])/max(truth[idx],1)); msum[names[idx]].append(100*abs(mp[j]-truth[idx])/max(truth[idx],1))
    bmap={name:float(np.mean(v)) for name,v in bsum.items()}; mmap={name:float(np.mean(v)) for name,v in msum.items()}
    b=float(np.mean(list(bmap.values()))); m=float(np.mean(list(mmap.values())))
    return {"target":target,"horizon":h,"K":k,"protocol_dispersion":spread,"rho":rho,"baseline_predictor":bpred,"baseline_mape":b,"method_mape":m,"relative_reduction_percent":100*(b-m)/max(b,1e-12),"route_counts":route_counts}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds',type=int,default=100); ap.add_argument('--family',choices=('E4','E5','all'),default='all'); ap.add_argument('--primary-only',action='store_true'); ap.add_argument('--output',type=Path,default=HERE/'batterylife_apace_ablation_sensitivity_fast.json'); args=ap.parse_args()
    loaded={(h,d):load_cells(d,h) for h in HORIZONS for d in DATASETS}; allout={"protocol":"fast exact-seed E4/E5; baseline predictors frozen from v2 development selection","variants":{}}
    ss=specs()
    ss=[s for s in ss if args.family=='all' or s['family']==args.family]
    hs=(50,) if args.primary_only else HORIZONS; ks=(3,) if args.primary_only else BUDGETS
    # The primary-only E5 run is intentionally H50/K3; E5 is a development
    # hyperparameter stability audit, not a new external endpoint.
    for i,s in enumerate(ss,1):
        print(f"[{i}/{len(ss)}] {s['name']}",flush=True); rows=[]
        for h in hs:
            for d in DATASETS:
                for k in ks: rows.append(one_domain(loaded[(h,d)],d,h,k,args.seeds,s))
        macro={}
        for h in hs:
            for k in ks:
                sub=[r for r in rows if r['horizon']==h and r['K']==k]; rel=[r['relative_reduction_percent'] for r in sub]
                macro[f'h{h}_k{k}']={"baseline_mape":float(np.mean([r['baseline_mape'] for r in sub])),"method_mape":float(np.mean([r['method_mape'] for r in sub])),"improved_same_worse_domains":[sum(x>1e-12 for x in rel),sum(abs(x)<=1e-12 for x in rel),sum(x< -1e-12 for x in rel)],"worst_relative_change_percent":float(min(rel))}
        allout['variants'][s['name']]={"spec":s,"macro":macro,"rows":rows}
    args.output.write_text(json.dumps(allout,indent=2)+'\n')


if __name__=='__main__': main()
