"""Freeze a pooled BatteryLife external cohort after label-blind screening."""
from __future__ import annotations
import argparse, hashlib, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import batterylife_asymmetric_cohort_router_v2 as v2
import batterylife_apace_stability_gate_candidate as gate
import batterylife_blind_prelabel_manifest as common
import batterylife_batterylife_prelabel as pre

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--datasets',nargs='+',required=True); ap.add_argument('--prelabel',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--seeds',type=int,default=100); a=ap.parse_args()
    cells=[]
    for ds in a.datasets:
        z=a.root/f'{ds}.zip'
        for n in pre.names(z):
            x=pre.load_member(z,n); cycles=sorted(x.get('cycle_data',[]),key=lambda c:c.get('cycle_number',0))
            if len(cycles)<50: continue
            cells.append({'name':f'{ds}::{Path(n).stem}','protocol':pre.protocol(x),'features':{str(h):pre.features(cycles,h) for h in (10,20,50)}})
    dev=json.loads((Path(__file__).parent/'batterylife_asymmetric_cohort_router_v2.json').read_text()); settings=[]; active=[]; freeze={}
    for h in (10,20,50):
        loaded=[{'name':c['name'],'protocol':c['protocol'],'curve':c['features'][str(h)]} for c in cells]
        st=gate.stability(loaded,h,'BatteryLifePooled',(1,3,5,10))[1]
        protocol=np.asarray([c['protocol'] for c in loaded]); curve=np.asarray([c['curve'] for c in loaded]); ps=pre.robust_scale(protocol); cs=pre.robust_scale(curve); dp=pre.distance_matrix(protocol,ps,1e9); dc=pre.distance_matrix(curve,cs,1e9); spread=float(np.median(dp[np.triu_indices(len(loaded),1)])); rho=v2.distance_concordance(loaded); dist={w:(dc if math.isinf(w) else np.sqrt(dp*dp+w*dc*dc)) for w in (0.,.125,.25,.5,1.,2.,math.inf)}; clients=np.arange(len(loaded)); names=[c['name'] for c in loaded]
        for k in (1,3,5,10):
            choice,ranking=common.external_predictor(dev,h,k); freeze[f'h{h}_k{k}']={'selected':choice,'ranking':ranking}; routes=defaultdict(int); episodes=[]
            for seed in range(1,a.seeds+1):
                rng=np.random.default_rng(60000000*h+10000*k+seed); perm=rng.permutation(len(loaded)); an=min(len(loaded)-2,max(int(math.ceil(.7*len(loaded))),k)); ac=np.sort(perm[:an]); test=np.sort(perm[an:]); kk=min(k,len(ac)); sh=rng.permutation(len(loaded)); tie=np.empty(len(loaded),int); tie[sh]=np.arange(len(loaded)); rs=np.sort(rng.choice(ac,size=kk,replace=False))
                if st[k]['active_allowed']: ms,route=v2.routed_support(spread,kk,rs,ac,clients,dist,tie)
                else: ms,route=rs,'fallback_stability_or_coverage'
                pred='support_median' if route.startswith('active_') and spread>=.60 and rho<v2.CONCORDANCE_THRESHOLD else ('w0.5_bw0.5' if route.startswith('active_') and spread>=.60 else ('w2_bw0.5' if route.startswith('active_') else choice['predictor']))
                routes[route]+=1; episodes.append({'seed':seed,'acquisition':[names[i] for i in ac],'test':[names[i] for i in test],'baseline_support':[names[i] for i in rs],'method_support':[names[i] for i in ms],'route':route,'baseline_predictor':choice['predictor'],'method_predictor':pred})
            settings.append({'horizon':h,'label_budget_k':k,'n_unlabeled_cells':len(loaded),'protocol_dispersion':spread,'distance_concordance_spearman':rho,'route_counts':dict(routes),'stability':st[k],'episodes':episodes})
            if any(x.startswith('active_') for x in routes): active.append(f'h{h}_k{k}')
    out={'phase':'BATTERYLIFE_POOLED_PRELABEL_FROZEN','dataset':'BatteryLife pooled HUST+Tongji+XJTU','label_read':False,'datasets':a.datasets,'active_settings':active,'predictor_freeze':freeze,'settings':settings,'prelabel_manifest_sha256':sha(a.prelabel),'method_candidate_sha256':sha(Path(__file__).parent/'batterylife_apace_stability_gate_candidate.py'),'manifest_builder_sha256':sha(__file__),'explicit_non_access_statement':'Only processed pickle metadata and first H cycles were read; Life labels archive was not opened.'}
    a.output.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n'); print(json.dumps({'n_cells':len(cells),'active_settings':active,'routes':{f"h{x['horizon']}_k{x['label_budget_k']}":x['route_counts'] for x in settings}},indent=2))
if __name__=='__main__': main()
