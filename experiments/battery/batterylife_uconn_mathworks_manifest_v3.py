"""Freeze MathWorks episode identities after label-blind prelabel screening."""
from __future__ import annotations
import argparse, hashlib, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import batterylife_asymmetric_cohort_router_v2 as v2
import batterylife_apace_stability_gate_candidate as gate
import batterylife_blind_prelabel_manifest as common
import batterylife_uconn_mathworks_prelabel as pre

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--prelabel',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--seeds',type=int,default=100); a=ap.parse_args()
    z=sorted(a.root.glob('batch*.zip')); mm=pre.members(z); rows=[]; cache={}
    for name in sorted(mm):
        sig=pre.read_cell(mm[name],50); cache[name]=sig; f=pre.curve_features(sig,50)
        if f is None: continue
        g=int(name.split('G')[1].split('C')[0]); c=int(name.split('C')[1]); first=np.asarray(sig[0],float)
        rows.append({'name':f'MathWorks_{name}','protocol':np.asarray([25.,g,c,first[4],first[0],first[1]],float),'sig':sig})
    dev=json.loads((Path(__file__).parent/'batterylife_asymmetric_cohort_router_v2.json').read_text()); settings=[]; active=[]; freeze={}
    for h in (10,20,50):
        cells=[{'name':r['name'],'protocol':r['protocol'],'curve':pre.curve_features(r['sig'],h)} for r in rows]
        st=gate.stability(cells,h,'UConn_MathWorks',(1,3,5,10))[1]
        for k in (1,3,5,10):
            choice,ranking=common.external_predictor(dev,h,k); freeze[f'h{h}_k{k}']={'selected':choice,'ranking':ranking}
            protocol=np.asarray([x['protocol'] for x in cells]); curve=np.asarray([x['curve'] for x in cells]); ps=pre.robust_scale(protocol); cs=pre.robust_scale(curve); dp=pre.distance_matrix(protocol,ps,1e9); dc=pre.distance_matrix(curve,cs,1e9); spread=float(np.median(dp[np.triu_indices(len(cells),1)])); rho=v2.distance_concordance(cells); dist={w:(dc if math.isinf(w) else np.sqrt(dp*dp+w*dc*dc)) for w in (0.,.125,.25,.5,1.,2.,math.inf)}; clients=np.arange(len(cells)); names=[x['name'] for x in cells]; routes=defaultdict(int); episodes=[]
            for seed in range(1,a.seeds+1):
                rng=np.random.default_rng(60000000*h+10000*k+seed); perm=rng.permutation(len(cells)); an=min(len(cells)-2,max(int(math.ceil(.7*len(cells))),k)); ac=np.sort(perm[:an]); test=np.sort(perm[an:]); kk=min(k,len(ac)); sh=rng.permutation(len(cells)); tie=np.empty(len(cells),int); tie[sh]=np.arange(len(cells)); rs=np.sort(rng.choice(ac,size=kk,replace=False));
                if st[k]['active_allowed']: ms,route=v2.routed_support(spread,kk,rs,ac,clients,dist,tie)
                else: ms,route=rs,'fallback_stability_or_coverage'
                pred='support_median' if route.startswith('active_') and spread>=.60 and rho<v2.CONCORDANCE_THRESHOLD else ('w0.5_bw0.5' if route.startswith('active_') and spread>=.60 else ('w2_bw0.5' if route.startswith('active_') else choice['predictor']))
                routes[route]+=1; episodes.append({'seed':seed,'acquisition':[names[i] for i in ac],'test':[names[i] for i in test],'baseline_support':[names[i] for i in rs],'method_support':[names[i] for i in ms],'route':route,'baseline_predictor':choice['predictor'],'method_predictor':pred})
            row={'horizon':h,'label_budget_k':k,'n_unlabeled_cells':len(cells),'protocol_dispersion':spread,'distance_concordance_spearman':rho,'route_counts':dict(routes),'stability':st[k],'episodes':episodes}; settings.append(row)
            if any(x.startswith('active_') for x in routes): active.append(f'h{h}_k{k}')
    out={'phase':'UCONN_MATHWORKS_V3_PRELABEL_FROZEN','dataset':'UConn-MathWorks LFP/Gr','label_read':False,'active_settings':active,'predictor_freeze':freeze,'settings':settings,'prelabel_manifest_sha256':sha(a.prelabel),'method_candidate_sha256':sha(Path(__file__).parent/'batterylife_apace_stability_gate_candidate.py'),'manifest_builder_sha256':sha(__file__),'explicit_non_access_statement':'Only first-life cycling members were read; RPT_note.csv and RPT-like members were not opened.'}
    a.output.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n'); print(json.dumps({'active_settings':active,'n_cells':len(rows)},indent=2))
if __name__=='__main__': main()
