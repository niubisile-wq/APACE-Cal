"""Monte-Carlo test-aware oracle search; an unattainable upper-bound diagnostic."""
from __future__ import annotations
import argparse, json, math, zipfile
from pathlib import Path
import numpy as np
from batterylife_asymmetric_cohort_router_v2 import predict
from batterylife_curve_aware_support import robust_scale
from batterylife_transductive_pool_acquisition import distance_matrix
import batterylife_batterylife_prelabel as pre

def labels(root):
 out={}
 with zipfile.ZipFile(root/'Life labels.zip') as z:
  for ds in ('HUST','XJTU'):
   out.update({f'{ds}::{Path(k).stem}':float(v) for k,v in json.loads(z.read(f'Life labels/{ds}_labels.json')).items()})
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();m=json.loads(a.manifest.read_text());labs=labels(a.root);loaded={h:[] for h in (10,20,50)}
 for ds in ('HUST','XJTU'):
  z=a.root/f'{ds}.zip'
  for n in pre.names(z):
   key=f'{ds}::{Path(n).stem}';x=pre.load_member(z,n);cyc=sorted(x['cycle_data'],key=lambda c:c.get('cycle_number',0))
   for h in (10,20,50):
    f=pre.features(cyc,h)
    if f is not None: loaded[h].append({'name':key,'life':labs[key],'protocol':pre.protocol(x),'curve':f})
 rows=[]
 for s in m['settings']:
  cells=loaded[s['horizon']]; idx={c['name']:i for i,c in enumerate(cells)};truth=np.array([c['life'] for c in cells]);p=np.array([c['protocol'] for c in cells]);q=np.array([c['curve'] for c in cells]);dp=distance_matrix(p,robust_scale(p),1e9);dc=distance_matrix(q,robust_scale(q),1e9);w=0.5;dist=np.sqrt(dp*dp+w*dc*dc);base=[];oracle=[]
  for e in s['episodes']:
   test=np.array([idx[n] for n in e['test']]);ac=np.array([idx[n] for n in e['acquisition']]);k=len(e['baseline_support']);bp=predict(e['baseline_predictor'],{w:(dc if math.isinf(w) else np.sqrt(dp*dp+w*dc*dc)) for w in (0,.125,.25,.5,1,2,math.inf)},truth,np.array([idx[n] for n in e['baseline_support']]),test);base.extend(100*np.abs(bp-truth[test])/np.maximum(truth[test],1.));best=float(np.mean(100*np.abs(bp-truth[test])/np.maximum(truth[test],1.)));rng=np.random.default_rng(90000000*s['horizon']+1000*s['label_budget_k']+e['seed']);
   for _ in range(500):
    sup=rng.choice(ac,size=k,replace=False);pr=predict(e['baseline_predictor'],{z:(dc if math.isinf(z) else np.sqrt(dp*dp+z*dc*dc)) for z in (0,.125,.25,.5,1,2,math.inf)},truth,sup,test);best=min(best,float(np.mean(100*np.abs(pr-truth[test])/np.maximum(truth[test],1.))))
   oracle.extend([best]*len(test))
  rows.append({'horizon':s['horizon'],'label_budget_k':s['label_budget_k'],'baseline_mape':float(np.mean(base)),'oracle_search_mape':float(np.mean(oracle)),'relative_reduction_percent':float(100*(np.mean(base)-np.mean(oracle))/max(np.mean(base),1e-12)),'search_candidates_per_episode':500})
 a.output.write_text(json.dumps({'phase':'TEST_AWARE_ORACLE_SEARCH','warning':'Uses held-out test labels and random search; upper-bound diagnostic only.','rows':rows},indent=2)+'\n');print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
