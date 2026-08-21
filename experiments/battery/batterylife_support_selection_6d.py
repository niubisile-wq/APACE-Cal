"""Six-dataset, label-budgeted protocol-aware support selection benchmark."""
import json,random
from collections import defaultdict
from pathlib import Path
import numpy as np
from batterylife_protocol_selection import protocol

ROOT=Path(__file__).resolve().parents[2];DS=('CALB','HNEI','MICH_EXP','CALCE','MICH','SNL');KS=(1,3,5,10);SEEDS=range(1,101)
MASKS={'random':(), 'temperature':(0,), 'soc_crate':(1,2,3,4), 'full_protocol':(0,1,2,3,4)}
def cells(ds):
 x=json.load(open(ROOT/'data/batterylife_processed/Life labels'/f'{ds}_labels.json'));return [(n,float(y)) for n,y in x.items() if y>=50]
def main():
 rows=[];audit={}
 for ds in DS:
  c=cells(ds);audit[ds]=len(c);v=np.asarray([protocol(n) for n,_ in c]);sd=np.nanstd(v,axis=0);sd[~np.isfinite(sd)|(sd==0)]=1.
  for i,(name,y) in enumerate(c):
   for k in KS:
    for seed in SEEDS:
     for method,dims in MASKS.items():
      # Common random numbers across selectors: when metadata cannot
      # distinguish candidates, every selector must return the same support.
      av=[j for j in range(len(c)) if j!=i];random.Random(100000*i+1000*k+seed).shuffle(av)
      def dist(j):
       if not dims:return 0.
       ix=np.asarray(dims);ok=np.isfinite(v[i,ix])&np.isfinite(v[j,ix]);return float(np.sqrt(np.sum(((v[i,ix][ok]-v[j,ix][ok])/sd[ix][ok])**2))) if ok.any() else 0.
      sel=sorted(av,key=dist)[:min(k,len(av))];pred=float(np.median([c[j][1] for j in sel]));e=abs(pred-y);rows.append({'dataset':ds,'held_out':name,'k':k,'seed':seed,'selector':method,'selected':[c[j][0] for j in sel],'prediction':pred,'truth':y,'abs_error':e,'ape':100*e/max(y,1)})
 out={'dataset_versions':'BatteryLife v12','protocol':'leave-one-cell-out; choose K target calibration cells using metadata only; predict held-out life by robust support median; 100 tie/random seeds','audit':audit,'rows':rows,'summary':{}}
 grouped=defaultdict(list)
 for r in rows:grouped[(r['dataset'],r['k'],r['selector'])].append((r['abs_error'],r['ape']))
 for ds in DS:
  for k in KS:
   for method in MASKS:
    q=grouped[(ds,k,method)];out['summary'][f'{ds}_k{k}_{method}']={'mae':float(np.mean([z[0] for z in q])),'mape':float(np.mean([z[1] for z in q])),'n_evaluations':len(q)}
 for k in KS:
  for method in MASKS:
   q=[out['summary'][f'{d}_k{k}_{method}'] for d in DS];out['summary'][f'macro_k{k}_{method}']={'mae':float(np.mean([z['mae'] for z in q])),'mape':float(np.mean([z['mape'] for z in q])),'n_datasets':len(q)}
 p=Path(__file__).with_name('batterylife_support_selection_6d.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'audit':audit,'summary':out['summary']},indent=2))
if __name__=='__main__':main()
