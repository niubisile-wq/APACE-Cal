"""Clean ablation of metadata used for three-cell calibration selection."""
import json,random
from pathlib import Path
import numpy as np
from batterylife_protocol_selection import protocol

BASE=Path(__file__).with_name('batterylife_clean_horizon.json')
MASKS={'random':(), 'temperature':(0,), 'soc_crate':(1,2,3,4), 'full_protocol':(0,1,2,3,4)}

def main():
 x=json.load(open(BASE)); unique={}
 for r in x['rows']:
  if r['method']=='raw': unique[(r['horizon'],r['source'],r['target'],r['model'],r['held_out'])]=(r['prediction'],r['truth'])
 groups={}
 for (h,s,t,m,n),v in unique.items(): groups.setdefault((h,s,t,m),[]).append((n,*v))
 rows=[]
 for (h,s,t,m),cells in groups.items():
  vec=np.asarray([protocol(n) for n,_,_ in cells]); sd=np.nanstd(vec,axis=0); sd[~np.isfinite(sd)|(sd==0)]=1.
  for i,(name,p,y) in enumerate(cells):
   for seed in range(1,11):
    for selector,dims in MASKS.items():
     av=[j for j in range(len(cells)) if j!=i]; random.Random(100000*i+100*seed+len(dims)).shuffle(av)
     def d(j):
      if not dims:return 0.
      ix=np.asarray(dims); ok=np.isfinite(vec[i,ix])&np.isfinite(vec[j,ix]);
      return float(np.sqrt(np.sum(((vec[i,ix][ok]-vec[j,ix][ok])/sd[ix][ok])**2))) if ok.any() else 0.
     sel=sorted(av,key=d)[:3]; pc=np.asarray([cells[j][1] for j in sel]); yc=np.asarray([cells[j][2] for j in sel]); pred=p*float(np.median(yc/np.maximum(pc,1e-6))); e=abs(pred-y)
     rows.append({'horizon':h,'source':s,'target':t,'model':m,'held_out':name,'seed':seed,'selector':selector,'selected':[cells[j][0] for j in sel],'abs_error':float(e),'ape':float(100*e/max(y,1))})
 out={'protocol':'same cell-level predictions; median-scale calibrator fixed; only label-free selection metadata ablated','rows':rows,'summary':{}}
 for h,s,t,m in sorted(groups):
  for selector in MASKS:
   z=[r for r in rows if (r['horizon'],r['source'],r['target'],r['model'],r['selector'])==(h,s,t,m,selector)]
   out['summary'][f'h{h}_{s}_to_{t}_{m}_{selector}']={'mae':float(np.mean([r['abs_error'] for r in z])),'mape':float(np.mean([r['ape'] for r in z])),'n_evaluations':len(z)}
 p=Path(__file__).with_name('batterylife_protocol_ablation.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['summary'],indent=2))
if __name__=='__main__':main()
