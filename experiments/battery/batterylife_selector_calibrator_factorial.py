"""Factorial audit: selector x local calibrator under identical label budgets."""
import json,collections
from pathlib import Path
import numpy as np
from batterylife_protocol_selection import protocol

SRC=Path(__file__).with_name('batterylife_support_selection_6d.json');SELECTORS=('random','temperature','soc_crate','full_protocol')
def ridge(q,V,y,sd):
 def phi(v):return np.r_[np.nan_to_num(v/sd,nan=0.),(~np.isfinite(v)).astype(float)]
 Z=np.asarray([phi(v) for v in V]);A=np.c_[np.ones(len(Z)),Z];pen=np.eye(A.shape[1]);pen[0,0]=0.;coef=np.linalg.solve(A.T@A+10*pen,A.T@np.log(y));return float(np.exp(np.r_[1.,phi(q)]@coef))
def main():
 x=json.load(open(SRC));labels={(r['dataset'],r['held_out']):r['truth'] for r in x['rows']};names={d:sorted(n for dd,n in labels if dd==d) for d in x['audit']};meta={};scale={}
 for d,ns in names.items():
  V=np.asarray([protocol(n) for n in ns]);sd=np.nanstd(V,axis=0);sd[~np.isfinite(sd)|(sd==0)]=1.;meta[d]={n:protocol(n) for n in ns};scale[d]=sd
 g=collections.defaultdict(list);cell=collections.defaultdict(list)
 for r in x['rows']:
  d=r['dataset'];V=np.asarray([meta[d][n] for n in r['selected']]);y=np.asarray([labels[(d,n)] for n in r['selected']]);pred=ridge(meta[d][r['held_out']],V,y,scale[d]);e=abs(pred-r['truth']);rec=(e,100*e/max(r['truth'],1));g[(d,r['k'],r['selector'])].append(rec);cell[(d,r['held_out'],r['k'],r['selector'])].append(rec)
 out={'source':SRC.name,'protocol':'same local-log-ridge calibrator for every selector; common random tie-breaking','summary':{},'per_cell':[]}
 for k in [1,3,5,10]:
  for s in SELECTORS:
   ds=[]
   for d in names:
    q=g[(d,k,s)];rec={'mae':float(np.mean([z[0] for z in q])),'mape':float(np.mean([z[1] for z in q]))};out['summary'][f'{d}_k{k}_{s}']=rec;ds.append(rec)
   out['summary'][f'macro_k{k}_{s}']={'mae':float(np.mean([z['mae'] for z in ds])),'mape':float(np.mean([z['mape'] for z in ds]))}
 for (d,n,k,s),q in cell.items():out['per_cell'].append({'dataset':d,'held_out':n,'k':k,'selector':s,'mae':float(np.mean([z[0] for z in q])),'mape':float(np.mean([z[1] for z in q]))})
 p=Path(__file__).with_name('batterylife_selector_calibrator_factorial.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['summary'],indent=2))
if __name__=='__main__':main()
