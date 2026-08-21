"""Strong local calibrator baselines on identical protocol-selected supports."""
import json
from pathlib import Path
import numpy as np
from batterylife_protocol_selection import protocol

SRC=Path(__file__).with_name('batterylife_support_selection_6d.json')
def main():
 x=json.load(open(SRC)); labels={};
 for r in x['rows']:
  labels[(r['dataset'],r['held_out'])]=r['truth']
 names={d:sorted(n for dd,n in labels if dd==d) for d in x['audit']}; meta={}; scale={}
 for d,ns in names.items():
  v=np.asarray([protocol(n) for n in ns]);sd=np.nanstd(v,axis=0);sd[~np.isfinite(sd)|(sd==0)]=1.;meta[d]={n:protocol(n) for n in ns};scale[d]=sd
 rows=[]
 for r in x['rows']:
  if r['selector']!='full_protocol':continue
  d=r['dataset'];test=meta[d][r['held_out']];sup=r['selected'];ys=np.asarray([labels[(d,n)] for n in sup]);vs=np.asarray([meta[d][n] for n in sup]);sd=scale[d]
  dist=[]
  for v in vs:
   ok=np.isfinite(test)&np.isfinite(v);dist.append(float(np.sqrt(np.sum(((test[ok]-v[ok])/sd[ok])**2))) if ok.any() else 0.)
  dist=np.asarray(dist); zero=dist<1e-12
  iw=zero.astype(float) if zero.any() else 1/np.maximum(dist,1e-6);rw=np.exp(-.5*dist**2)
  # closed-form local ridge in standardized metadata, with missing indicators
  def phi(v):return np.r_[np.nan_to_num(v/sd,nan=0.),(~np.isfinite(v)).astype(float)]
  Z=np.asarray([phi(v) for v in vs]);zt=phi(test);A=np.c_[np.ones(len(Z)),Z];pen=np.eye(A.shape[1]);pen[0,0]=0.;coef=np.linalg.solve(A.T@A+10.*pen,A.T@np.log(ys));ridge=float(np.exp(np.r_[1.,zt]@coef))
  preds={'median':float(np.median(ys)),'mean':float(np.mean(ys)),'inverse_distance':float(iw@ys/iw.sum()),'rbf':float(rw@ys/rw.sum()),'local_log_ridge':ridge}
  for method,p in preds.items():
   e=abs(p-r['truth']);rows.append({'dataset':d,'held_out':r['held_out'],'k':r['k'],'seed':r['seed'],'method':method,'prediction':p,'truth':r['truth'],'abs_error':e,'ape':100*e/max(r['truth'],1)})
 out={'source':SRC.name,'protocol':'identical full-protocol-selected supports; only local calibrator differs','rows':rows,'summary':{}}
 for k in [1,3,5,10]:
  for method in ['median','mean','inverse_distance','rbf','local_log_ridge']:
   ds=[]
   for d in names:
    q=[r for r in rows if (r['dataset'],r['k'],r['method'])==(d,k,method)];rec={'mae':float(np.mean([r['abs_error'] for r in q])),'mape':float(np.mean([r['ape'] for r in q]))};out['summary'][f'{d}_k{k}_{method}']=rec;ds.append(rec)
   out['summary'][f'macro_k{k}_{method}']={'mae':float(np.mean([z['mae'] for z in ds])),'mape':float(np.mean([z['mape'] for z in ds]))}
 p=Path(__file__).with_name('batterylife_local_calibrator_baselines.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['summary'],indent=2))
if __name__=='__main__':main()
