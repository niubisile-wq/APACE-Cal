"""Support-only LOOCV selection among robust local calibrators."""
import json
from pathlib import Path
import numpy as np
from batterylife_protocol_selection import protocol

SRC=Path(__file__).with_name('batterylife_support_selection_6d.json'); METHODS=('median','mean','inverse_distance','rbf','local_log_ridge')
def predict(method,q,V,y,sd):
 dist=[]
 for v in V:
  ok=np.isfinite(q)&np.isfinite(v);dist.append(float(np.sqrt(np.sum(((q[ok]-v[ok])/sd[ok])**2))) if ok.any() else 0.)
 dist=np.asarray(dist)
 if method=='median':return float(np.median(y))
 if method=='mean':return float(np.mean(y))
 if method in ('inverse_distance','rbf'):
  zero=dist<1e-12
  if method=='inverse_distance':w=zero.astype(float) if zero.any() else 1/np.maximum(dist,1e-6)
  else:w=np.exp(-.5*dist**2)
  return float(w@y/w.sum())
 def phi(v):return np.r_[np.nan_to_num(v/sd,nan=0.),(~np.isfinite(v)).astype(float)]
 Z=np.asarray([phi(v) for v in V]);A=np.c_[np.ones(len(Z)),Z];pen=np.eye(A.shape[1]);pen[0,0]=0.;coef=np.linalg.solve(A.T@A+10*pen,A.T@np.log(y));return float(np.exp(np.r_[1.,phi(q)]@coef))
def main():
 x=json.load(open(SRC));labels={(r['dataset'],r['held_out']):r['truth'] for r in x['rows']};names={d:sorted(n for dd,n in labels if dd==d) for d in x['audit']};meta={};scale={}
 for d,ns in names.items():
  V=np.asarray([protocol(n) for n in ns]);sd=np.nanstd(V,axis=0);sd[~np.isfinite(sd)|(sd==0)]=1.;meta[d]={n:protocol(n) for n in ns};scale[d]=sd
 rows=[]
 for r in x['rows']:
  if r['selector']!='full_protocol' or r['k']<3:continue
  d=r['dataset'];sup=r['selected'];V=np.asarray([meta[d][n] for n in sup]);y=np.asarray([labels[(d,n)] for n in sup]);scores={}
  for method in METHODS:
   loss=[]
   for i in range(len(sup)):
    ix=[j for j in range(len(sup)) if j!=i];p=predict(method,V[i],V[ix],y[ix],scale[d]);loss.append(abs(np.log(max(p,1))-np.log(y[i])))
   scores[method]=float(np.mean(loss))
  chosen=min(METHODS,key=lambda m:(scores[m],METHODS.index(m)));p=predict(chosen,meta[d][r['held_out']],V,y,scale[d]);e=abs(p-r['truth']);rows.append({'dataset':d,'held_out':r['held_out'],'k':r['k'],'seed':r['seed'],'chosen':chosen,'support_cv':scores,'prediction':p,'truth':r['truth'],'abs_error':e,'ape':100*e/max(r['truth'],1)})
 out={'source':SRC.name,'protocol':'full-protocol support fixed; calibrator selected by support-only LOOCV absolute log error','rows':rows,'summary':{}}
 for k in [3,5,10]:
  ds=[]
  for d in names:
   q=[r for r in rows if (r['dataset'],r['k'])==(d,k)];rec={'mae':float(np.mean([r['abs_error'] for r in q])),'mape':float(np.mean([r['ape'] for r in q])),'selection_counts':{m:sum(r['chosen']==m for r in q) for m in METHODS}};out['summary'][f'{d}_k{k}']=rec;ds.append(rec)
  out['summary'][f'macro_k{k}']={'mae':float(np.mean([z['mae'] for z in ds])),'mape':float(np.mean([z['mape'] for z in ds]))}
 p=Path(__file__).with_name('batterylife_adaptive_calibrator.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['summary'],indent=2))
if __name__=='__main__':main()
