"""Empirical local conformal intervals from support-only LOO log residuals."""
import json,collections,math
from pathlib import Path
import numpy as np
from batterylife_protocol_selection import protocol
from batterylife_selector_calibrator_factorial import ridge

SRC=Path(__file__).with_name('batterylife_support_selection_6d.json');SELECTORS=('random','full_protocol');LEVELS=(.9,.95)
def main():
 x=json.load(open(SRC));labels={(r['dataset'],r['held_out']):r['truth'] for r in x['rows']};names={d:sorted(n for dd,n in labels if dd==d) for d in x['audit']};meta={};scale={}
 for d,ns in names.items():
  V=np.asarray([protocol(n) for n in ns]);sd=np.nanstd(V,axis=0);sd[~np.isfinite(sd)|(sd==0)]=1.;meta[d]={n:protocol(n) for n in ns};scale[d]=sd
 g=collections.defaultdict(list);cell=collections.defaultdict(list);cases=[]
 for r in x['rows']:
  if r['selector'] not in SELECTORS or r['k']<3:continue
  d=r['dataset'];sup=r['selected'];V=np.asarray([meta[d][n] for n in sup]);y=np.asarray([labels[(d,n)] for n in sup]);p=ridge(meta[d][r['held_out']],V,y,scale[d]);res=[]
  for i in range(len(y)):
   ix=[j for j in range(len(y)) if j!=i];pi=ridge(V[i],V[ix],y[ix],scale[d]);res.append(abs(np.log(max(pi,1))-np.log(y[i])))
  res=np.sort(res)
  for level in LEVELS:
   rank=min(len(res),math.ceil((len(res)+1)*level));q=float(res[rank-1]);lo=float(np.exp(np.log(p)-q));hi=float(np.exp(np.log(p)+q));covered=lo<=r['truth']<=hi;width=(hi-lo)/max(r['truth'],1);rec=(float(covered),width,q);g[(d,r['k'],r['selector'],level)].append(rec);cell[(d,r['held_out'],r['k'],r['selector'],level)].append(rec);cases.append({'dataset':d,'held_out':r['held_out'],'k':r['k'],'seed':r['seed'],'selector':r['selector'],'level':level,'log_error':float(abs(np.log(max(p,1))-np.log(r['truth']))),'base_log_radius':q,'prediction':p,'truth':r['truth']})
 out={'source':SRC.name,'protocol':'local log-ridge prediction; support-only LOO absolute-log residuals; finite-sample corrected empirical quantile','summary':{},'per_cell':[],'cases':cases}
 for k in [3,5,10]:
  for s in SELECTORS:
   for level in LEVELS:
    ds=[]
    for d in names:
     q=g[(d,k,s,level)];rec={'coverage':float(np.mean([z[0] for z in q])),'normalized_width':float(np.mean([z[1] for z in q])),'log_radius':float(np.mean([z[2] for z in q]))};out['summary'][f'{d}_k{k}_{s}_{level}']=rec;ds.append(rec)
    out['summary'][f'macro_k{k}_{s}_{level}']={'coverage':float(np.mean([z['coverage'] for z in ds])),'normalized_width':float(np.mean([z['normalized_width'] for z in ds])),'log_radius':float(np.mean([z['log_radius'] for z in ds]))}
 for key,q in cell.items():out['per_cell'].append({'dataset':key[0],'held_out':key[1],'k':key[2],'selector':key[3],'level':key[4],'coverage':float(np.mean([z[0] for z in q])),'normalized_width':float(np.mean([z[1] for z in q]))})
 p=Path(__file__).with_name('batterylife_local_conformal.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['summary'],indent=2))
if __name__=='__main__':main()
