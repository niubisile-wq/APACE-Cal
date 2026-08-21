"""Nested LODO selection for pooled absolute log-life calibration."""
import json
from pathlib import Path
import numpy as np
from batterylife_crossdomain_baseline import load
from batterylife_pooled_log_calibration import DS,HS,build,model
from batterylife_multisource_residual import select

ALPHAS=(0.,.25,.5,.75,1.)
def fit_scores(train,target,mname,alphas,seeds):
 m=model(mname);m.fit(np.asarray([x[2] for x in train]),np.log([x[3] for x in train]));lp=m.predict(np.asarray([x[2] for x in target]));rows=[]
 for i,(n,_,_,y) in enumerate(target):
  for seed in seeds:
   ix=select(target,i,seed);ly=np.log([target[j][3] for j in ix]);lb=float(np.median(ly));lc=float(lp[i]+np.median(ly-lp[ix]));
   for a in alphas:
    p=float(np.exp(lb+a*(lc-lb)));e=abs(p-y);rows.append((a,n,seed,p,float(y),float(e),float(100*e/max(y,1))))
 return rows
def main():
 cells=sum((load(d) for d in DS),[]);rows=[];choices=[]
 for h in HS:
  data=build(cells,h)
  for outer in DS:
   srcds=[d for d in DS if d!=outer];cands=[]
   for mn in ['ridge','rf','extra','gbr']:
    scores={a:[] for a in ALPHAS}
    for val in srcds:
     tr=[x for x in data if x[1] in srcds and x[1]!=val];va=[x for x in data if x[1]==val];rr=fit_scores(tr,va,mn,ALPHAS,range(1,6))
     for a in ALPHAS:scores[a].append(float(np.mean([r[6] for r in rr if r[0]==a])))
    for a in ALPHAS:cands.append({'model':mn,'alpha':a,'inner_domain_mape':scores[a],'inner_macro_mape':float(np.mean(scores[a]))})
   best=min(cands,key=lambda z:(z['inner_macro_mape'],z['alpha'],z['model']));choices.append({'horizon':h,'outer_target':outer,'selected':best,'all_candidates':cands})
   tr=[x for x in data if x[1]!=outer];te=[x for x in data if x[1]==outer];rr=fit_scores(tr,te,best['model'],sorted(set([0.,best['alpha']])),range(1,11))
   for mode,a in [('target_median',0.),('nested_pooled',best['alpha'])]:
    for aa,n,seed,p,y,e,ape in rr:
     if aa==a:rows.append({'horizon':h,'target':outer,'model':best['model'],'alpha':a,'mode':mode,'held_out':n,'seed':seed,'prediction':p,'truth':y,'abs_error':e,'ape':ape})
 out={'dataset_versions':'BatteryLife v12','protocol':'outer six-dataset LODO; pooled absolute model and geometric calibration shrinkage selected only by inner source-domain macro-MAPE','choices':choices,'rows':rows,'summary':{}}
 for h in HS:
  for t in DS:
   for mode in ['target_median','nested_pooled']:
    q=[r for r in rows if (r['horizon'],r['target'],r['mode'])==(h,t,mode)];out['summary'][f'h{h}_to_{t}_{mode}']={'mae':float(np.mean([r['abs_error'] for r in q])),'mape':float(np.mean([r['ape'] for r in q])),'n_evaluations':len(q)}
 p=Path(__file__).with_name('batterylife_nested_pooled.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'choices':choices,'summary':out['summary']},indent=2))
if __name__=='__main__':main()
