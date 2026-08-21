"""Nested source-domain validation for safe relative-residual transfer.

For every outer target dataset, model family and residual shrinkage are chosen
using only leave-one-source-dataset-out validation among the remaining source
datasets. Outer target test labels never participate in candidate selection.
"""
import json,random
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from batterylife_crossdomain_baseline import load
from batterylife_multisource_residual import build,residual,select

DS=('CALB','HNEI','MICH_EXP','CALCE','MICH','SNL'); HS=(10,20,50); ALPHAS=(0.,.25,.5,.75,1.)
def model(name):
 if name=='ridge':return make_pipeline(StandardScaler(),Ridge(alpha=10.))
 if name=='rf':return RandomForestRegressor(n_estimators=300,min_samples_leaf=2,max_features=.8,random_state=819,n_jobs=-1)
 if name=='extra':return ExtraTreesRegressor(n_estimators=300,min_samples_leaf=2,max_features=.8,random_state=819,n_jobs=-1)
 return GradientBoostingRegressor(n_estimators=100,max_depth=2,learning_rate=.03,loss='huber',random_state=819)
def fit_scores(train,target,mname,alphas,seeds):
 m=model(mname);m.fit(np.asarray([x[2] for x in train]),residual(train));zp=m.predict(np.asarray([x[2] for x in target])); rows=[]
 for i,(n,_,_,y) in enumerate(target):
  for seed in seeds:
   ix=select(target,i,seed);base=float(np.median([target[j][3] for j in ix]));dz=float(zp[i]-np.median(zp[ix]));
   for alpha in alphas:
    p=base*np.exp(alpha*dz);e=abs(p-y);rows.append((alpha,n,seed,float(p),float(y),float(e),float(100*e/max(y,1))))
 return rows
def main():
 cells=sum((load(d) for d in DS),[]);all_rows=[];choices=[]
 for h in HS:
  data=build(cells,h)
  for outer in DS:
   source_ds=[d for d in DS if d!=outer]; candidates=[]
   for mn in ['ridge','rf','extra','gbr']:
    by_alpha={a:[] for a in ALPHAS}
    for val in source_ds:
     tr=[x for x in data if x[1] in source_ds and x[1]!=val];va=[x for x in data if x[1]==val]
     r=fit_scores(tr,va,mn,ALPHAS,range(1,6))
     for alpha in ALPHAS: by_alpha[alpha].append(float(np.mean([x[6] for x in r if x[0]==alpha])))
    for alpha in ALPHAS:
     domain_scores=by_alpha[alpha];candidates.append({'model':mn,'alpha':alpha,'inner_domain_mape':domain_scores,'inner_macro_mape':float(np.mean(domain_scores))})
   best=min(candidates,key=lambda x:(x['inner_macro_mape'],x['alpha'],x['model']));choices.append({'horizon':h,'outer_target':outer,'source_datasets':source_ds,'selected':best,'all_candidates':candidates})
   tr=[x for x in data if x[1]!=outer];te=[x for x in data if x[1]==outer];rr=fit_scores(tr,te,best['model'],sorted(set([0.,best['alpha']])),range(1,11))
   for mode,alpha in [('target_median',0.),('nested_selected',best['alpha'])]:
    for a,n,seed,p,y,e,ape in rr:
     if a==alpha:all_rows.append({'horizon':h,'target':outer,'model':best['model'],'alpha':alpha,'mode':mode,'held_out':n,'seed':seed,'prediction':p,'truth':y,'abs_error':e,'ape':ape})
 out={'dataset_versions':'BatteryLife v12','protocol':'outer LODO; model and alpha selected only by macro-MAPE over inner source-domain LODO; 3 protocol-nearest target calibration cells','choices':choices,'rows':all_rows,'summary':{}}
 for h in HS:
  for t in DS:
   for mode in ['target_median','nested_selected']:
    q=[r for r in all_rows if (r['horizon'],r['target'],r['mode'])==(h,t,mode)];out['summary'][f'h{h}_to_{t}_{mode}']={'mae':float(np.mean([r['abs_error'] for r in q])),'mape':float(np.mean([r['ape'] for r in q])),'n_evaluations':len(q)}
 p=Path(__file__).with_name('batterylife_nested_transfer.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'choices':choices,'summary':out['summary']},indent=2))
if __name__=='__main__':main()
