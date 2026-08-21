"""Five-source pooled log-life model with protocol-retrieved local calibration."""
import json,random
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from batterylife_crossdomain_baseline import load
from batterylife_relative_residual import invariant_feature
from batterylife_multisource_residual import select
from batterylife_protocol_selection import protocol

DS=('CALB','HNEI','MICH_EXP','CALCE','MICH','SNL');HS=(10,20,50)
def feat(name,x):
 p=protocol(name);miss=(~np.isfinite(p)).astype(float);p=np.nan_to_num(p,nan=0.);return np.r_[invariant_feature(x),p,miss]
def build(cells,h):
 out=[]
 for n,d,rs in cells:
  q=[r for r in rs if r[0][0]<=h]
  if q and q[-1][1]>=h:out.append((n,d,feat(n,q[-1][0]),float(q[-1][1])))
 return out
def model(name):
 if name=='ridge':return make_pipeline(StandardScaler(),Ridge(alpha=10.))
 if name=='rf':return RandomForestRegressor(n_estimators=500,min_samples_leaf=2,max_features=.8,random_state=819,n_jobs=-1)
 if name=='extra':return ExtraTreesRegressor(n_estimators=500,min_samples_leaf=2,max_features=.8,random_state=819,n_jobs=-1)
 return GradientBoostingRegressor(n_estimators=150,max_depth=2,learning_rate=.03,loss='huber',random_state=819)
def main():
 cells=sum((load(d) for d in DS),[]);rows=[];audit={}
 for h in HS:
  data=build(cells,h);audit[str(h)]={d:sum(x[1]==d for x in data) for d in DS}
  for target_ds in DS:
   tr=[x for x in data if x[1]!=target_ds];te=[x for x in data if x[1]==target_ds]
   for mn in ['ridge','rf','extra','gbr']:
    m=model(mn);m.fit(np.asarray([x[2] for x in tr]),np.log([x[3] for x in tr]));lp=m.predict(np.asarray([x[2] for x in te]))
    for i,(name,_,_,y) in enumerate(te):
     for seed in range(1,11):
      ix=select(te,i,seed);ly=np.log([te[j][3] for j in ix]);base=float(np.exp(np.median(ly)));delta=float(np.median(ly-lp[ix]));cal=float(np.exp(lp[i]+delta));raw=float(np.exp(lp[i]));
      for method,p in [('raw_pooled',raw),('target_median',base),('log_scale_calibrated',cal)]:
       e=abs(p-y);rows.append({'horizon':h,'target':target_ds,'model':mn,'held_out':name,'seed':seed,'method':method,'prediction':p,'truth':y,'abs_error':e,'ape':100*e/max(y,1)})
 out={'dataset_versions':'BatteryLife v12','protocol':'five-source LODO pooled log-life; normalized early degradation + protocol metadata; three protocol-nearest target cells calibrate log scale','audit':audit,'rows':rows,'summary':{}}
 for h in HS:
  for t in DS:
   for m in ['ridge','rf','extra','gbr']:
    for method in ['raw_pooled','target_median','log_scale_calibrated']:
     q=[r for r in rows if (r['horizon'],r['target'],r['model'],r['method'])==(h,t,m,method)];out['summary'][f'h{h}_to_{t}_{m}_{method}']={'mae':float(np.mean([r['abs_error'] for r in q])),'mape':float(np.mean([r['ape'] for r in q])),'n_evaluations':len(q)}
 p=Path(__file__).with_name('batterylife_pooled_log_calibration.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'audit':audit,'summary':out['summary']},indent=2))
if __name__=='__main__':main()
