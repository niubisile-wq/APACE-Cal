"""Leave-one-dataset-out multi-source relative-residual calibration."""
import json,random
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from batterylife_crossdomain_baseline import load
from batterylife_protocol_selection import protocol
from batterylife_relative_residual import invariant_feature

DS=('CALB','HNEI','MICH_EXP','CALCE','MICH','SNL'); HS=(10,20,50)
def build(cells,h):
 out=[]
 for n,d,rs in cells:
  q=[r for r in rs if r[0][0]<=h]
  if q and q[-1][1]>=h:out.append((n,d,invariant_feature(q[-1][0]),float(q[-1][1])))
 return out
def residual(items):
 logs=np.log([x[3] for x in items]); z=[]
 for i,(_,d,_,_) in enumerate(items):
  temp=protocol(items[i][0])[0]; ix=[j for j,x in enumerate(items) if x[1]==d and protocol(x[0])[0]==temp]; ix=ix if len(ix)>=3 else [j for j,x in enumerate(items) if x[1]==d]; z.append(logs[i]-np.median(logs[ix]))
 return np.asarray(z)
def select(t,i,seed):
 v=np.asarray([protocol(x[0]) for x in t]);sd=np.nanstd(v,axis=0);sd[~np.isfinite(sd)|(sd==0)]=1.;av=[j for j in range(len(t)) if j!=i];random.Random(i*10000+seed).shuffle(av)
 def dist(j):
  ok=np.isfinite(v[i])&np.isfinite(v[j]);return float(np.sqrt(np.sum(((v[i,ok]-v[j,ok])/sd[ok])**2))) if ok.any() else 0.
 return sorted(av,key=dist)[:3]
def main():
 allcells=sum((load(d) for d in DS),[]);rows=[]
 for h in HS:
  data=build(allcells,h)
  for target_ds in DS:
   src=[x for x in data if x[1]!=target_ds];tar=[x for x in data if x[1]==target_ds];X=np.asarray([x[2] for x in src]);z=residual(src)
   models={'ridge':make_pipeline(StandardScaler(),Ridge(alpha=10.)),'rf':RandomForestRegressor(n_estimators=500,min_samples_leaf=2,max_features=.8,random_state=819,n_jobs=-1),'extra':ExtraTreesRegressor(n_estimators=500,min_samples_leaf=2,max_features=.8,random_state=819,n_jobs=-1),'gbr':GradientBoostingRegressor(n_estimators=100,max_depth=2,learning_rate=.03,loss='huber',random_state=819)}
   for mn,m in models.items():
    m.fit(X,z);zp=m.predict(np.asarray([x[2] for x in tar]))
    for i,(name,_,_,y) in enumerate(tar):
     for seed in range(1,11):
      ix=select(tar,i,seed);yc=np.asarray([tar[j][3] for j in ix]);zc=zp[ix];base=float(np.median(yc));dz=float(zp[i]-np.median(zc));
      slopes=[]
      for u in range(len(ix)):
       for v in range(u):
        if abs(zc[u]-zc[v])>1e-8: slopes.append((np.log(yc[u])-np.log(yc[v]))/(zc[u]-zc[v]))
      slope=float(np.clip(np.median(slopes),0.,3.)) if slopes else 0.
      candidates=[('target_median',base),('relative_log',base*np.exp(dz)),('relative_linear',base*(1+dz)),('monotone_residual',base*np.exp(slope*dz))]
      for method,p in candidates:
       p=max(float(p),1.);e=abs(p-y);rows.append({'horizon':h,'sources':[d for d in DS if d!=target_ds],'target':target_ds,'model':mn,'held_out':name,'seed':seed,'method':method,'prediction':p,'truth':y,'abs_error':e,'ape':100*e/max(y,1)})
 out={'dataset_versions':'BatteryLife v12','protocol':'leave-one-dataset-out; two source datasets learn dataset+temperature centered residual; three protocol-nearest target cells restore scale','rows':rows,'summary':{}}
 for h in HS:
  for t in DS:
   for m in ['ridge','rf','extra','gbr']:
    for method in ['target_median','relative_log','relative_linear','monotone_residual']:
     q=[r for r in rows if (r['horizon'],r['target'],r['model'],r['method'])==(h,t,m,method)];out['summary'][f'h{h}_to_{t}_{m}_{method}']={'mae':float(np.mean([r['abs_error'] for r in q])),'mape':float(np.mean([r['ape'] for r in q])),'n_evaluations':len(q)}
 p=Path(__file__).with_name('batterylife_multisource_residual.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['summary'],indent=2))
if __name__=='__main__':main()
