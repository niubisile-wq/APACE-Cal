"""Protocol-conditioned relative residual transfer.

The source model learns only log-life deviations within source temperature
groups from scale-normalized early degradation features. Three protocol-nearest
target cells restore the absolute target scale. This directly tests whether
source degradation information adds value beyond a target-only median baseline.
"""
import json,random
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from batterylife_crossdomain_baseline import load
from batterylife_protocol_selection import protocol

DATASETS=('CALB','HNEI','MICH_EXP'); HORIZONS=(10,20,50); SEEDS=range(1,11)

def invariant_feature(x):
    x=np.asarray(x,float); q=max(abs(x[3]),1e-6)
    return np.asarray([x[2]/q,(x[1]-x[3])/q,x[4],x[5],x[6],x[7]/q,x[8],np.log1p(max(x[9],0))])

def build(cells,h):
 out=[]
 for n,_,rs in cells:
  z=[r for r in rs if r[0][0]<=h]
  if z and z[-1][1]>=h: out.append((n,invariant_feature(z[-1][0]),float(z[-1][1])))
 return out

def source_residual_labels(cells):
    logs=np.log([y for _,_,y in cells]); temps=np.asarray([protocol(n)[0] for n,_,_ in cells]); z=[]
    global_med=float(np.median(logs))
    for i,t in enumerate(temps):
        ix=np.where(temps==t)[0]; center=float(np.median(logs[ix])) if len(ix)>=3 else global_med; z.append(logs[i]-center)
    return np.asarray(z)

def nearest(cells,i,seed):
    v=np.asarray([protocol(n) for n,_,_ in cells]); sd=np.nanstd(v,axis=0); sd[~np.isfinite(sd)|(sd==0)]=1.; av=[j for j in range(len(cells)) if j!=i]; random.Random(10000*i+seed).shuffle(av)
    def d(j):
      ok=np.isfinite(v[i])&np.isfinite(v[j]); return float(np.sqrt(np.sum(((v[i,ok]-v[j,ok])/sd[ok])**2))) if ok.any() else 0.
    return sorted(av,key=d)[:3]

def main():
 loaded={d:load(d) for d in DATASETS}; rows=[]; audit={}
 for h in HORIZONS:
  sets={d:build(loaded[d],h) for d in DATASETS}; audit[str(h)]={d:len(v) for d,v in sets.items()}
  for src in DATASETS:
   X=np.asarray([x for _,x,_ in sets[src]]); z=source_residual_labels(sets[src])
   models={
    'ridge':make_pipeline(StandardScaler(),Ridge(alpha=10.)),
    'rf':RandomForestRegressor(n_estimators=500,min_samples_leaf=2,max_features=.8,random_state=819,n_jobs=-1),
    'extra':ExtraTreesRegressor(n_estimators=500,min_samples_leaf=2,max_features=.8,random_state=819,n_jobs=-1),
    'gbr':GradientBoostingRegressor(n_estimators=100,max_depth=2,learning_rate=.03,loss='huber',random_state=819),
   }
   for mn,m in models.items():
    m.fit(X,z)
    for dst in DATASETS:
     if dst==src:continue
     target=sets[dst]; zp=m.predict(np.asarray([x for _,x,_ in target]))
     for i,(name,_,y) in enumerate(target):
      for seed in SEEDS:
       ix=nearest(target,i,seed); yc=np.asarray([target[j][2] for j in ix]); base=float(np.median(yc)); dz=float(zp[i]-np.median(zp[ix])); candidates={'target_median':base,'relative_log':base*np.exp(dz),'relative_linear':base*(1+dz)}
       for method,p in candidates.items():
        p=max(float(p),1.); e=abs(p-y); rows.append({'horizon':h,'source':src,'target':dst,'model':mn,'held_out':name,'seed':seed,'selected':[target[j][0] for j in ix],'method':method,'prediction':p,'truth':y,'abs_error':e,'ape':100*e/max(y,1)})
 out={'dataset_versions':'BatteryLife v12','protocol':'one sample per cell; source learns within-temperature log-life residual; 3 protocol-nearest target cells restore scale','audit':audit,'rows':rows,'summary':{}}
 for h in HORIZONS:
  for s in DATASETS:
   for t in DATASETS:
    if s==t:continue
    for m in ['ridge','rf','extra','gbr']:
     for method in ['target_median','relative_log','relative_linear']:
      q=[r for r in rows if (r['horizon'],r['source'],r['target'],r['model'],r['method'])==(h,s,t,m,method)]
      out['summary'][f'h{h}_{s}_to_{t}_{m}_{method}']={'mae':float(np.mean([r['abs_error'] for r in q])),'mape':float(np.mean([r['ape'] for r in q])),'n_evaluations':len(q)}
 p=Path(__file__).with_name('batterylife_relative_residual.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'audit':audit,'summary':out['summary']},indent=2))
if __name__=='__main__':main()
