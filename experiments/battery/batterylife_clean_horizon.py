"""Leakage-resistant, cell-level cross-dataset early-life protocol.

At a fixed observation horizon each cell contributes exactly one feature vector
and one life label. Source models never see target cells. For every target test
cell, three *other* target cells are sampled as historical labeled calibration
cells. This avoids treating correlated checkpoints as independent examples.
"""
import json, random
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from batterylife_crossdomain_baseline import load

DATASETS=('CALB','HNEI','MICH_EXP')
HORIZONS=(10,20,50)
SEEDS=tuple(range(1,11))

def at_horizon(ds,h):
    out=[]
    for name,_,rows in load(ds):
        eligible=[r for r in rows if r[0][0] <= h]
        if not eligible: continue
        x,y=eligible[-1]
        # v12 MICH_EXP contains three explicit life=1 records; impossible at h>=10.
        if y < h: continue
        out.append((name,np.asarray(x,float),float(y)))
    return out

def metrics(y,p):
    e=abs(float(p)-float(y)); return e,100*e/max(float(y),1.)

def main():
    rows=[]; audit={}
    for h in HORIZONS:
      sets={d:at_horizon(d,h) for d in DATASETS}; audit[str(h)]={d:len(v) for d,v in sets.items()}
      for src in DATASETS:
       for dst in DATASETS:
        if src==dst or len(sets[src])<5 or len(sets[dst])<5: continue
        X=np.asarray([x for _,x,_ in sets[src]]); y=np.asarray([z for _,_,z in sets[src]])
        models={
          'ridge':make_pipeline(StandardScaler(),Ridge(alpha=10.0)),
          'rf':RandomForestRegressor(n_estimators=500,min_samples_leaf=2,max_features=.8,random_state=819,n_jobs=-1),
        }
        for model_name,model in models.items():
         model.fit(X,y); target=sets[dst]; target_pred=np.asarray([model.predict(x.reshape(1,-1))[0] for _,x,_ in target])
         for held,(name,_,yt) in enumerate(target):
          avail=[i for i in range(len(target)) if i!=held]
          raw=target_pred[held]
          re,rm=metrics(yt,raw)
          for seed in SEEDS:
           rng=random.Random(100000*h+1000*held+seed); idx=rng.sample(avail,min(3,len(avail)))
           pc=target_pred[idx]; yc=np.asarray([target[i][2] for i in idx])
           candidates={
             'raw':raw,
             'median_bias':raw+float(np.median(yc-pc)),
             'median_scale':raw*float(np.median(yc/np.maximum(pc,1e-6))),
           }
           # Three genuinely independent cells; affine is intentionally audited
           # despite its high variance, not silently selected when it wins.
           A=np.c_[pc,np.ones(len(pc))]; a,b=np.linalg.lstsq(A,yc,rcond=None)[0]
           candidates['affine_ols']=a*raw+b
           for method,p in candidates.items():
            e,mp=metrics(yt,p); rows.append({'horizon':h,'source':src,'target':dst,'model':model_name,'held_out':name,'seed':seed,'cal_cells':[target[i][0] for i in idx],'method':method,'prediction':float(p),'truth':yt,'abs_error':e,'ape':mp,'raw_abs_error':re,'raw_ape':rm})
    result={'dataset_versions':'BatteryLife v12','protocol':'one independent sample per cell at fixed early horizon; target test cell always held out','seeds':list(SEEDS),'audit':audit,'rows':rows,'summary':{}}
    for h in HORIZONS:
     for src in DATASETS:
      for dst in DATASETS:
       if src==dst: continue
       for model in ['ridge','rf']:
        for method in ['raw','median_bias','median_scale','affine_ols']:
         z=[r for r in rows if r['horizon']==h and r['source']==src and r['target']==dst and r['model']==model and r['method']==method]
         if z: result['summary'][f'h{h}_{src}_to_{dst}_{model}_{method}']={'mae':float(np.mean([r['abs_error'] for r in z])),'mape':float(np.mean([r['ape'] for r in z])),'median_ae':float(np.median([r['abs_error'] for r in z])),'n_evaluations':len(z)}
    path=Path(__file__).with_name('batterylife_clean_horizon.json'); path.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'audit':audit,'summary':result['summary']},indent=2))
if __name__=='__main__': main()
