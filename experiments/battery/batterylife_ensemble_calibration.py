"""Deterministic ensemble of few-shot affine calibrators.

For each held-out target cell, form 20 reproducible 3-cell calibration subsets
from the remaining target cells. Each subset fits an affine map from source
predictions to target labels; the final prediction is the median across maps.
This tests whether calibration gains survive subset choice without selecting a
favorable seed.
"""
import json, itertools
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor
from batterylife_crossdomain_baseline import load

def main():
 cells=load('CALB')+load('HNEI'); rows=[]
 for src,dst in [('CALB','HNEI'),('HNEI','CALB')]:
  source=[x for _,d,rs in cells if d==src for x in rs]
  m=RandomForestRegressor(n_estimators=150,random_state=819,n_jobs=-1,min_samples_leaf=2).fit(np.asarray([x[0] for x in source]),np.asarray([x[1] for x in source]))
  target=[(n,rs) for n,d,rs in cells if d==dst]
  target_pred=[m.predict(np.asarray([x[0] for x in rs])) for _,rs in target]
  for held,(name,testrows) in enumerate(target):
   avail=[i for i in range(len(target)) if i!=held]; combos=list(itertools.combinations(avail,3));
   # fixed spread across lexicographic combinations; all combinations for <=15 cells
   if len(combos)>10: combos=[combos[int(round(i*(len(combos)-1)/9))] for i in range(10)]
   yt=np.asarray([x[1] for x in testrows]); raw=target_pred[held]
   for cname, C in [('ols',LinearRegression),('huber',HuberRegressor)]:
    preds=[]
    for idxs in combos:
     pc=np.concatenate([target_pred[j] for j in idxs]); yc=np.asarray([x[1] for j in idxs for x in target[j][1]]); c=C() if cname=='ols' else C(epsilon=1.35,max_iter=2000); c.fit(pc.reshape(-1,1),yc); preds.append(c.predict(raw.reshape(-1,1)))
    adj=np.median(np.asarray(preds),axis=0); e=np.abs(adj-yt); er=np.abs(raw-yt)
    rows.append({'source':src,'target':dst,'held_out':name,'calibrator':cname,'n_subsets':len(combos),'raw_mae':float(er.mean()),'raw_mape':float(np.mean(er/np.maximum(yt,1))*100),'mae':float(e.mean()),'mape':float(np.mean(e/np.maximum(yt,1))*100),'subset_mae_std':float(np.mean(np.std(np.asarray(preds),axis=0)) )})
 out={'dataset_versions':'BatteryLife v12','rows':rows}
 for src,dst in [('CALB','HNEI'),('HNEI','CALB')]:
  for c in ['ols','huber']:
   z=[r for r in rows if r['source']==src and r['target']==dst and r['calibrator']==c]; out[f'{src}_to_{dst}_{c}']={'mae_mean':float(np.mean([r['mae'] for r in z])),'mape_mean':float(np.mean([r['mape'] for r in z])),'subset_pred_std_mean':float(np.mean([r['subset_mae_std'] for r in z]))}
 Path(__file__).with_name('batterylife_ensemble_calibration.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
if __name__=='__main__': main()
