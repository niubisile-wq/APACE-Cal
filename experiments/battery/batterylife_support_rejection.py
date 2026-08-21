"""Support-aware selective prediction diagnostic for the calibration candidate."""
import json,itertools
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from batterylife_crossdomain_baseline import load

def main():
 cells=load('CALB')+load('HNEI'); out=[]
 for src,dst in [('CALB','HNEI'),('HNEI','CALB')]:
  s=[x for _,d,rs in cells if d==src for x in rs]; m=RandomForestRegressor(n_estimators=100,random_state=819,n_jobs=-1,min_samples_leaf=2).fit(np.asarray([x[0] for x in s]),np.asarray([x[1] for x in s]))
  t=[(n,rs) for n,d,rs in cells if d==dst]; pred=[m.predict(np.asarray([x[0] for x in rs])) for _,rs in t]
  for h,(name,rs) in enumerate(t):
   av=[i for i in range(len(t)) if i!=h]; cs=list(itertools.combinations(av,3)); cs=[cs[int(round(i*(len(cs)-1)/9))] for i in range(min(10,len(cs)))]
   pp=[]; ycal=[]
   for ix in cs:
    pc=np.concatenate([pred[j] for j in ix]); yc=np.asarray([x[1] for j in ix for x in t[j][1]]); c=LinearRegression().fit(pc.reshape(-1,1),yc); pp.append(c.predict(pred[h].reshape(-1,1)))
   p=np.median(pp,axis=0); u=np.std(pp,axis=0); y=np.asarray([x[1] for x in rs])
   for th in [25,50,100,200,500]:
    keep=u<=th
    if keep.any(): out.append({'source':src,'target':dst,'held_out':name,'threshold':th,'coverage':float(keep.mean()),'selective_mae':float(np.mean(np.abs(p[keep]-y[keep]))),'selective_mape':float(np.mean(np.abs(p[keep]-y[keep])/np.maximum(y[keep],1))*100)})
  
 result={'dataset_versions':'BatteryLife v12','rows':out}
 for src,dst in [('CALB','HNEI'),('HNEI','CALB')]:
  for th in [25,50,100,200,500]:
   z=[r for r in out if r['source']==src and r['target']==dst and r['threshold']==th];
   if z: result[f'{src}_to_{dst}_{th}']={'coverage_mean':float(np.mean([r['coverage'] for r in z])),'mae_mean':float(np.mean([r['selective_mae'] for r in z])),'mape_mean':float(np.mean([r['selective_mape'] for r in z]))}
 Path(__file__).with_name('batterylife_support_rejection.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
