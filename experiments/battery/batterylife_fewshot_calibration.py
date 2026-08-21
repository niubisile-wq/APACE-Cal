"""Few-shot affine target-domain calibration for BatteryLife transfer."""
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from batterylife_crossdomain_baseline import load
ROOT=Path(__file__).resolve().parents[2]
def fit_source(src_cells,src):
 tr=[x for _,d,rs in src_cells if d==src for x in rs]; X=np.asarray([x[0] for x in tr]); y=np.asarray([x[1] for x in tr]); m=RandomForestRegressor(n_estimators=300,random_state=819,n_jobs=-1,min_samples_leaf=2); m.fit(X,y); return m
def main():
 cells=load('CALB')+load('HNEI'); allrows=[]
 for src,dst in [('CALB','HNEI'),('HNEI','CALB')]:
  m=fit_source(cells,src); target=[(n,d,rs) for n,d,rs in cells if d==dst]
  for held,(name,_,testrows) in enumerate(target):
   other=[x for j,x in enumerate(target) if j!=held]
   for k in [3,len(other)]:
    cal_cells=other if k==len(other) else other[:k]
    cal=[x for _,_,rs in cal_cells for x in rs]; Xc=np.asarray([x[0] for x in cal]); yc=np.asarray([x[1] for x in cal]); pc=m.predict(Xc)
    calmodel=LinearRegression().fit(pc.reshape(-1,1),yc)
    Xt=np.asarray([x[0] for x in testrows]); yt=np.asarray([x[1] for x in testrows]); raw=m.predict(Xt); adj=calmodel.predict(raw.reshape(-1,1))
    for mode,p in [('raw',raw),('calibrated',adj)]:
     allrows.append({'train_dataset':src,'target_dataset':dst,'held_out':name,'calibration_cells':k,'mode':mode,'mae':float(np.mean(np.abs(p-yt))),'mape':float(np.mean(np.abs(p-yt)/np.maximum(yt,1))*100)})
 out={'dataset_versions':'BatteryLife v12','rows':allrows}
 for src,dst in [('CALB','HNEI'),('HNEI','CALB')]:
  for k in [3,'all']:
   for mode in ['raw','calibrated']:
    z=[r for r in allrows if r['train_dataset']==src and r['target_dataset']==dst and r['calibration_cells']==(len([1 for _,d,_ in cells if d==dst])-1 if k=='all' else k) and r['mode']==mode]
    if z: out[f'{src}_to_{dst}_k{k}_{mode}_mean_mae']=float(np.mean([r['mae'] for r in z])); out[f'{src}_to_{dst}_k{k}_{mode}_mean_mape']=float(np.mean([r['mape'] for r in z]))
 Path(__file__).with_name('batterylife_fewshot_calibration.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
