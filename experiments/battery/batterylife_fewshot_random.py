"""Randomized few-shot calibration robustness for CALB<->HNEI."""
import json, random
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from batterylife_crossdomain_baseline import load
ROOT=Path(__file__).resolve().parents[2]

def main():
    cells=load('CALB')+load('HNEI'); rows=[]
    for src,dst in [('CALB','HNEI'),('HNEI','CALB')]:
        source=[x for _,d,rs in cells if d==src for x in rs]
        X=np.asarray([x[0] for x in source]); y=np.asarray([x[1] for x in source])
        m=RandomForestRegressor(n_estimators=300,random_state=819,n_jobs=-1,min_samples_leaf=2); m.fit(X,y)
        target=[(n,rs) for n,d,rs in cells if d==dst]
        for seed in [1,2,3,4,5]:
            rng=random.Random(seed)
            for held,(name,testrows) in enumerate(target):
                other=[x for j,x in enumerate(target) if j!=held]; cal=rng.sample(other,min(3,len(other)))
                cr=[x for _,rs in cal for x in rs]; pc=m.predict(np.asarray([x[0] for x in cr]))
                calfit=LinearRegression().fit(pc.reshape(-1,1),np.asarray([x[1] for x in cr]))
                teX=np.asarray([x[0] for x in testrows]); yt=np.asarray([x[1] for x in testrows]); raw=m.predict(teX); adj=calfit.predict(raw.reshape(-1,1))
                rows += [{'source':src,'target':dst,'seed':seed,'held_out':name,'mode':'raw','mae':float(np.mean(abs(raw-yt))),'mape':float(np.mean(abs(raw-yt)/np.maximum(yt,1))*100)}, {'source':src,'target':dst,'seed':seed,'held_out':name,'mode':'calibrated','mae':float(np.mean(abs(adj-yt))),'mape':float(np.mean(abs(adj-yt)/np.maximum(yt,1))*100)}]
    out={'dataset_versions':'BatteryLife v12','rows':rows}
    for src,dst in [('CALB','HNEI'),('HNEI','CALB')]:
        for mode in ['raw','calibrated']:
            z=[r for r in rows if r['source']==src and r['target']==dst and r['mode']==mode]
            out[f'{src}_to_{dst}_{mode}_mae_mean']=float(np.mean([r['mae'] for r in z])); out[f'{src}_to_{dst}_{mode}_mae_std']=float(np.std([r['mae'] for r in z],ddof=1)); out[f'{src}_to_{dst}_{mode}_mape_mean']=float(np.mean([r['mape'] for r in z]))
    Path(__file__).with_name('batterylife_fewshot_random.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
if __name__=='__main__': main()
