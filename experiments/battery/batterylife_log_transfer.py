"""Log-life target transform probe for CALB<->HNEI transfer."""
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from batterylife_crossdomain_baseline import load
ROOT=Path(__file__).resolve().parents[2]

def main():
    cells=load('CALB')+load('HNEI'); out=[]
    for src,dst in [('CALB','HNEI'),('HNEI','CALB')]:
        tr=[x for _,d,rs in cells if d==src for x in rs]; te=[x for _,d,rs in cells if d==dst for x in rs]
        X=np.asarray([x[0] for x in tr]); y=np.asarray([x[1] for x in tr]); Xt=np.asarray([x[0] for x in te]); yt=np.asarray([x[1] for x in te])
        for mode in ['raw','log']:
            m=RandomForestRegressor(n_estimators=300,random_state=819,n_jobs=-1,min_samples_leaf=2)
            m.fit(X,np.log1p(y) if mode=='log' else y); p=m.predict(Xt); p=np.expm1(p) if mode=='log' else p
            out.append({'train_dataset':src,'test_dataset':dst,'target':mode,'mae':float(mean_absolute_error(yt,p)),'mape':float(np.mean(np.abs(p-yt)/np.maximum(yt,1))*100),'p95_abs_error':float(np.percentile(np.abs(p-yt),95))})
    result={'dataset_versions':'BatteryLife v12','rows':out}; (ROOT/'experiments/battery/batterylife_log_transfer.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
