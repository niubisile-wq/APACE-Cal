"""Active protocol-aware few-shot calibration candidate.

The scientific protocol is fixed: source cells train the predictor; one target
cell is held out for testing; three *other* target cells are historical
calibration cells.  Selection uses only early-cycle feature summaries, while
their labels are used only to fit the calibrator.  This script compares
farthest-point selection with random selection and OLS/Huber affine maps.
"""
import json, random
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.preprocessing import StandardScaler
from batterylife_crossdomain_baseline import load

ROOT = Path(__file__).resolve().parents[2]

def cell_signature(rows):
    # Early behavior only; no lifetime labels and no late-cycle information.
    early = rows[:min(10, len(rows))]
    return np.median(np.asarray([x[0] for x in early], float), axis=0)

def farthest_indices(signatures, held, k=3):
    avail = [i for i in range(len(signatures)) if i != held]
    if not avail: return []
    z = StandardScaler().fit_transform(np.asarray(signatures))
    # deterministic medoid-like start: closest to global center, then farthest point.
    center = z[avail].mean(axis=0)
    chosen = [min(avail, key=lambda i: np.linalg.norm(z[i]-center))]
    while len(chosen) < min(k, len(avail)):
        nxt = max((i for i in avail if i not in chosen),
                  key=lambda i: min(np.linalg.norm(z[i]-z[j]) for j in chosen))
        chosen.append(nxt)
    return chosen

def score(y, p):
    e = np.abs(np.asarray(y)-np.asarray(p))
    return float(e.mean()), float(np.mean(e/np.maximum(np.asarray(y),1))*100)

def main():
    cells = load('CALB') + load('HNEI'); all_rows=[]
    for src, dst in [('CALB','HNEI'), ('HNEI','CALB')]:
        source=[x for _,d,rs in cells if d==src for x in rs]
        X=np.asarray([x[0] for x in source]); y=np.asarray([x[1] for x in source])
        model=RandomForestRegressor(n_estimators=300, random_state=819, n_jobs=-1, min_samples_leaf=2).fit(X,y)
        target=[(n,rs) for n,d,rs in cells if d==dst]
        sig=[cell_signature(rs) for _,rs in target]
        for held,(name,testrows) in enumerate(target):
            selections={'active':farthest_indices(sig,held,3)}
            for seed in [1,2,3,4,5]:
                rng=random.Random(seed); avail=[i for i in range(len(target)) if i!=held]
                selections[f'random_{seed}']=rng.sample(avail,min(3,len(avail)))
            teX=np.asarray([x[0] for x in testrows]); yt=np.asarray([x[1] for x in testrows]); raw=model.predict(teX)
            for selector,idxs in selections.items():
                cr=[x for j in idxs for x in target[j][1]]; pc=model.predict(np.asarray([x[0] for x in cr])); yc=np.asarray([x[1] for x in cr])
                for calname, cal in [('ols',LinearRegression()), ('huber',HuberRegressor(epsilon=1.35, max_iter=2000))]:
                    cal.fit(pc.reshape(-1,1),yc); adj=cal.predict(raw.reshape(-1,1)); mae,mape=score(yt,adj); rmae,rmape=score(yt,raw)
                    resid=np.abs(yc-cal.predict(pc.reshape(-1,1))); q90=float(np.quantile(resid,.9,method='higher')); q95=float(np.quantile(resid,.95,method='higher'))
                    cov90=float(np.mean((yt>=adj-q90)&(yt<=adj+q90))); cov95=float(np.mean((yt>=adj-q95)&(yt<=adj+q95)))
                    all_rows.append({'source':src,'target':dst,'held_out':name,'selector':selector,'calibrator':calname,'n_cal_cells':len(idxs),'raw_mae':rmae,'raw_mape':rmape,'mae':mae,'mape':mape,'q90':q90,'q95':q95,'coverage90':cov90,'coverage95':cov95})
    out={'dataset_versions':'BatteryLife v12','rows':all_rows}
    for src,dst in [('CALB','HNEI'),('HNEI','CALB')]:
      for sel in ['active','random_1','random_2','random_3','random_4','random_5']:
       for cal in ['ols','huber']:
        z=[r for r in all_rows if r['source']==src and r['target']==dst and r['selector']==sel and r['calibrator']==cal]
        if z: out[f'{src}_to_{dst}_{sel}_{cal}']={'mae_mean':float(np.mean([r['mae'] for r in z])),'mape_mean':float(np.mean([r['mape'] for r in z])),'coverage90_mean':float(np.mean([r['coverage90'] for r in z]))}
    path=Path(__file__).with_name('batterylife_active_robust.json'); path.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
if __name__=='__main__': main()
