"""Strict leave-one-cell-out CALB BatteryLife v12 baseline.

Uses only processed CALB curves and official v12 life labels. This is a
data-chain/benchmark result, not a claim of cross-chemistry generalization.
"""
from pathlib import Path
import json, pickle, glob
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/'data/batterylife_processed/CALB'; LABEL=ROOT/'data/batterylife_processed/Life labels/CALB_labels.json'
def scalar(arr, fn, default=0.):
    x=np.asarray([z for z in arr if z is not None and np.isfinite(z)],dtype=float)
    return float(fn(x)) if x.size else default
def read_cell(p,life):
    d=pickle.load(open(p,'rb')); cycles=d['cycle_data']; rows=[]
    cap=[]
    for c in cycles:
        q=scalar(c['discharge_capacity_in_Ah'],np.max); cap.append(q)
    for i in range(9,len(cycles)):
        lo=max(0,i-9); q=np.asarray(cap[lo:i+1],float); cyc=cycles[i]
        slope=np.polyfit(np.arange(len(q)),q,1)[0] if len(q)>1 else 0.
        f=[i+1, q[-1], slope, scalar(q,np.mean), scalar(cyc['voltage_in_V'],np.mean), scalar(cyc['voltage_in_V'],np.min), scalar(cyc['voltage_in_V'],np.max), scalar(cyc['current_in_A'],np.mean), scalar(cyc['temperature_in_C'],np.mean), scalar(cyc['time_in_s'],np.max)]
        rows.append((f,life))
    return rows
def main():
    labels=json.load(open(LABEL)); cells=[]
    for p in sorted(DATA.glob('*.pkl')):
        if p.name not in labels: continue
        rows=read_cell(p,int(labels[p.name]));
        if rows: cells.append((p.name,rows))
    results=[]; horizon=[]
    for held,(name,rows) in enumerate(cells):
        train=[r for j,(_,rs) in enumerate(cells) if j!=held for r in rs]; test=rows
        X=np.asarray([r[0] for r in train]); y=np.asarray([r[1] for r in train]); Xt=np.asarray([r[0] for r in test]); yt=np.asarray([r[1] for r in test])
        models={'ridge':make_pipeline(StandardScaler(),Ridge(alpha=10.0)),'rf':RandomForestRegressor(n_estimators=300,random_state=819,n_jobs=-1,min_samples_leaf=2)}
        for m,model in models.items():
            model.fit(X,y); pred=model.predict(Xt); results.append({'held_out':name,'model':m,'n_test':len(yt),'mae':float(mean_absolute_error(yt,pred)),'p95_abs_error':float(np.percentile(np.abs(yt-pred),95))})
            for h in [10,20,50,100]:
                mask=Xt[:,0]<=h
                if mask.any(): horizon.append({'held_out':name,'model':m,'horizon_cycles':h,'n':int(mask.sum()),'mae':float(mean_absolute_error(yt[mask],pred[mask]))})
    out={'dataset':'BatteryLife v12 CALB','cells':len(cells),'checkpoints':sum(len(r) for _,r in cells),'feature_names':['cycle','capacity_last','capacity_slope_10','capacity_mean_10','voltage_mean','voltage_min','voltage_max','current_mean','temperature_mean','time_max'],'results':results,'horizon_results':horizon}
    for m in ['ridge','rf']:
        z=[r for r in results if r['model']==m]; out[m+'_mean_mae']=float(np.mean([r['mae'] for r in z])); out[m+'_worst_mae']=float(np.max([r['mae'] for r in z])); out[m+'_mean_p95_abs_error']=float(np.mean([r['p95_abs_error'] for r in z]))
        for h in [10,20,50,100]:
            z=[r['mae'] for r in horizon if r['model']==m and r['horizon_cycles']==h]; out[f'{m}_h{h}_mean_mae']=float(np.mean(z)) if z else None
    Path(__file__).with_name('calb_lobo_baseline.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
