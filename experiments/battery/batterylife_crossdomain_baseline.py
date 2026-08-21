"""Cross-dataset BatteryLife v12 baseline: CALB <-> HNEI.

Reports both leave-one-cell-out within the pooled set and strict
leave-one-dataset-out transfer. No random cycle split is used.
"""
from pathlib import Path
import json,pickle,glob
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
ROOT=Path(__file__).resolve().parents[2]; BASE=ROOT/'data/batterylife_processed';
def stat(a,fn,default=0.):
 if a is None: return default
 try: vals=np.asarray(a,dtype=object).reshape(-1)
 except Exception: vals=[a]
 x=np.asarray([z for z in vals if z is not None and np.isfinite(z)],float); return float(fn(x)) if x.size else default
def load(ds):
 labels=json.load(open(BASE/'Life labels'/f'{ds}_labels.json')); cells=[]
 for p in sorted((BASE/ds).glob('*.pkl')):
  if p.name not in labels: continue
  d=pickle.load(open(p,'rb')); cyc=d['cycle_data']; caps=[stat(c['discharge_capacity_in_Ah'],np.max,np.nan) for c in cyc]; rows=[]
  for i in range(9,len(cyc)):
   q=np.asarray(caps[max(0,i-9):i+1],float); q=q[np.isfinite(q)]
   if len(q)<3: continue
   slope=np.polyfit(np.arange(len(q)),q,1)[0]
   c=cyc[i]; rows.append(([i+1,q[-1],slope,stat(q,np.mean),stat(c['voltage_in_V'],np.mean),stat(c['voltage_in_V'],np.min),stat(c['voltage_in_V'],np.max),stat(c['current_in_A'],np.mean),stat(c['temperature_in_C'],np.mean),stat(c['time_in_s'],np.max)], int(labels[p.name])))
  if rows: cells.append((p.name,ds,rows))
 return cells
def model_scores(train,test):
 X=np.asarray([x[0] for x in train]); y=np.asarray([x[1] for x in train]); Xt=np.asarray([x[0] for x in test]); yt=np.asarray([x[1] for x in test]); out=[]
 for name,m in [('ridge',make_pipeline(StandardScaler(),Ridge(alpha=10.))),('rf',RandomForestRegressor(n_estimators=300,random_state=819,n_jobs=-1,min_samples_leaf=2))]:
  m.fit(X,y); pred=m.predict(Xt); out.append({'model':name,'n_test':len(yt),'mae':float(mean_absolute_error(yt,pred)),'p95_abs_error':float(np.percentile(np.abs(yt-pred),95))})
 return out
def main():
 cells=load('CALB')+load('HNEI'); results=[]
 # pooled leave-one-cell-out
 for i,(name,ds,rows) in enumerate(cells):
  train=[x for j,(_,_,rs) in enumerate(cells) if j!=i for x in rs]; results += [{'protocol':'pooled_loco','held_out':name,'held_out_dataset':ds,**r} for r in model_scores(train,rows)]
 # strict dataset transfer
 for src,dst in [('CALB','HNEI'),('HNEI','CALB')]:
  train=[x for _,d,rs in cells if d==src for x in rs]; test=[x for _,d,rs in cells if d==dst for x in rs]
  results += [{'protocol':'dataset_transfer','train_dataset':src,'test_dataset':dst,**r} for r in model_scores(train,test)]
 out={'dataset_versions':'BatteryLife v12','datasets':{d:sum(1 for _,ds,_ in cells if ds==d) for d in ['CALB','HNEI']},'cells':len(cells),'checkpoints':sum(len(r) for _,_,r in cells),'results':results}
 for protocol in ['pooled_loco','dataset_transfer']:
  for m in ['ridge','rf']:
   z=[r for r in results if r['protocol']==protocol and r['model']==m];
   if z: out[f'{protocol}_{m}_mean_mae']=float(np.mean([r['mae'] for r in z])); out[f'{protocol}_{m}_worst_mae']=float(np.max([r['mae'] for r in z]))
 Path(__file__).with_name('batterylife_crossdomain_baseline.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__':main()
