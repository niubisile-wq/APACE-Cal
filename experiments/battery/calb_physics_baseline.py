"""Capacity-fade extrapolation baseline on BatteryLife v12 CALB."""
from pathlib import Path
import json,pickle,glob
import numpy as np
from sklearn.metrics import mean_absolute_error
ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/'data/batterylife_processed/CALB'; LABEL=ROOT/'data/batterylife_processed/Life labels/CALB_labels.json'
def cap(x):
 a=np.asarray([z for z in x if z is not None and np.isfinite(z)],float); return float(np.max(a)) if len(a) else np.nan
def main():
 labels=json.load(open(LABEL)); rows=[]
 for p in sorted(DATA.glob('*.pkl')):
  d=pickle.load(open(p,'rb')); life=labels.get(p.name); cs=[cap(c['discharge_capacity_in_Ah']) for c in d['cycle_data']]; nominal=float(d.get('nominal_capacity_in_Ah') or np.nan)
  if life is None or not np.isfinite(nominal): continue
  initial=float(np.nanmedian(np.asarray(cs[:min(5,len(cs))],float)))
  for i in range(9,len(cs)):
   q=np.asarray(cs[max(0,i-9):i+1],float); q=q[np.isfinite(q)]
   if len(q)<3: continue
   slope=np.polyfit(np.arange(len(q)),q,1)[0]; current=i+1
   # Processed CALB curves use partial SOC windows and their metadata nominal
   # capacity is not the measured discharge plateau; use the first-cycle
   # measured capacity for a fair within-cell fade threshold.
   target=.8*initial
   pred=current+(target-q[-1])/slope if slope < -1e-8 else current+1000
   pred=float(np.clip(pred,current,5000)); rows.append({'cell':p.name,'cycle':current,'pred':pred,'life':int(life),'abs':abs(pred-life)})
 out={'dataset':'BatteryLife v12 CALB','checkpoints':len(rows),'mean_mae':float(np.mean([r['abs'] for r in rows])),'p95_abs_error':float(np.percentile([r['abs'] for r in rows],95)),'worst_cell_mae':{},'horizon_mae':{}}
 for cell in sorted(set(r['cell'] for r in rows)):
  z=[r['abs'] for r in rows if r['cell']==cell]; out['worst_cell_mae'][cell]=float(np.mean(z))
 out['worst_cell']=max(out['worst_cell_mae'],key=out['worst_cell_mae'].get); out['worst_mae']=out['worst_cell_mae'][out['worst_cell']]
 for h in [10,20,50,100]:
  z=[r['abs'] for r in rows if r['cycle']<=h]; out['horizon_mae'][str(h)]=float(np.mean(z)) if z else None
 Path(__file__).with_name('calb_physics_baseline.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__':main()
