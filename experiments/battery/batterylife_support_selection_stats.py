"""Paired statistics and hierarchical bootstrap for the six-domain K=3 result."""
import json,collections
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

SRC=Path(__file__).with_name('batterylife_support_selection_6d.json');DS=('CALB','HNEI','MICH_EXP','CALCE','MICH','SNL')
def main():
 x=json.load(open(SRC));rows=x['rows'];out={'source':SRC.name,'k':3,'per_dataset':{}}
 tables={metric:{sel:collections.defaultdict(list) for sel in ['random','full_protocol']} for metric in ['abs_error','ape']}
 for r in rows:
  if r['k']==3 and r['selector'] in ['random','full_protocol']:
   for metric in tables:tables[metric][r['selector']][(r['dataset'],r['held_out'])].append(r[metric])
 for d in DS:
  rec={}
  for metric in tables:
   keys=sorted(k for k in tables[metric]['random'] if k[0]==d);a=np.asarray([np.mean(tables[metric]['random'][k]) for k in keys]);b=np.asarray([np.mean(tables[metric]['full_protocol'][k]) for k in keys]);rec[metric]={'random_mean':float(a.mean()),'protocol_mean':float(b.mean()),'relative_reduction_pct':float(100*(1-b.mean()/a.mean())),'improved_cells':int((b<a).sum()),'n_cells':len(keys),'wilcoxon_greater_p':float(wilcoxon(a,b,alternative='greater').pvalue) if not np.allclose(a,b) else 1.0}
  out['per_dataset'][d]=rec
 rng=np.random.default_rng(819);out['macro_bootstrap']={}
 for metric in tables:
  arrays={}
  for d in DS:
   keys=sorted(k for k in tables[metric]['random'] if k[0]==d);arrays[d]=(np.asarray([np.mean(tables[metric]['random'][k]) for k in keys]),np.asarray([np.mean(tables[metric]['full_protocol'][k]) for k in keys]))
  boots=[]
  for _ in range(10000):
   ar=[];bp=[]
   for d in DS:
    a,b=arrays[d];ix=rng.integers(0,len(a),len(a));ar.append(a[ix].mean());bp.append(b[ix].mean())
   boots.append(100*(1-np.mean(bp)/np.mean(ar)))
  point=100*(1-np.mean([arrays[d][1].mean() for d in DS])/np.mean([arrays[d][0].mean() for d in DS]));out['macro_bootstrap'][metric]={'relative_reduction_pct':float(point),'ci95_pct':[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))],'bootstrap_repetitions':10000}
 p=Path(__file__).with_name('batterylife_support_selection_stats.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
