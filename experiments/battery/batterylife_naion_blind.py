"""Locked blind evaluation on the seventh, chemically distinct NA-ion domain."""
import json,random,math
from pathlib import Path
import numpy as np
from batterylife_protocol_selection import protocol
from batterylife_selector_calibrator_factorial import ridge

ROOT=Path(__file__).resolve().parents[2];SEEDS=range(1,101);FACTORS={(3,.9):3.,(3,.95):6.,(10,.9):1.,(10,.95):2.}
def main():
 lab=json.load(open(ROOT/'data/batterylife_processed/Life labels/NA-ion_labels.json'));cells=sorted((n,float(y)) for n,y in lab.items() if y>=50);V=np.asarray([protocol(n) for n,_ in cells]);sd=np.nanstd(V,axis=0);sd[~np.isfinite(sd)|(sd==0)]=1.;rows=[]
 for i,(name,y) in enumerate(cells):
  for k in [3,10]:
   for seed in SEEDS:
    av=[j for j in range(len(cells)) if j!=i];random.Random(100000*i+1000*k+seed).shuffle(av);sel=av[:k];Vs=V[sel];ys=np.asarray([cells[j][1] for j in sel]);pred=ridge(V[i],Vs,ys,sd);median=float(np.median(ys));res=[]
    for u in range(len(sel)):
     ix=[j for j in range(len(sel)) if j!=u];pu=ridge(Vs[u],Vs[ix],ys[ix],sd);res.append(abs(np.log(max(pu,1))-np.log(ys[u])))
    res=np.sort(res)
    for level in [.9,.95]:
     rank=min(len(res),math.ceil((len(res)+1)*level));q=float(res[rank-1])*FACTORS[(k,level)];err=abs(np.log(max(pred,1))-np.log(y));lo=np.exp(np.log(pred)-q);hi=np.exp(np.log(pred)+q);rows.append({'held_out':name,'k':k,'seed':seed,'level':level,'factor':FACTORS[(k,level)],'median_prediction':median,'ridge_prediction':pred,'truth':y,'median_abs_error':abs(median-y),'ridge_abs_error':abs(pred-y),'covered':bool(err<=q),'normalized_width':float((hi-lo)/y)})
 out={'dataset_versions':'BatteryLife v12','dataset':'NA-ion','blind_status':'not used in six-domain method or interval-rule development','n_cells':len(cells),'metadata_audit':'all released cells share non-discriminating 25C/0-100 SOC metadata; protocol selector exactly reduces to common-random support','locked_factors':{f'k{k}_{l}':v for (k,l),v in FACTORS.items()},'rows':rows,'summary':{}}
 for k in [3,10]:
  q0=[r for r in rows if r['k']==k and r['level']==.9];out['summary'][f'k{k}_point']={'median_mae':float(np.mean([r['median_abs_error'] for r in q0])),'ridge_mae':float(np.mean([r['ridge_abs_error'] for r in q0])),'median_mape':float(np.mean([100*r['median_abs_error']/r['truth'] for r in q0])),'ridge_mape':float(np.mean([100*r['ridge_abs_error']/r['truth'] for r in q0]))}
  for level in [.9,.95]:
   q=[r for r in rows if r['k']==k and r['level']==level];out['summary'][f'k{k}_{level}']={'coverage':float(np.mean([r['covered'] for r in q])),'normalized_width':float(np.mean([r['normalized_width'] for r in q]))}
 p=Path(__file__).with_name('batterylife_naion_blind.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
