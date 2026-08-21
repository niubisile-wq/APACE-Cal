"""Dataset-held-out selection of robust-median/local-ridge blend strength."""
import json
from pathlib import Path
import numpy as np
SRC=Path(__file__).with_name('batterylife_local_calibrator_baselines.json');DS=('CALB','HNEI','MICH_EXP','CALCE','MICH','SNL');ALPHAS=(0.,.25,.5,.75,1.)
def main():
 x=json.load(open(SRC));tab={}
 for r in x['rows']:tab[(r['dataset'],r['held_out'],r['k'],r['seed'],r['method'])]=(r['prediction'],r['truth'])
 choices=[];rows=[]
 for k in [3,5,10]:
  for outer in DS:
   scores=[]
   for a in ALPHAS:
    per=[]
    for d in DS:
     if d==outer:continue
     es=[]
     for key,(pm,y) in tab.items():
      dd,n,kk,s,m=key
      if dd==d and kk==k and m=='median':
       pr=tab[(dd,n,kk,s,'local_log_ridge')][0];p=np.exp((1-a)*np.log(pm)+a*np.log(pr));es.append(100*abs(p-y)/max(y,1))
     per.append(np.mean(es))
    scores.append((float(np.mean(per)),a))
   score,a=min(scores);choices.append({'k':k,'outer_target':outer,'selected_alpha':a,'inner_macro_mape':score,'all_scores':[{'alpha':aa,'macro_mape':ss} for ss,aa in scores]})
   for key,(pm,y) in tab.items():
    d,n,kk,s,m=key
    if d==outer and kk==k and m=='median':
     pr=tab[(d,n,k,s,'local_log_ridge')][0];p=float(np.exp((1-a)*np.log(pm)+a*np.log(pr)));e=abs(p-y);rows.append({'dataset':d,'held_out':n,'k':k,'seed':s,'alpha':a,'prediction':p,'truth':y,'abs_error':e,'ape':100*e/max(y,1)})
 out={'source':SRC.name,'protocol':'blend alpha selected only on the other five datasets by macro-MAPE','choices':choices,'rows':rows,'summary':{}}
 for k in [3,5,10]:
  ds=[]
  for d in DS:
   q=[r for r in rows if (r['dataset'],r['k'])==(d,k)];rec={'mae':float(np.mean([r['abs_error'] for r in q])),'mape':float(np.mean([r['ape'] for r in q]))};out['summary'][f'{d}_k{k}']=rec;ds.append(rec)
  out['summary'][f'macro_k{k}']={'mae':float(np.mean([z['mae'] for z in ds])),'mape':float(np.mean([z['mape'] for z in ds]))}
 p=Path(__file__).with_name('batterylife_lodo_blend.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'choices':choices,'summary':out['summary']},indent=2))
if __name__=='__main__':main()
