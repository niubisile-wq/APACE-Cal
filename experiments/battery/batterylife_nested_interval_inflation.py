"""Source-domain-selected conservative inflation for local conformal intervals."""
import json
from pathlib import Path
import numpy as np
SRC=Path(__file__).with_name('batterylife_local_conformal.json');DS=('CALB','HNEI','MICH_EXP','CALCE','MICH','SNL');FACTORS=(1.,1.25,1.5,2.,3.,4.,6.,8.)
def eval_cases(cases,f):
 cov=np.mean([c['log_error']<=f*c['base_log_radius'] for c in cases]);wid=np.mean([(np.exp(np.log(c['prediction'])+f*c['base_log_radius'])-np.exp(np.log(c['prediction'])-f*c['base_log_radius']))/max(c['truth'],1) for c in cases]);return float(cov),float(wid)
def main():
 x=json.load(open(SRC));cases=[c for c in x['cases'] if c['selector']=='full_protocol'];groups={}
 for c in cases:groups.setdefault((c['dataset'],c['k'],c['level']),[]).append(c)
 choices=[];rows=[]
 for k in [3,5,10]:
  for level in [.9,.95]:
   for outer in DS:
    candidates=[]
    for f in FACTORS:
     covs=[]
     for d in DS:
      if d!=outer:covs.append(eval_cases(groups[(d,k,level)],f)[0])
     candidates.append({'factor':f,'inner_coverages':covs,'inner_min_coverage':min(covs),'inner_macro_coverage':float(np.mean(covs))})
    feasible=[z for z in candidates if z['inner_min_coverage']>=level];chosen=min(feasible,key=lambda z:z['factor']) if feasible else max(candidates,key=lambda z:z['factor']);choices.append({'k':k,'level':level,'outer_target':outer,'selected':chosen,'all_candidates':candidates})
    q=groups[(outer,k,level)];cov,wid=eval_cases(q,chosen['factor']);rows.append({'k':k,'level':level,'target':outer,'factor':chosen['factor'],'coverage':cov,'normalized_width':wid})
 out={'source':SRC.name,'protocol':'inflation factor selected only on other five datasets; smallest factor whose minimum source-domain coverage meets target','choices':choices,'rows':rows,'summary':{}}
 for k in [3,5,10]:
  for level in [.9,.95]:
   q=[r for r in rows if (r['k'],r['level'])==(k,level)];out['summary'][f'macro_k{k}_{level}']={'coverage':float(np.mean([r['coverage'] for r in q])),'min_domain_coverage':float(np.min([r['coverage'] for r in q])),'normalized_width':float(np.mean([r['normalized_width'] for r in q]))}
 p=Path(__file__).with_name('batterylife_nested_interval_inflation.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'rows':rows,'summary':out['summary']},indent=2))
if __name__=='__main__':main()
