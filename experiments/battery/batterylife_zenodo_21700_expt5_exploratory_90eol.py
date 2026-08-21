"""Nonblind exploratory 90%-EOL evaluation for frozen Expt5 manifest."""
from __future__ import annotations

import csv, io, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import batterylife_asymmetric_cohort_router as v1
import batterylife_asymmetric_cohort_router_v2 as v2
import batterylife_zenodo_21700_prelabel as io217

HERE=Path(__file__).parent
MANIFEST=HERE/'batterylife_zenodo_21700_expt5_prelabel.json'
AUDIT=HERE/'batterylife_zenodo_21700_expt5_postlabel_audit.json'
OUT=HERE/'batterylife_zenodo_21700_expt5_exploratory_90eol.json'
API='https://zenodo.org/api/records/10637534/files/Expt%205%20-%20Standard%20Cycle%20Aging%20(Control).zip/content'

def life90():
    _, entries=io217.central_directory(API); out={}
    for c in 'ABCDEFGH':
        n=next(n for n in entries if 'Performance Summary' in n and f'cell {c} (' in n and n.endswith('Processed Data.csv'))
        rows=csv.DictReader(io.StringIO(io217.member_bytes(API,entries[n]).decode('utf8','replace')))
        vals=[float(r['Ageing Cycles']) for r in rows if r.get('SoH') not in ('',None) and r.get('Ageing Cycles') not in ('',None) and float(r['SoH'])<=.90]
        out[f'21700_expt5_cell_{c}']=min(vals)
    return out

def main():
    m=json.loads(MANIFEST.read_text()); life=life90(); results=[]
    for h in (10,20,50):
        cells=[{'name':c['name'],'life':life[c['name']],'protocol':c['protocol'],'curve':c['curve_h50_capacity_Ah'][:h]} for c in m['cells']]
        names=[c['name'] for c in cells]; y=np.asarray([c['life'] for c in cells],float); p=np.asarray([c['protocol'] for c in cells],float); q=np.asarray([c['curve'] for c in cells],float)
        dp=io217.distance_matrix(p,io217.robust_scale(p),1e9); dc=io217.distance_matrix(q,io217.robust_scale(q),1e9); dist={w:dc if math.isinf(w) else np.sqrt(dp**2+w*dc**2) for w in v1.WEIGHTS}; spread=v1.protocol_dispersion(dp)
        for setting in [s for s in m['settings'] if s['horizon']==h]:
            k=setting['label_budget_k']; store={'baseline':defaultdict(list),'method':defaultdict(list)}
            for ep in setting['episodes']:
                idx={n:i for i,n in enumerate(names)}; test=np.asarray([idx[n] for n in ep['test']]); bs=np.asarray([idx[n] for n in ep['baseline_support']]); ms=np.asarray([idx[n] for n in ep['method_support']])
                bp=ep['baseline_predictor']; mp=ep['method_predictor']
                for arm,pred,sup in [('baseline',bp,bs),('method',mp,ms)]:
                    z=v2.predict(pred,dist,y,sup,test)
                    for pos,i in enumerate(test):
                        e=abs(float(z[pos])-y[i]); store[arm][names[i]].append((e,100*e/max(y[i],1)))
            summary={}
            for arm in ('baseline','method'):
                vals=[np.asarray(store[arm][n]) for n in names]
                summary[arm]={'mae':float(np.mean([v[:,0].mean() for v in vals])),'mape':float(np.mean([v[:,1].mean() for v in vals]))}
            results.append({'horizon':h,'label_budget_k':k,'protocol_dispersion':spread,'route_counts':setting['route_counts'],'summary':summary,'relative_mape_reduction_percent':100*(summary['baseline']['mape']-summary['method']['mape'])/summary['baseline']['mape']})
    out={'phase':'EXPLORATORY_NONBLIND_90_PERCENT_EOL','manifest':MANIFEST.name,'label_audit':AUDIT.name,'eol_definition':'first performance-summary Ageing Cycles with SoH <= 0.90','results':results,'warning':'Dataset-local 90% EOL was used only after label-contract audit; this is not independent blind confirmation.'}
    OUT.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({f"h{r['horizon']}_k{r['label_budget_k']}":r['relative_mape_reduction_percent'] for r in results},indent=2))

if __name__=='__main__': main()
