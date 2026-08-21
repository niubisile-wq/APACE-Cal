"""Unattainable test-aware support oracle; upper bound only, never a method claim."""
from __future__ import annotations
import argparse, copy, json, zipfile
from pathlib import Path
import numpy as np
import batterylife_blind_manifest_eval as ev
import batterylife_asymmetric_cohort_router_v2 as v2
import batterylife_batterylife_prelabel as pre

def labels(root):
    out={}
    with zipfile.ZipFile(root/'Life labels.zip') as z:
        for ds in ('HUST','XJTU'):
            out.update({f'{ds}::{Path(k).stem}':float(v) for k,v in json.loads(z.read(f'Life labels/{ds}_labels.json')).items()})
    return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();m=json.loads(a.manifest.read_text());labs=labels(a.root);loaded={h:[] for h in (10,20,50)}
    for ds in ('HUST','XJTU'):
        z=a.root/f'{ds}.zip'
        for n in pre.names(z):
            key=f'{ds}::{Path(n).stem}'; x=pre.load_member(z,n); cyc=sorted(x['cycle_data'],key=lambda c:c.get('cycle_number',0))
            for h in (10,20,50):
                f=pre.features(cyc,h)
                if f is not None: loaded[h].append({'name':key,'life':labs[key],'protocol':pre.protocol(x),'curve':f})
    ev.predict=v2.predict; rows=[]
    for s in m['settings']:
        cells=loaded[s['horizon']]; truth={c['name']:c['life'] for c in cells}; ss=copy.deepcopy(s)
        for e in ss['episodes']:
            target_mean=float(np.mean([truth[n] for n in e['test']])); ac=e['acquisition']; k=len(e['baseline_support']); chosen=sorted(ac,key=lambda n:abs(truth[n]-target_mean))[:k]; e['method_support']=chosen; e['method_predictor']=e['baseline_predictor']; e['route']='oracle_test_aware'
        r=ev.evaluate_setting(cells,ss);rows.append(r)
    out={'phase':'BATTERYLIFE_HUST_XJTU_TEST_AWARE_ORACLE','warning':'Uses held-out test labels to choose supports; unattainable upper bound only.','results':rows};a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({f"h{x['horizon']}_k{x['label_budget_k']}":{'baseline':x['baseline']['mape'],'oracle':x['method']['mape'],'reduction':x['relative_mape_reduction_percent']} for x in rows},indent=2))
if __name__=='__main__':main()
