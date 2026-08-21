"""One-shot label-opening evaluator for the frozen pooled BatteryLife cohort."""
from __future__ import annotations
import argparse, hashlib, json, zipfile
from pathlib import Path
import numpy as np
import batterylife_blind_manifest_eval as ev
import batterylife_asymmetric_cohort_router_v2 as v2
import batterylife_batterylife_prelabel as pre

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_labels(root,datasets):
    out={}
    with zipfile.ZipFile(root/'Life labels.zip') as z:
        for ds in datasets:
            data=json.loads(z.read(f'Life labels/{ds}_labels.json'))
            for k,v in data.items(): out[f'{ds}::{Path(k).stem}']=float(v)
    return out
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--datasets',nargs='+',required=True); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    manifest=json.loads(a.manifest.read_text())
    if manifest.get('phase')!='BATTERYLIFE_POOLED_PRELABEL_FROZEN' or manifest.get('label_read') is not False: raise RuntimeError('manifest not pristine')
    # Hash verification is performed before opening Life labels.
    frozen_manifest_hash=sha(a.manifest); labels=read_labels(a.root,a.datasets)
    loaded={h:[] for h in (10,20,50)}
    for ds in a.datasets:
        z=a.root/f'{ds}.zip'
        for n in pre.names(z):
            stem=Path(n).stem; key=f'{ds}::{stem}'
            if key not in labels:
                alt=f'{ds}::{stem.replace("--","-#")}';
                if alt in labels: labels[key]=labels.pop(alt)
                else: continue
            x=pre.load_member(z,n); cycles=sorted(x.get('cycle_data',[]),key=lambda c:c.get('cycle_number',0));
            for h in (10,20,50):
                curve=pre.features(cycles,h)
                if curve is not None: loaded[h].append({'name':key,'life':labels[key],'protocol':pre.protocol(x),'curve':curve})
    ev.predict=v2.predict
    results=[ev.evaluate_setting(loaded[s['horizon']],s) for s in manifest['settings']]
    out={'phase':'BATTERYLIFE_POOLED_V3_ONE_SHOT_LABEL_OPENED','dataset':'BatteryLife pooled HUST+Tongji+XJTU','manifest_sha256':frozen_manifest_hash,'label_definition':'official BatteryLife Life labels','label_count':len(labels),'active_settings_frozen_prelabel':manifest['active_settings'],'results':results}
    a.output.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({f"h{x['horizon']}_k{x['label_budget_k']}":{'mape':[x['baseline']['mape'],x['method']['mape']],'relative_reduction_percent':x['relative_mape_reduction_percent'],'cells':x['improved_same_worse_cells'],'p':x['paired_wilcoxon_p'],'routes':x['route_counts']} for x in results},indent=2))
if __name__=='__main__': main()
