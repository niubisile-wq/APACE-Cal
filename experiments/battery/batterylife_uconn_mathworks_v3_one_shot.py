"""One-shot, post-freeze MathWorks audit.  Labels are read only here."""
from __future__ import annotations
import argparse, hashlib, json, re, subprocess
from pathlib import Path
import numpy as np, pandas as pd
import batterylife_blind_manifest_eval as ev
import batterylife_uconn_mathworks_prelabel as pre

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def all_members(zips):
    out={}
    for z in zips:
        p=subprocess.Popen(['unzip','-Z1',str(z)],stdout=subprocess.PIPE,text=True,errors='ignore'); assert p.stdout
        for n in p.stdout:
            n=n.rstrip('\r\n'); m=re.search(r'Batch [^/]+/([^/]+)/cycling (\d+)\.csv$',n)
            if m: out.setdefault(m.group(1),[]).append((z,n,int(m.group(2))))
        p.wait()
    return out
def labels(zips):
    out={}
    for cell,parts in all_members(zips).items():
        life_max=0
        for z,n,num in parts:
            p=subprocess.Popen(['unzip','-p',str(z),n],stdout=subprocess.PIPE,stderr=subprocess.PIPE); assert p.stdout
            try:
                head=pd.read_csv(p.stdout,usecols=['Phase','Cycle'],nrows=2)
                if not head['Phase'].astype(str).str.contains('First life',case=False,na=False).any(): continue
                # A cycling file can contain multiple records but one life segment.
                p.kill() if p.poll() is None else None
                q=subprocess.Popen(['unzip','-p',str(z),n],stdout=subprocess.PIPE,stderr=subprocess.PIPE); assert q.stdout
                for ch in pd.read_csv(q.stdout,usecols=['Cycle'],chunksize=200000):
                    vals=pd.to_numeric(ch['Cycle'],errors='coerce');
                    if vals.notna().any(): life_max=max(life_max,float(vals.max()))
                q.stdout.close(); q.wait()
            finally:
                if p.stdout: p.stdout.close()
                p.wait()
        if life_max < 50: raise RuntimeError(f'bad label {cell}: {life_max}')
        out[f'MathWorks_{cell}']=life_max
    return out
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    m=json.loads(a.manifest.read_text())
    if m.get('phase')!='UCONN_MATHWORKS_V3_PRELABEL_FROZEN' or m.get('label_read') is not False: raise RuntimeError('manifest not pristine prelabel freeze')
    z=sorted(a.root.glob('batch*.zip')); mm=pre.members(z); labs=labels(z); cache={n:pre.read_cell(mm[n],50) for n in sorted(mm)}; loaded={}
    for h in (10,20,50): loaded[h]=[{'name':f'MathWorks_{n}','life':labs[f'MathWorks_{n}'],'protocol':np.asarray([25.,int(n.split('G')[1].split('C')[0]),int(n.split('C')[1]),float(cache[n][0][4]),float(cache[n][0][0]),float(cache[n][0][1])]),'curve':pre.curve_features(cache[n],h)} for n in sorted(mm) if pre.curve_features(cache[n],h) is not None]
    ev.predict=__import__('batterylife_asymmetric_cohort_router_v2',fromlist=['predict']).predict
    results=[ev.evaluate_setting(loaded[s['horizon']],s) for s in m['settings']]
    out={'phase':'UCONN_MATHWORKS_V3_ONE_SHOT_LABEL_OPENED','dataset':'UConn-MathWorks LFP/Gr','manifest_sha256':sha(a.manifest),'label_definition':'maximum first-life Cycle observed across first-life cycling members','label_count':len(labs),'active_settings_frozen_prelabel':m['active_settings'],'results':results}
    a.output.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({f"h{x['horizon']}_k{x['label_budget_k']}":{'mape':[x['baseline']['mape'],x['method']['mape']],'relative_reduction_percent':x['relative_mape_reduction_percent'],'cells':x['improved_same_worse_cells'],'p':x['paired_wilcoxon_p']} for x in results},indent=2))
if __name__=='__main__': main()
