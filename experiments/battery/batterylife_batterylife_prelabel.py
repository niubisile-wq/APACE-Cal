"""Label-blind prelabel reader for BatteryLife processed archives.

This script never opens the Life labels archive.  It uses only the processed
cell metadata and the first H cycles of each pickle member.
"""
from __future__ import annotations
import argparse, hashlib, json, pickle, re, subprocess, warnings
from pathlib import Path
import numpy as np
from batterylife_asymmetric_cohort_router_v2 import distance_concordance
from batterylife_transductive_pool_acquisition import distance_matrix
from batterylife_curve_aware_support import robust_scale

def names(z):
    p=subprocess.Popen(['unzip','-Z1',str(z)],stdout=subprocess.PIPE,text=True,errors='ignore'); assert p.stdout
    out=[]
    for n in p.stdout:
        n=n.rstrip('\r\n')
        if n.endswith('.pkl'): out.append(n)
    p.wait(); return sorted(out)

def load_member(z,n):
    p=subprocess.Popen(['unzip','-p',str(z),n],stdout=subprocess.PIPE,stderr=subprocess.PIPE); assert p.stdout
    try: return pickle.load(p.stdout)
    finally:
        p.stdout.close(); err=p.stderr.read() if p.stderr else b''; rc=p.wait()
        if rc not in (0,-13,13,141): raise RuntimeError(f'unzip failed {n}: {err[-300:]!r}')

def scalar(v):
    if isinstance(v,(int,float,np.integer,np.floating)) and np.isfinite(v): return float(v)
    if isinstance(v,list):
        vals=[]
        for x in v:
            if isinstance(x,dict): vals.extend([float(y) for y in x.values() if isinstance(y,(int,float)) and np.isfinite(y)])
            elif isinstance(x,(int,float)) and np.isfinite(x): vals.append(float(x))
        return float(np.median(vals)) if vals else 0.0
    return 0.0

def signature(c, nominal):
    def arr(k):
        v=c.get(k); return np.asarray(v if v is not None else [],float)
    qch,qdc=arr('charge_capacity_in_Ah'),arr('discharge_capacity_in_Ah'); v=arr('voltage_in_V'); i=arr('current_in_A'); t=arr('time_in_s')
    def mx(a): return float(np.nanmax(a)) if np.isfinite(a).any() else np.nan
    return np.asarray([mx(qdc)/max(nominal,1e-9),mx(qch)/max(nominal,1e-9),float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan,float(np.nanmax(v)-np.nanmin(v)) if np.isfinite(v).any() else np.nan,float(np.nanmean(np.abs(i)))/max(nominal,1e-9) if np.isfinite(i).any() else np.nan,float(np.log1p(np.nanmax(t)-np.nanmin(t))) if np.isfinite(t).any() else np.nan,scalar(c.get('temperature_in_C')),scalar(c.get('internal_resistance_in_ohm'))],float)

def features(cycles,h):
    if len(cycles)<h:return None
    s=np.asarray([signature(c,1.0) for c in cycles[:h]],float); pos=np.rint(np.linspace(0,h-1,5)).astype(int); x=np.linspace(0,1,h); slopes=[]
    for j in range(s.shape[1]):
        ok=np.isfinite(s[:,j]); slopes.append(float(np.polyfit(x[ok],s[ok,j],1)[0]) if ok.sum()>=3 else np.nan)
    return np.r_[s[pos].reshape(-1),slopes,s[-1]-s[0]]

def protocol(x):
    # Only pre-cycle metadata; already_spent_cycles and all life-derived fields are excluded.
    group=float(re.search(r'_(\d+)-',str(x.get('cell_id','0'))).group(1)) if re.search(r'_(\d+)-',str(x.get('cell_id','0'))) else 0.0
    def rate_list(v):
        vals=[]
        if isinstance(v,list):
            for q in v:
                if isinstance(q,dict):
                    z=q.get('rate_in_C')
                    if isinstance(z,(int,float)) and np.isfinite(z): vals.append(float(z))
        return vals
    cr,dr=rate_list(x.get('charge_protocol')),rate_list(x.get('discharge_protocol'))
    cid=str(x.get('cell_id',''))
    # Dataset-specific protocol strings are part of the public pre-cycle
    # metadata, not labels: Tongji encodes CY25-025/CY25-05 and XJTU encodes 2C/3C.
    m=re.search(r'CY25-([0-9]+)',cid); tongji_rate=(float(m.group(1))/100.0 if m else 0.0)
    m=re.search(r'(\d+(?:\.\d+)?)C',cid); xjtu_rate=(float(m.group(1)) if m else 0.0)
    return np.asarray([scalar(x.get('nominal_capacity_in_Ah')),scalar(x.get('depth_of_charge')),scalar(x.get('depth_of_discharge')),scalar(x.get('max_voltage_limit_in_V')),scalar(x.get('min_voltage_limit_in_V')),group,cr[0] if cr else 0.,dr[0] if dr else 0.,cr[1] if len(cr)>1 else 0.,dr[1] if len(dr)>1 else 0.,tongji_rate,xjtu_rate],float)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--datasets',nargs='+',required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args(); rows=[]
    for ds in a.datasets:
        z=a.root/f'{ds}.zip'
        for n in names(z):
            x=load_member(z,n); cycles=sorted(x.get('cycle_data',[]),key=lambda c:c.get('cycle_number',0));
            if len(cycles)<50: continue
            fs={str(h):features(cycles,h) for h in (10,20,50)}; rows.append({'dataset':ds,'name':Path(n).stem,'archive_member':n,'protocol':protocol(x).tolist(),'horizons':{h:fs[h] is not None for h in fs},'curve_sha256':{h:hashlib.sha256(np.nan_to_num(fs[h],nan=0.).tobytes()).hexdigest() for h in fs}})
    out={'phase':'BATTERYLIFE_PRELABEL_V1','label_read':False,'datasets':a.datasets,'cells':rows,'n_cells':len(rows),'explicit_non_access_statement':'Only processed pickle metadata and first H cycle_data were read; Life labels archive was not opened.'}
    for ds in a.datasets:
        rr=[r for r in rows if r['dataset']==ds]
        for h in ('10','20','50'):
            e=[r for r in rr if r['horizons'][h]]; p=np.asarray([r['protocol'] for r in e],float); dp=distance_matrix(p,robust_scale(p),1e9); spread=float(np.median(dp[np.triu_indices(len(dp),1)])) if len(e)>1 else float('nan'); out[f'{ds}_h{h}']={'eligible_cells':len(e),'protocol_dispersion':spread}
    a.output.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n'); print(json.dumps({k:v for k,v in out.items() if k.startswith(tuple(a.datasets))},indent=2))
if __name__=='__main__': main()
