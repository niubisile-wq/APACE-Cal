"""Label-blind prelabel reader for the UConn-MathWorks second-life archive.

Only first-life cycling members are streamed.  RPT_note.csv and any RPT-like
members are intentionally not opened in this phase.
"""
from __future__ import annotations
import argparse, hashlib, json, re, subprocess, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from batterylife_asymmetric_cohort_router_v2 import distance_concordance
from batterylife_transductive_pool_acquisition import distance_matrix
from batterylife_curve_aware_support import robust_scale

CELL_RE = re.compile(r"Batch [^/]+/([^/]+)/cycling 1\.csv$")

def members(archives):
    out = {}
    for z in archives:
        proc = subprocess.Popen(["unzip", "-Z1", str(z)], stdout=subprocess.PIPE, text=True, errors="ignore")
        assert proc.stdout is not None
        for n in proc.stdout:
            n = n.rstrip("\r\n")
            m = CELL_RE.search(n)
            if m: out.setdefault(m.group(1), []).append((z, n))
        proc.wait()
    return {k: sorted(v) for k,v in out.items()}

def signature(frame, nominal=4.9):
    state = frame["State"].astype(str).str.lower()
    chg = frame[state.str.contains("chg", na=False) & ~state.str.contains("dchg", na=False)]
    dchg = frame[state.str.contains("dchg", na=False)]
    def qmax(x):
        a = pd.to_numeric(x["Capacity(Ah)"], errors="coerce").to_numpy(float)
        return float(np.nanmax(a)) if np.isfinite(a).any() else np.nan
    v = pd.to_numeric(frame["Voltage(V)"], errors="coerce").to_numpy(float)
    i = pd.to_numeric(frame["Current(A)"], errors="coerce").to_numpy(float)
    return np.asarray([qmax(dchg)/nominal, qmax(chg)/nominal,
        float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan,
        float(np.nanmax(v)-np.nanmin(v)) if np.isfinite(v).any() else np.nan,
        float(np.nanmean(np.abs(i)))/nominal if np.isfinite(i).any() else np.nan,
        float(np.log1p(len(frame))), np.nan, np.nan], float)

def read_cell(parts, horizon=50):
    sigs={}; use=["State","Cycle","Current(A)","Voltage(V)","Capacity(Ah)"]
    for z,n in parts:
        p=subprocess.Popen(["unzip","-p",str(z),n],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            for chunk in pd.read_csv(p.stdout,usecols=use,chunksize=200000):
                for cyc,g in chunk.groupby(pd.to_numeric(chunk["Cycle"],errors="coerce"),sort=False):
                    if np.isfinite(cyc) and 1 <= cyc <= horizon: sigs.setdefault(int(cyc),[]).append(signature(g))
                if len(sigs)>=horizon: break
        finally:
            if p.stdout: p.stdout.close()
            err=p.stderr.read() if p.stderr else b''; rc=p.wait()
            if rc not in (0,-13,13,141) and not sigs: raise RuntimeError(f"unzip failed {n}: {err[-200:]!r}")
        if len(sigs)>=horizon: break
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        return [np.nanmean(sigs[k],axis=0) for k in sorted(sigs)[:horizon]]

def curve_features(sigs,h):
    if len(sigs)<h:return None
    s=np.asarray(sigs[:h],float); pos=np.rint(np.linspace(0,h-1,5)).astype(int); x=np.linspace(0,1,h)
    slopes=[]
    for j in range(s.shape[1]):
        ok=np.isfinite(s[:,j]); slopes.append(float(np.polyfit(x[ok],s[ok,j],1)[0]) if ok.sum()>=3 else np.nan)
    return np.r_[s[pos].reshape(-1),slopes,s[-1]-s[0]]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    zips=sorted(a.root.glob('batch*.zip')); mm=members(zips); rows=[]; cache={}
    for name in sorted(mm):
        # Pre-registered protocol proxy: ambient condition, group, charge/discharge
        # current and first-cycle capacity.  No end-of-life information is used.
        sig=read_cell(mm[name],50); cache[name]=sig; f=curve_features(sig,50)
        if f is None: continue
        g=int(re.search(r'G(\d+)',name).group(1)); c=int(re.search(r'C(\d+)',name).group(1))
        first=np.asarray(sig[0],float); proto=np.asarray([25.,float(g),float(c),first[4],first[0],first[1]],float)
        rows.append({'cell_id':name,'group':g,'protocol':proto.tolist(),'horizons':{str(h):curve_features(sig,h) is not None for h in (10,20,50)},'curve_sha256':{str(h):hashlib.sha256(np.nan_to_num(curve_features(sig,h),nan=0.).tobytes()).hexdigest() for h in (10,20,50)}})
    out={'dataset':'UConn-MathWorks LFP/Gr','label_read':False,'cells':rows,'n_cells':len(rows),'protocol':'first-life cycling members only; RPT_note and RPT-like members unopened'}
    for h in (10,20,50):
        e=[r for r in rows if r['horizons'][str(h)]]; p=np.asarray([r['protocol'] for r in e]); dp=distance_matrix(p,robust_scale(p),1e9); spread=float(np.median(dp[np.triu_indices(len(dp),1)])) if len(e)>1 else float('nan'); curves=[curve_features(cache[r['cell_id']],h) for r in e]; rho=distance_concordance([{'protocol':r['protocol'],'curve':curves[i]} for i,r in enumerate(e)])
        out[f'h{h}']={'eligible_cells':len(e),'protocol_dispersion':spread,'rho':rho,'active_k3':bool(spread>0.30 and not (0.30<=spread<0.60))}
    a.output.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n'); print(json.dumps({k:v for k,v in out.items() if k.startswith('h')},indent=2))
if __name__=='__main__': main()
