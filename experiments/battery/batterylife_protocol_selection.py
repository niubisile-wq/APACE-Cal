"""Protocol-aware calibration-cell selection on the clean horizon predictions.

Selection uses filename-encoded operating temperature, SOC interval, and C-rate
only. No target life label or degradation feature is used for selecting cells.
"""
import json,re,random
from pathlib import Path
import numpy as np

BASE=Path(__file__).with_name('batterylife_clean_horizon.json')

def protocol(name):
    # Temperature: CALB_35_... or ..._25C_...
    m=re.search(r'^CALB_(-?\d+)_',name) or re.search(r'_(-?\d+)C_',name)
    temp=float(m.group(1)) if m else np.nan
    soc=re.search(r'_(-?\d+)-(-?\d+)_([0-9.]+)-([0-9.]+)C',name)
    if soc: return np.array([temp,float(soc.group(1)),float(soc.group(2)),float(soc.group(3)),float(soc.group(4))])
    return np.array([temp,np.nan,np.nan,np.nan,np.nan])

def distance(a,b,scale):
    mask=np.isfinite(a)&np.isfinite(b)&np.isfinite(scale)&(scale>0)
    if not mask.any(): return 0.
    return float(np.sqrt(np.sum(((a[mask]-b[mask])/scale[mask])**2)))

def main():
    x=json.load(open(BASE)); clean=x['rows']; rows=[]
    # raw prediction table, uniquely defined despite repetition over seed/method
    keys={}
    for r in clean:
        if r['method']=='raw': keys[(r['horizon'],r['source'],r['target'],r['model'],r['held_out'])]=(r['prediction'],r['truth'])
    groups={}
    for (h,s,t,m,n),(p,y) in keys.items(): groups.setdefault((h,s,t,m),[]).append((n,p,y))
    for (h,s,t,m),cells in groups.items():
        vec=np.asarray([protocol(n) for n,_,_ in cells]); scale=np.nanstd(vec,axis=0); scale[~np.isfinite(scale)|(scale==0)]=1.
        for i,(name,raw,y) in enumerate(cells):
          for seed in range(1,11):
            # Random tie-breaking is mandatory: identical protocols cannot be
            # distinguished label-free, so alphabetical ordering would be a
            # hidden favorable subset choice.
            avail=[j for j in range(len(cells)) if j!=i]; random.Random(10000*i+seed).shuffle(avail)
            chosen=sorted(avail,key=lambda j:distance(vec[i],vec[j],scale))[:min(3,len(avail))]
            pc=np.asarray([cells[j][1] for j in chosen]); yc=np.asarray([cells[j][2] for j in chosen])
            preds={'protocol_target_median':float(np.median(yc)),'protocol_bias':raw+float(np.median(yc-pc)),'protocol_scale':raw*float(np.median(yc/np.maximum(pc,1e-6)))}
            for method,p in preds.items():
                e=abs(p-y); rows.append({'horizon':h,'source':s,'target':t,'model':m,'held_out':name,'seed':seed,'selected':[cells[j][0] for j in chosen],'method':method,'prediction':float(p),'truth':y,'abs_error':float(e),'ape':float(100*e/max(y,1))})
    out={'base_protocol':str(BASE.name),'selection':'three nearest cells by standardized temperature/SOC/C-rate metadata; no labels used for selection','rows':rows,'summary':{}}
    for h,s,t,m in sorted(groups):
      for method in ['protocol_target_median','protocol_bias','protocol_scale']:
       z=[r for r in rows if (r['horizon'],r['source'],r['target'],r['model'],r['method'])==(h,s,t,m,method)]
       out['summary'][f'h{h}_{s}_to_{t}_{m}_{method}']={'mae':float(np.mean([r['abs_error'] for r in z])),'mae_std':float(np.std([r['abs_error'] for r in z],ddof=1)),'mape':float(np.mean([r['ape'] for r in z])),'n_evaluations':len(z)}
    path=Path(__file__).with_name('batterylife_protocol_selection.json'); path.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out['summary'],indent=2))
if __name__=='__main__':main()
