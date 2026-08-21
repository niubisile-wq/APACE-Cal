"""Pre-label manifest builder for the UConn-ISU-ILCC external candidate.

Only cycling ZIP members and the public protocol-parameter CSV are read here;
the RPT archive (which contains capacity/life information) is deliberately not
opened.  The script streams the first 50 cycles needed by the frozen feature
contract and records an unlabeled manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from batterylife_asymmetric_cohort_router_v2 import distance_concordance
from batterylife_curve_aware_support import robust_scale
from batterylife_transductive_pool_acquisition import distance_matrix


DEFAULT_ROOT = Path("/autodl-fs/data/battery_external_uconn_isu")
CELL_RE = re.compile(r"(?:cycling_)?cell[_-](\d{2})", re.I)


def signature(frame: pd.DataFrame, nominal=1.2) -> np.ndarray:
    state = frame["State"].astype(str).str.lower()
    dchg = frame[state.str.contains("dchg", na=False)]
    chg = frame[state.str.contains("chg", na=False) & ~state.str.contains("dchg", na=False)]
    def qmax(x):
        v = pd.to_numeric(x["Capacity (Ah)"], errors="coerce").to_numpy(float)
        return float(np.nanmax(v)) if np.isfinite(v).any() else np.nan
    voltage = pd.to_numeric(frame["Voltage (V)"], errors="coerce").to_numpy(float)
    current = pd.to_numeric(frame["Current (A)"], errors="coerce").to_numpy(float)
    tm = pd.to_numeric(frame["Time (s)"], errors="coerce").to_numpy(float)
    vf = voltage[np.isfinite(voltage)]
    cf = current[np.isfinite(current)]
    tf = tm[np.isfinite(tm)]
    return np.asarray([
        qmax(dchg) / nominal, qmax(chg) / nominal,
        float(np.nanmedian(vf)) if vf.size else np.nan,
        float(np.nanmax(vf) - np.nanmin(vf)) if vf.size else np.nan,
        float(np.nanmean(np.abs(cf))) / nominal if cf.size else np.nan,
        float(np.log1p(np.nanmax(tf) - np.nanmin(tf))) if tf.size else np.nan,
        np.nan, np.nan,
    ], dtype=float)


def curve_features(signatures: list[np.ndarray], horizon: int) -> np.ndarray | None:
    if len(signatures) < horizon:
        return None
    s = np.asarray(signatures[:horizon], dtype=float)
    pos = np.rint(np.linspace(0, horizon - 1, 5)).astype(int)
    slopes = []
    for j in range(s.shape[1]):
        ok = np.isfinite(s[:, j])
        slopes.append(float(np.polyfit(np.linspace(0, 1, horizon)[ok], s[ok, j], 1)[0]) if ok.sum() >= 3 else np.nan)
    return np.r_[s[pos].reshape(-1), np.asarray(slopes), s[-1] - s[0]]


def params(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = csv.DictReader(f)
        out = {}
        for row in rows:
            cid = f"{int(row['Cell ID']):02d}"
            out[cid] = {
                "group": int(row["Group Number"]),
                "protocol": np.asarray([
                    25.0, float(row["DoD 1"]) * 100.0, 0.0,
                    float(row["Chg C-Rate 1"]), float(row["DChg C-Rate 1"])
                ], dtype=float),
            }
        return out


def member_map(zips: list[Path]) -> dict[str, list[tuple[Path, str]]]:
    out: dict[str, list[tuple[Path, str]]] = {}
    for path in zips:
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if not name.lower().endswith(".csv"):
                    continue
                m = CELL_RE.search(Path(name).name)
                if m:
                    out.setdefault(m.group(1), []).append((path, name))
    for key in out:
        out[key].sort(key=lambda x: x[1])
    return out


def read_cell(parts, horizon=50):
    signatures = {}
    usecols = ["Cycle Number", "State", "Voltage (V)", "Current (A)", "Time (s)", "Capacity (Ah)"]
    for archive, member in parts:
        # The UConn ZIPs use enhanced-64K deflate (method 9), which the
        # standard Python zipfile reader does not implement.  The system
        # unzip utility supports it and lets us keep the stream unextracted.
        proc = subprocess.Popen(
            ["unzip", "-p", str(archive), member],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.stdout is not None
        stream = proc.stdout
        stopped_early = False
        try:
            for chunk in pd.read_csv(stream, usecols=usecols, chunksize=200_000):
                cycles = pd.to_numeric(chunk["Cycle Number"], errors="coerce")
                for number, frame in chunk.assign(_cycle=cycles).groupby("_cycle", sort=False):
                    if not np.isfinite(number) or number > horizon:
                        continue
                    signatures.setdefault(int(number), []).append(signature(frame))
                if len(signatures) >= horizon and max(signatures) >= horizon - 1:
                    stopped_early = True
                    break
        finally:
            stream.close()
            stderr = proc.stderr.read() if proc.stderr is not None else b""
            return_code = proc.wait()
            if return_code != 0 and not (signatures and return_code in (-13, 13, 141)):
                raise RuntimeError(f"unzip failed for {member}: {stderr[-500:]!r}")
        if len(signatures) >= horizon and max(signatures) >= horizon - 1:
            break
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ordered = [np.nanmean(signatures[k], axis=0) for k in sorted(signatures)[:horizon]]
    return ordered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--output", type=Path, default=DEFAULT_ROOT / "uconn_isu_prelabel_manifest_v3.json")
    args = ap.parse_args()
    root = args.root
    p = params(root / "cycling_parameters.csv")
    archives = [root / "cycling_part1.zip", root / "cycling_part2.zip"]
    members = member_map(archives)
    rows = []
    signature_cache = {}
    for cid in sorted(p):
        sig = read_cell(members.get(cid, []), 50)
        signature_cache[cid] = sig
        features = {str(h): curve_features(sig, h) for h in (10, 20, 50)}
        rows.append({
            "cell_id": cid, "group": p[cid]["group"],
            "protocol": p[cid]["protocol"].tolist(),
            "horizons": {h: (features[str(h)] is not None) for h in (10, 20, 50)},
            "curve_sha256": {
                str(h): hashlib.sha256(np.nan_to_num(features[str(h)], nan=0.0).tobytes()).hexdigest()
                if features[str(h)] is not None else None for h in (10, 20, 50)
            },
        })
    output = {"dataset": "UConn-ISU-ILCC LFP/Gr", "label_read": False,
              "cells": rows, "n_cells": len(rows),
              "protocol": "cycling ZIP only; RPT archive unopened"}
    for h in (10, 20, 50):
        eligible = [r for r in rows if r["horizons"][h]]
        matrix = np.asarray([r["protocol"] for r in eligible], dtype=float)
        dp = distance_matrix(matrix, robust_scale(matrix), 1e9)
        spread = float(np.median(dp[np.triu_indices(len(dp), 1)]))
        # Curves are reconstructed again only in-memory for rho; no labels.
        curves = []
        for cid in [r["cell_id"] for r in eligible]:
            curves.append(curve_features(signature_cache[cid], h))
        cmat = np.asarray(curves, dtype=float)
        rho_cells = [{"protocol": r["protocol"], "curve": cmat[i]} for i, r in enumerate(eligible)]
        output[f"h{h}"] = {"eligible_cells": len(eligible), "protocol_dispersion": spread,
                            "rho": distance_concordance(rho_cells),
                            "active_k3": bool(spread > 0.30 and not (0.30 <= spread < 0.60)),
                            "active_k5": bool(spread < 0.30)}
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k.startswith("h")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
