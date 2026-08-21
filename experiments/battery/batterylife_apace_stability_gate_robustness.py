"""Targeted unlabeled stress tests for the stability-gate candidate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import batterylife_apace_stability_gate_candidate as candidate


HERE = Path(__file__).parent
ORIGINAL_LOAD = candidate.load_cells


def perturbed(dataset, horizon, mode):
    cells = ORIGINAL_LOAD(dataset, horizon)
    result = []
    for cell in cells:
        item = dict(cell)
        token = f"{dataset}|{horizon}|{cell['name']}|{mode}".encode()
        seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**32)
        rng = np.random.default_rng(seed)
        protocol = np.asarray(cell["protocol"], dtype=float).copy()
        curve = np.asarray(cell["curve"], dtype=float).copy()
        if mode.startswith("protocol_missing"):
            fraction = float(mode.replace("protocol_missing", "")) / 100.0
            mask = rng.random(protocol.size) < fraction
            protocol[mask] = np.nan
        elif mode.startswith("curve_noise"):
            fraction = float(mode.replace("curve_noise", "").replace("pct", "")) / 100.0
            finite = np.isfinite(curve)
            curve[finite] *= 1.0 + rng.normal(0.0, fraction, int(finite.sum()))
        else:
            raise KeyError(mode)
        item["protocol"], item["curve"] = protocol, curve
        result.append(item)
    return result


def main():
    for mode in ("protocol_missing25", "protocol_missing50", "curve_noise0.5pct"):
        candidate.load_cells = lambda dataset, horizon, _mode=mode: perturbed(dataset, horizon, _mode)
        output = HERE / f"batterylife_apace_stability_gate_{mode}.json"
        try:
            candidate.run((10, 20, 50), (3,), 100, output)
        finally:
            candidate.load_cells = ORIGINAL_LOAD
        data = json.loads(output.read_text())
        data["stress_mode"] = mode
        output.write_text(json.dumps(data, indent=2) + "\n")
        print(mode)
        for h in (10, 20, 50):
            row = data[f"macro_h{h}_k3"]
            print(h, row)


if __name__ == "__main__":
    main()
