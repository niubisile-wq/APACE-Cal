"""Nested audit of early-curve-aware support retrieval for PASS-Cal.

The outer target dataset is never used to choose the protocol/curve distance
weight.  For every held-out target cell, support identities are chosen from
metadata and the first H cycles only; cycle-life labels are read only after
selection.  Each cell contributes one prediction, so there is no checkpoint
pseudo-replication.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from batterylife_protocol_selection import protocol


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "batterylife_processed"
DATASETS = ("CALB", "HNEI", "MICH_EXP", "CALCE", "MICH", "SNL")
HORIZONS = (10, 20, 50)
KS = (1, 3, 5, 10)
WEIGHTS = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, math.inf)


def finite_array(value) -> np.ndarray:
    """Return finite numeric values without treating None as zero."""
    if value is None:
        return np.empty(0)
    try:
        values = np.asarray(value, dtype=object).reshape(-1)
    except Exception:
        values = np.asarray([value], dtype=object)
    output = []
    for item in values:
        if item is None:
            continue
        try:
            item = float(item)
        except (TypeError, ValueError):
            continue
        if np.isfinite(item):
            output.append(item)
    return np.asarray(output, dtype=float)


def aggregate(value, fn, default=np.nan) -> float:
    values = finite_array(value)
    return float(fn(values)) if values.size else float(default)


def cycle_signature(cycle: dict, nominal_capacity: float) -> np.ndarray:
    qd = aggregate(cycle.get("discharge_capacity_in_Ah"), np.max)
    qc = aggregate(cycle.get("charge_capacity_in_Ah"), np.max)
    voltage = finite_array(cycle.get("voltage_in_V"))
    current = finite_array(cycle.get("current_in_A"))
    time = finite_array(cycle.get("time_in_s"))
    temp = finite_array(cycle.get("temperature_in_C"))
    resistance = finite_array(cycle.get("internal_resistance_in_ohm"))
    scale = nominal_capacity if np.isfinite(nominal_capacity) and nominal_capacity > 0 else np.nan
    return np.asarray(
        [
            qd / scale,
            qc / scale,
            np.nanmedian(voltage) if voltage.size else np.nan,
            (np.nanmax(voltage) - np.nanmin(voltage)) if voltage.size else np.nan,
            np.nanmean(np.abs(current)) / scale if current.size else np.nan,
            np.log1p(np.nanmax(time) - np.nanmin(time)) if time.size else np.nan,
            np.nanmedian(temp) if temp.size else np.nan,
            np.log1p(np.nanmedian(np.abs(resistance))) if resistance.size else np.nan,
        ],
        dtype=float,
    )


def robust_slope(values: np.ndarray) -> float:
    ok = np.isfinite(values)
    if ok.sum() < 3:
        return np.nan
    x = np.linspace(0.0, 1.0, len(values))[ok]
    return float(np.polyfit(x, values[ok], 1)[0])


def curve_features(data: dict, horizon: int) -> np.ndarray | None:
    cycles = data.get("cycle_data", [])
    if len(cycles) < horizon:
        return None
    nominal = aggregate(data.get("nominal_capacity_in_Ah"), np.median)
    signatures = np.asarray([cycle_signature(c, nominal) for c in cycles[:horizon]])
    # Five fixed relative observations preserve shape without making longer
    # horizons contribute more dimensions.
    positions = np.rint(np.linspace(0, horizon - 1, 5)).astype(int)
    snapshots = signatures[positions].reshape(-1)
    slopes = np.asarray([robust_slope(signatures[:, j]) for j in range(signatures.shape[1])])
    deltas = signatures[-1] - signatures[0]
    return np.r_[snapshots, slopes, deltas]


def protocol_rate(data: dict, key: str, nominal_capacity: float) -> float:
    entries = data.get(key, [])
    if isinstance(entries, dict):
        entries = [entries]
    values = []
    if isinstance(entries, (list, tuple)):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            rate = aggregate(entry.get("rate_in_C"), np.median)
            if np.isfinite(rate) and abs(rate) > 1e-12:
                values.append(abs(rate))
                continue
            current = aggregate(entry.get("current_in_A"), np.median)
            if (
                np.isfinite(current)
                and abs(current) > 1e-12
                and np.isfinite(nominal_capacity)
                and nominal_capacity > 0
            ):
                values.append(abs(current) / nominal_capacity)
    return float(np.median(values)) if values else np.nan


def structured_protocol(data: dict, name: str, horizon: int) -> np.ndarray:
    """Fill filename protocol fields from structured BatteryML metadata."""
    output = np.asarray(protocol(name), dtype=float).copy()
    cycles = data.get("cycle_data", [])[:horizon]
    temperatures = np.concatenate(
        [finite_array(cycle.get("temperature_in_C")) for cycle in cycles]
    ) if cycles else np.empty(0)
    # Cycle sensor temperature is authoritative.  Filename tokens such as
    # ``_2C_`` can denote a C-rate and were previously misread as 2 degrees C.
    if temperatures.size:
        output[0] = float(np.median(temperatures))
    soc = finite_array(data.get("SOC_interval"))
    if soc.size >= 2:
        soc = soc[:2] * (100.0 if np.nanmax(np.abs(soc[:2])) <= 1.5 else 1.0)
        if not np.isfinite(output[1]):
            output[1] = float(soc[0])
        if not np.isfinite(output[2]):
            output[2] = float(soc[1])
    nominal = aggregate(data.get("nominal_capacity_in_Ah"), np.median)
    if not np.isfinite(output[3]):
        output[3] = protocol_rate(data, "charge_protocol", nominal)
    if not np.isfinite(output[4]):
        output[4] = protocol_rate(data, "discharge_protocol", nominal)
    return output


def load_cells(dataset: str, horizon: int) -> list[dict]:
    labels = json.load(open(BASE / "Life labels" / f"{dataset}_labels.json"))
    cells = []
    for path in sorted((BASE / dataset).glob("*.pkl")):
        if path.name not in labels:
            continue
        life = float(labels[path.name])
        if life < horizon:
            continue
        data = pickle.load(open(path, "rb"))
        features = curve_features(data, horizon)
        if features is None:
            continue
        cells.append(
            {
                "name": path.name,
                "life": life,
                "protocol": structured_protocol(data, path.name, horizon),
                "curve": features,
            }
        )
    return cells


def robust_scale(matrix: np.ndarray) -> np.ndarray:
    # Column-wise handling avoids NumPy's noisy all-NaN warnings while keeping
    # missing sensor channels explicitly neutral in nan_rms().
    scale = np.ones(matrix.shape[1], dtype=float)
    for j in range(matrix.shape[1]):
        values = matrix[:, j]
        values = values[np.isfinite(values)]
        if not values.size:
            continue
        candidate = np.percentile(values, 75) - np.percentile(values, 25)
        if not np.isfinite(candidate) or candidate <= 1e-12:
            candidate = np.std(values)
        if np.isfinite(candidate) and candidate > 1e-12:
            scale[j] = candidate
    return scale


def nan_rms(a: np.ndarray, b: np.ndarray, scale: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(scale) & (scale > 0)
    if not ok.any():
        return 0.0
    return float(np.sqrt(np.mean(np.square((a[ok] - b[ok]) / scale[ok]))))


def combined_distance(dp: float, dc: float, weight: float) -> float:
    if math.isinf(weight):
        return dc
    return float(np.sqrt(dp * dp + weight * dc * dc))


def fit_local_log_ridge(query: np.ndarray, support_x: np.ndarray, support_y: np.ndarray, scale: np.ndarray) -> float:
    def transform(row):
        normalized = row / scale
        return np.r_[np.nan_to_num(normalized, nan=0.0), (~np.isfinite(row)).astype(float)]

    x = np.asarray([transform(row) for row in support_x])
    q = transform(query).reshape(1, -1)
    model = Ridge(alpha=10.0).fit(x, np.log(support_y))
    return float(np.exp(model.predict(q)[0]))


def evaluate_domain(cells: list[dict], k: int, weight: float, seeds: int, curve_mask=None) -> list[dict]:
    p = np.asarray([c["protocol"] for c in cells], dtype=float)
    c = np.asarray([x["curve"] for x in cells], dtype=float)
    if curve_mask is not None:
        c = c[:, np.asarray(curve_mask, dtype=int)]
    p_scale, c_scale = robust_scale(p), robust_scale(c)
    rows = []
    prediction_cache = {}
    for held, query in enumerate(cells):
        available = [i for i in range(len(cells)) if i != held]
        dp = np.asarray([nan_rms(p[held], p[j], p_scale) for j in available])
        dc = np.asarray([nan_rms(c[held], c[j], c_scale) for j in available])
        distance = np.asarray([combined_distance(a, b, weight) for a, b in zip(dp, dc)])
        for seed in range(1, seeds + 1):
            rng = random.Random(10_000_000 * held + 10_000 * k + seed)
            tie = list(range(len(available)))
            rng.shuffle(tie)
            tie_rank = np.empty(len(tie), dtype=int)
            tie_rank[tie] = np.arange(len(tie))
            order = np.lexsort((tie_rank, distance))
            chosen = [available[j] for j in order[: min(k, len(order))]]
            # Keep the established calibrator fixed so this experiment isolates
            # the value of curve-aware retrieval.
            cache_key = (held, tuple(chosen))
            if cache_key not in prediction_cache:
                prediction_cache[cache_key] = fit_local_log_ridge(
                    p[held], p[chosen], np.asarray([cells[j]["life"] for j in chosen]), p_scale
                )
            pred = prediction_cache[cache_key]
            truth = query["life"]
            error = abs(pred - truth)
            rows.append(
                {
                    "held_out": query["name"],
                    "seed": seed,
                    "selected": [cells[j]["name"] for j in chosen],
                    "prediction": pred,
                    "truth": truth,
                    "abs_error": error,
                    "ape": 100.0 * error / max(truth, 1.0),
                }
            )
    return rows


def summarize(rows: list[dict]) -> dict:
    return {
        "mae": float(np.mean([r["abs_error"] for r in rows])),
        "mape": float(np.mean([r["ape"] for r in rows])),
        "median_ae": float(np.median([r["abs_error"] for r in rows])),
        "n_evaluations": len(rows),
    }


def paired_cell_summary(baseline_rows: list[dict], method_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(lambda: {"baseline_ae": [], "baseline_ape": [], "method_ae": [], "method_ape": []})
    for row in baseline_rows:
        grouped[row["held_out"]]["baseline_ae"].append(row["abs_error"])
        grouped[row["held_out"]]["baseline_ape"].append(row["ape"])
    for row in method_rows:
        grouped[row["held_out"]]["method_ae"].append(row["abs_error"])
        grouped[row["held_out"]]["method_ape"].append(row["ape"])
    return [
        {
            "held_out": name,
            **{key: float(np.mean(values)) for key, values in record.items()},
        }
        for name, record in sorted(grouped.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()

    all_cells = {(h, d): load_cells(d, h) for h in HORIZONS for d in DATASETS}
    cache: dict[tuple[int, str, int, float], list[dict]] = {}
    domain_scores: dict[tuple[int, str, int, float], dict] = {}
    for horizon in HORIZONS:
        for dataset in DATASETS:
            for k in KS:
                for weight in WEIGHTS:
                    key = (horizon, dataset, k, weight)
                    cache[key] = evaluate_domain(all_cells[(horizon, dataset)], k, weight, args.seeds)
                    domain_scores[key] = summarize(cache[key])

    nested = []
    for horizon in HORIZONS:
        for target in DATASETS:
            development = [d for d in DATASETS if d != target]
            for k in KS:
                candidates = []
                for weight in WEIGHTS:
                    scores = [domain_scores[(horizon, d, k, weight)] for d in development]
                    candidates.append(
                        (
                            float(np.mean([s["mape"] for s in scores])),
                            float(np.mean([s["mae"] for s in scores])),
                            weight,
                        )
                    )
                _, _, selected_weight = min(candidates)
                test_rows = cache[(horizon, target, k, selected_weight)]
                baseline_rows = cache[(horizon, target, k, 0.0)]
                nested.append(
                    {
                        "horizon": horizon,
                        "target": target,
                        "k": k,
                        "selected_curve_weight": "inf" if math.isinf(selected_weight) else selected_weight,
                        "selection_candidates": [
                            {"weight": "inf" if math.isinf(w) else w, "development_macro_mape": m, "development_macro_mae": a}
                            for m, a, w in candidates
                        ],
                        "protocol_only": summarize(baseline_rows),
                        "nested_curve_aware": summarize(test_rows),
                        "per_cell": paired_cell_summary(baseline_rows, test_rows),
                    }
                )

    output = {
        "dataset_version": "BatteryLife v12",
        "protocol": "outer leave-one-dataset-out weight selection; inner cell-level leave-one-out; support selected without life labels; one prediction per cell",
        "seeds": args.seeds,
        "audit": {f"h{h}_{d}": len(all_cells[(h, d)]) for h in HORIZONS for d in DATASETS},
        "nested_results": nested,
    }
    for horizon in HORIZONS:
        for k in KS:
            selected = [r for r in nested if r["horizon"] == horizon and r["k"] == k]
            output[f"macro_h{horizon}_k{k}"] = {
                "protocol_mae": float(np.mean([r["protocol_only"]["mae"] for r in selected])),
                "curve_aware_mae": float(np.mean([r["nested_curve_aware"]["mae"] for r in selected])),
                "protocol_mape": float(np.mean([r["protocol_only"]["mape"] for r in selected])),
                "curve_aware_mape": float(np.mean([r["nested_curve_aware"]["mape"] for r in selected])),
            }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k.startswith("macro_") or k == "audit"}, indent=2))


if __name__ == "__main__":
    main()
