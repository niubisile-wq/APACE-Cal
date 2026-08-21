"""Fair K-label calibration audit for the official PBT ensemble.

The released PBT checkpoints are used only on pretraining-unseen CALB and
NA-ion cells covered by the released prompt embeddings.  For each held-out
cell, curve-aware retrieval selects K other target cells without life labels.
The same selected cells then calibrate the PBT ensemble with a median log-bias
and fit the local PASS-Cal predictor.  Thus both competitors receive identical
target labels and identical retrieval information.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from batterylife_curve_aware_naion_frozen import frozen_weight
from batterylife_curve_aware_support import evaluate_domain, load_cells, summarize


HERE = Path(__file__).parent
PBT_SOURCE = HERE / "pbt_official_unseen_eval.json"
DEV_SOURCE = HERE / "batterylife_curve_aware_support.json"
ADAPTIVE_SOURCE = HERE / "batterylife_adaptive_effective_support.json"
OUTPUT = Path(__file__).with_suffix(".json")
HORIZONS = (10, 20, 50)
KS = (1, 3, 5, 10)


def decode_weight(value):
    return math.inf if value == "inf" else float(value)


def selected_rule(adaptive, dataset, horizon, k):
    if dataset == "CALB":
        record = next(
            r
            for r in adaptive["results"]
            if r["target"] == "CALB" and r["horizon"] == horizon and r["budget_k"] == k
        )
        chosen = record["selected"]
        return int(chosen["effective_k"]), decode_weight(chosen["weight"]), "outer LODO development domains"
    records = [
        r for r in adaptive["results"] if r["horizon"] == horizon and r["budget_k"] == k
    ]
    scores = defaultdict(list)
    for record in records:
        for candidate in record["selection_candidates"]:
            key = (int(candidate["effective_k"]), str(candidate["weight"]))
            scores[key].append(
                (candidate["development_macro_mape"], candidate["development_macro_mae"])
            )
    ranked = []
    for (effective_k, weight_text), values in scores.items():
        ranked.append(
            (
                float(np.mean([value[0] for value in values])),
                float(np.mean([value[1] for value in values])),
                effective_k,
                decode_weight(weight_text),
            )
        )
    _, _, effective_k, weight = min(ranked)
    return effective_k, weight, "six-domain-frozen external adaptive rule"


def ensemble_predictions(pbt, dataset, horizon):
    runs = [r for r in pbt["runs"] if r["dataset"] == dataset and r["horizon"] == horizon]
    if len(runs) != 3:
        raise RuntimeError(f"Expected three PBT checkpoints for {dataset} H={horizon}")
    by_file = defaultdict(list)
    checkpoint_maps = []
    for run in runs:
        mapping = {row["file"]: float(row["prediction"]) for row in run["rows"]}
        checkpoint_maps.append(mapping)
        for name, prediction in mapping.items():
            by_file[name].append(prediction)
    if any(len(values) != 3 for values in by_file.values()):
        raise RuntimeError("PBT checkpoint cell coverage differs")
    ensemble = {name: float(np.mean(values)) for name, values in by_file.items()}
    return ensemble, checkpoint_maps


def error_row(held_out, seed, prediction, truth, selected=None):
    error = abs(prediction - truth)
    row = {
        "held_out": held_out,
        "seed": seed,
        "prediction": float(prediction),
        "truth": float(truth),
        "abs_error": float(error),
        "ape": float(100.0 * error / max(truth, 1.0)),
    }
    if selected is not None:
        row["selected"] = selected
    return row


def calibrate_rows(selection_rows, cells, predictions):
    truth = {cell["name"]: float(cell["life"]) for cell in cells}
    rows = []
    for row in selection_rows:
        name = row["held_out"]
        # PBT can extrapolate outside the physical positive-life range under
        # domain shift.  A one-cycle floor makes the log-bias rule defined and
        # is fixed before inspecting any target truth.
        query_log = math.log(max(predictions[name], 1.0))
        residuals = [
            math.log(truth[support]) - math.log(max(predictions[support], 1.0))
            for support in row["selected"]
        ]
        prediction = math.exp(query_log + float(np.median(residuals)))
        rows.append(error_row(name, row["seed"], prediction, truth[name], row["selected"]))
    return rows


def aggregate_by_cell(rows):
    grouped = defaultdict(lambda: {"ae": [], "ape": []})
    for row in rows:
        grouped[row["held_out"]]["ae"].append(row["abs_error"])
        grouped[row["held_out"]]["ape"].append(row["ape"])
    return {
        name: {metric: float(np.mean(values)) for metric, values in metrics.items()}
        for name, metrics in grouped.items()
    }


def paired_test(baseline_rows, method_rows):
    baseline, method = aggregate_by_cell(baseline_rows), aggregate_by_cell(method_rows)
    names = sorted(set(baseline) & set(method))
    output = {"n_cells": len(names)}
    for metric in ("ae", "ape"):
        a = np.asarray([baseline[name][metric] for name in names])
        b = np.asarray([method[name][metric] for name in names])
        output[metric] = {
            "baseline_mean": float(np.mean(a)),
            "method_mean": float(np.mean(b)),
            "improved_same_worse": [
                int(np.sum(b < a - 1e-9)),
                int(np.sum(np.abs(b - a) <= 1e-9)),
                int(np.sum(b > a + 1e-9)),
            ],
            "wilcoxon_p": float(wilcoxon(a, b).pvalue),
        }
    return output


def main():
    pbt = json.load(open(PBT_SOURCE))
    dev = json.load(open(DEV_SOURCE))
    adaptive = json.load(open(ADAPTIVE_SOURCE))
    output = {
        "pbt_source": PBT_SOURCE.name,
        "development_source": DEV_SOURCE.name,
        "adaptive_source": ADAPTIVE_SOURCE.name,
        "protocol": (
            "official three-checkpoint PBT ensemble; prompt-covered cells only; "
            "identical curve-aware K-support and target labels for PBT log-bias "
            "calibration and local PASS-Cal; held-out query label never used"
        ),
        "results": {},
    }
    for dataset, cell_dataset in (("CALB", "CALB"), ("NAion", "NA-ion")):
        for horizon in HORIZONS:
            ensemble, checkpoint_maps = ensemble_predictions(pbt, dataset, horizon)
            cells = [cell for cell in load_cells(cell_dataset, horizon) if cell["name"] in ensemble]
            if {cell["name"] for cell in cells} != set(ensemble):
                raise RuntimeError(f"Feature/PBT cell mismatch for {dataset} H={horizon}")
            truth = {cell["name"]: float(cell["life"]) for cell in cells}
            raw_rows = [error_row(name, 0, prediction, truth[name]) for name, prediction in ensemble.items()]
            for k in KS:
                effective_k, weight, weight_source = selected_rule(adaptive, dataset, horizon, k)
                protocol_rows = evaluate_domain(cells, effective_k, 0.0, 100)
                local_rows = evaluate_domain(cells, effective_k, weight, 100)
                pbt_protocol_calibrated = calibrate_rows(protocol_rows, cells, ensemble)
                pbt_calibrated = calibrate_rows(local_rows, cells, ensemble)
                checkpoint_calibrated = [
                    summarize(calibrate_rows(local_rows, cells, mapping)) for mapping in checkpoint_maps
                ]
                key = f"{dataset}_h{horizon}_k{k}"
                output["results"][key] = {
                    "n_cells": len(cells),
                    "label_budget_k": k,
                    "effective_k": effective_k,
                    "curve_weight": "inf" if math.isinf(weight) else weight,
                    "weight_source": weight_source,
                    "pbt_raw_ensemble": summarize(raw_rows),
                    "pbt_protocol_logbias_ensemble": summarize(pbt_protocol_calibrated),
                    "pbt_logbias_ensemble": summarize(pbt_calibrated),
                    "pbt_logbias_checkpoint_summaries": checkpoint_calibrated,
                    "local_passcal": summarize(local_rows),
                    "pbt_raw_vs_calibrated": paired_test(raw_rows, pbt_calibrated),
                    "pbt_protocol_vs_curve_retrieval": paired_test(
                        pbt_protocol_calibrated, pbt_calibrated
                    ),
                    "pbt_calibrated_vs_local": paired_test(pbt_calibrated, local_rows),
                }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    compact = {
        key: {
            "n": value["n_cells"],
            "raw": value["pbt_raw_ensemble"],
            "pbt_protocol_cal": value["pbt_protocol_logbias_ensemble"],
            "pbt_cal": value["pbt_logbias_ensemble"],
            "local": value["local_passcal"],
            "pbt_cal_vs_local": value["pbt_calibrated_vs_local"],
        }
        for key, value in output["results"].items()
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
