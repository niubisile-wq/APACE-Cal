"""Group-level blind audit on the compact Farasis early-cycle benchmark.

The compact NPZ is a third-party deterministic derivative of the official
CC-BY-4.0 Farasis release.  This script verifies its published SHA-256,
aggregates replicate cells into material/protocol groups, and never treats
replicates as independent prediction units.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

import batterylife_fixed_pool_acquisition as acquisition
import batterylife_fixed_pool_nested as fixed_pool
from batterylife_curve_aware_support import evaluate_domain as evaluate_query
from batterylife_curve_aware_support import robust_slope, summarize
from batterylife_external_frozen_adaptive import frozen_rule


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "farasis_compact" / "battery_pbt_eval_v1.npz"
MANIFEST = DATA.with_name("manifest.json")
QUERY_DEV = Path(__file__).with_name("batterylife_adaptive_effective_support.json")
FIXED_DEV = Path(__file__).with_name("batterylife_fixed_pool_nested.json")
ACQUISITION_DEV = Path(__file__).with_name("batterylife_fixed_pool_acquisition.json")
OUTPUT = Path(__file__).with_suffix(".json")
EXPECTED_SHA256 = "eaef2350a7c8ee06e959e4c55db66e2f567c65287602b3729871ae435c0d5715"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def farasis_signature(curves, horizon):
    signatures = []
    for cycle in curves[:horizon]:
        voltage, current, capacity = cycle
        signatures.append(
            [
                float(np.max(capacity[150:])),
                float(np.max(capacity[:150])),
                float(np.median(voltage)),
                float(np.max(voltage) - np.min(voltage)),
                float(np.mean(np.abs(current))),
                np.nan,
                np.nan,
                np.nan,
            ]
        )
    signatures = np.asarray(signatures, dtype=float)
    positions = np.rint(np.linspace(0, horizon - 1, 5)).astype(int)
    snapshots = signatures[positions].reshape(-1)
    slopes = np.asarray([robust_slope(signatures[:, j]) for j in range(8)])
    deltas = signatures[-1] - signatures[0]
    return np.r_[snapshots, slopes, deltas]


def finite_column_mean(matrix):
    output = np.full(matrix.shape[1], np.nan)
    for column in range(matrix.shape[1]):
        values = matrix[:, column]
        values = values[np.isfinite(values)]
        if len(values):
            output[column] = float(np.mean(values))
    return output


def load_groups(horizon):
    archive = np.load(DATA, allow_pickle=False)
    groups = defaultdict(list)
    for index, group_id in enumerate(archive["group_id"]):
        groups[int(group_id)].append(index)
    cells = []
    for group_id, indices in sorted(groups.items()):
        signatures = np.asarray([farasis_signature(archive["curves"][i], horizon) for i in indices])
        cells.append(
            {
                "name": f"farasis-group-{group_id:02d}",
                "life": float(np.mean(archive["life"][indices])),
                "protocol": np.full(5, np.nan),
                "curve": finite_column_mean(signatures),
                "replicates": len(indices),
            }
        )
    return cells


def average_frozen_choice(dev, horizon, budget, ranking_key):
    records = [
        record
        for record in dev["results"]
        if record["horizon"] == horizon
        and int(record.get("k", record.get("label_budget_k"))) == budget
    ]
    scores = defaultdict(list)
    for record in records:
        for candidate in record[ranking_key]:
            scores[candidate["candidate"]].append(
                (candidate["development_macro_mape"], candidate["development_macro_mae"])
            )
    ranked = [
        {
            "candidate": candidate,
            "macro_mape": float(np.mean([value[0] for value in values])),
            "macro_mae": float(np.mean([value[1] for value in values])),
        }
        for candidate, values in scores.items()
    ]
    return min(ranked, key=lambda row: (row["macro_mape"], row["macro_mae"])), ranked


def paired_from_rows(baseline_rows, method_rows):
    baseline = defaultdict(list)
    method = defaultdict(list)
    for row in baseline_rows:
        baseline[row["held_out"]].append(row["ape"])
    for row in method_rows:
        method[row["held_out"]].append(row["ape"])
    names = sorted(set(baseline) & set(method))
    a = np.asarray([np.mean(baseline[name]) for name in names])
    b = np.asarray([np.mean(method[name]) for name in names])
    return {
        "n_groups": len(names),
        "improved_same_worse": [
            int(np.sum(b < a - 1e-9)),
            int(np.sum(np.abs(b - a) <= 1e-9)),
            int(np.sum(b > a + 1e-9)),
        ],
        "wilcoxon_p": float(wilcoxon(a, b).pvalue),
    }


def main():
    digest = sha256(DATA)
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"Farasis compact SHA-256 mismatch: {digest}")
    manifest = json.load(open(MANIFEST))
    if manifest["archive_sha256"] != digest or manifest["retained_cells"] != 116:
        raise RuntimeError("Manifest/archive audit mismatch")
    query_dev = json.load(open(QUERY_DEV))
    fixed_dev = json.load(open(FIXED_DEV))
    acquisition_dev = json.load(open(ACQUISITION_DEV))
    output = {
        "source": "aashay96/scientific-posttrain-battery-pbt-eval derivative of Zenodo 17654407",
        "sha256": digest,
        "manifest": {
            key: manifest[key]
            for key in (
                "source_record",
                "source_raw_archive_sha256",
                "source_labels_sha256",
                "batteryml_commit",
                "batterylife_preprocessing_commit",
                "retained_cells",
                "retained_groups",
                "exclusions",
                "expert_review_status",
            )
        },
        "unit": "material/protocol group; replicate cells aggregated before evaluation",
        "results": {},
    }

    for horizon in (10, 20, 50):
        cells = load_groups(horizon)
        if len(cells) != manifest["retained_groups"]:
            raise RuntimeError("Farasis group count mismatch")
        for budget in (1, 3, 5, 10):
            # Query-adaptive rule, included only under its correct per-query
            # interpretation and never substituted for total-K evidence.
            query_choice, _ = frozen_rule(query_dev, horizon, budget)
            query_k = min(int(query_choice["effective_k"]), len(cells) - 1)
            query_weight = math.inf if query_choice["weight"] == "inf" else float(query_choice["weight"])
            query_baseline = evaluate_query(cells, query_k, 0.0, 100)
            query_method = evaluate_query(cells, query_k, query_weight, 100)

            fixed_summary, fixed_cells, _, _ = fixed_pool.evaluate_domain(
                cells, horizon, budget, seeds=100
            )
            effective_k = min(budget, len(cells) - 1)
            fixed_baseline_choice, _ = average_frozen_choice(
                fixed_dev, horizon, budget, "baseline_ranking"
            )
            fixed_method_choice, _ = average_frozen_choice(
                fixed_dev, horizon, budget, "method_ranking"
            )

            acquisition_summary, acquisition_cells, _, _, acquisition_k = acquisition.evaluate_domain(
                cells, horizon, budget
            )
            acquisition_baseline_choice, _ = average_frozen_choice(
                acquisition_dev, horizon, budget, "baseline_ranking"
            )
            acquisition_method_choice, _ = average_frozen_choice(
                acquisition_dev, horizon, budget, "method_ranking"
            )
            key = f"h{horizon}_k{budget}"
            output["results"][key] = {
                "n_groups": len(cells),
                "replicate_counts": [cell["replicates"] for cell in cells],
                "query_adaptive": {
                    "effective_k": query_k,
                    "weight": query_choice["weight"],
                    "protocol_only": summarize(query_baseline),
                    "curve_aware": summarize(query_method),
                    "paired": paired_from_rows(query_baseline, query_method),
                },
                "fixed_random_pool": {
                    "effective_k": effective_k,
                    "baseline_candidate": fixed_baseline_choice["candidate"],
                    "method_candidate": fixed_method_choice["candidate"],
                    "baseline": fixed_summary[fixed_baseline_choice["candidate"]],
                    "method": fixed_summary[fixed_method_choice["candidate"]],
                },
                "active_acquisition": {
                    "effective_k": acquisition_k,
                    "baseline_candidate": acquisition_baseline_choice["candidate"],
                    "method_candidate": acquisition_method_choice["candidate"],
                    "baseline": acquisition_summary[acquisition_baseline_choice["candidate"]],
                    "method": acquisition_summary[acquisition_method_choice["candidate"]],
                },
            }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    compact = {
        key: {
            "query": [value["query_adaptive"]["protocol_only"]["mape"], value["query_adaptive"]["curve_aware"]["mape"]],
            "fixed": [value["fixed_random_pool"]["baseline"]["mape"], value["fixed_random_pool"]["method"]["mape"]],
            "active": [value["active_acquisition"]["baseline"]["mape"], value["active_acquisition"]["method"]["mape"]],
        }
        for key, value in output["results"].items()
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
