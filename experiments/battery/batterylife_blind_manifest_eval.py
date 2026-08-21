"""One-shot label-opening evaluator for a frozen APACE-Cal manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from batterylife_asymmetric_cohort_router import predict
from batterylife_curve_aware_support import (
    curve_features,
    load_cells,
    robust_scale,
    structured_protocol,
)
from batterylife_transductive_pool_acquisition import WEIGHTS, distance_matrix


HERE = Path(__file__).parent


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(rows):
    return {
        "mae": float(np.mean([row["ae"] for row in rows])),
        "mape": float(np.mean([row["ape"] for row in rows])),
        "n_cells": len(rows),
    }


def paired_p(a, b):
    difference = np.asarray(a) - np.asarray(b)
    if np.all(np.abs(difference) <= 1e-12):
        return 1.0
    return float(wilcoxon(a, b).pvalue)


def aggregate(names, storage):
    rows = []
    for name in names:
        value = storage[name]
        if value["count"]:
            rows.append(
                {
                    "held_out": name,
                    "ae": value["ae_sum"] / value["count"],
                    "ape": value["ape_sum"] / value["count"],
                    "test_episodes": value["count"],
                }
            )
    return rows


def load_cells_archive(archive, labels, horizon):
    """First label-aware ZIP read; called only after manifest verification."""
    cells = []
    with zipfile.ZipFile(archive) as bundle:
        members = sorted(
            name for name in bundle.namelist()
            if name.lower().endswith(".pkl") and not name.endswith("/")
        )
        for member in members:
            name = Path(member).name
            if name not in labels:
                continue
            life = float(labels[name])
            with bundle.open(member) as stream:
                data = pickle.load(stream)
            features = curve_features(data, horizon)
            if features is None:
                continue
            if life < horizon:
                raise RuntimeError(
                    f"Label life {life} shorter than visible horizon {horizon}: {name}"
                )
            cells.append(
                {
                    "name": name,
                    "life": life,
                    "protocol": structured_protocol(data, name, horizon),
                    "curve": features,
                }
            )
    return cells


def evaluate_setting(cells, setting):
    horizon = setting["horizon"]
    names = [cell["name"] for cell in cells]
    index = {name: position for position, name in enumerate(names)}
    truth = np.asarray([cell["life"] for cell in cells], dtype=float)
    protocol = np.asarray([cell["protocol"] for cell in cells], dtype=float)
    curve = np.asarray([cell["curve"] for cell in cells], dtype=float)
    dp = distance_matrix(protocol, robust_scale(protocol), 1e9)
    dc = distance_matrix(curve, robust_scale(curve), 1e9)
    distances = {
        weight: dc
        if math.isinf(weight)
        else np.sqrt(np.square(dp) + weight * np.square(dc))
        for weight in WEIGHTS
    }
    manifest_names = set()
    for episode in setting["episodes"]:
        manifest_names.update(episode["acquisition"])
        manifest_names.update(episode["test"])
    if manifest_names != set(names):
        missing = sorted(manifest_names - set(names))
        extra = sorted(set(names) - manifest_names)
        raise RuntimeError(f"Manifest/evaluator cell mismatch missing={missing} extra={extra}")

    stores = {
        arm: defaultdict(lambda: {"ae_sum": 0.0, "ape_sum": 0.0, "count": 0})
        for arm in ("baseline", "method")
    }
    for episode in setting["episodes"]:
        test = np.asarray([index[name] for name in episode["test"]], dtype=int)
        for arm in ("baseline", "method"):
            support = np.asarray(
                [index[name] for name in episode[f"{arm}_support"]], dtype=int
            )
            prediction = predict(
                episode[f"{arm}_predictor"], distances, truth, support, test
            )
            ae = np.abs(prediction - truth[test])
            ape = 100.0 * ae / np.maximum(truth[test], 1.0)
            for position, held in enumerate(test):
                row = stores[arm][names[held]]
                row["ae_sum"] += float(ae[position])
                row["ape_sum"] += float(ape[position])
                row["count"] += 1
    baseline_rows = aggregate(names, stores["baseline"])
    method_rows = aggregate(names, stores["method"])
    baseline_map = {row["held_out"]: row for row in baseline_rows}
    method_map = {row["held_out"]: row for row in method_rows}
    paired_names = sorted(set(baseline_map) & set(method_map))
    a = np.asarray([baseline_map[name]["ape"] for name in paired_names])
    b = np.asarray([method_map[name]["ape"] for name in paired_names])
    return {
        "horizon": horizon,
        "label_budget_k": setting["label_budget_k"],
        "protocol_dispersion": setting["protocol_dispersion"],
        "route_counts": setting["route_counts"],
        "baseline": summarize(baseline_rows),
        "method": summarize(method_rows),
        "relative_mape_reduction_percent": float(
            100.0
            * (np.mean(a) - np.mean(b))
            / max(float(np.mean(a)), 1e-12)
        ),
        "paired_wilcoxon_p": paired_p(a, b),
        "improved_same_worse_cells": [
            int(np.sum(b < a - 1e-12)),
            int(np.sum(np.abs(b - a) <= 1e-12)),
            int(np.sum(b > a + 1e-12)),
        ],
        "per_cell": [
            {
                "held_out": name,
                "baseline_ape": baseline_map[name]["ape"],
                "method_ape": method_map[name]["ape"],
                "baseline_ae": baseline_map[name]["ae"],
                "method_ae": method_map[name]["ae"],
            }
            for name in paired_names
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--label-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.load(open(args.manifest))
    if manifest["phase"] != "PRELABEL_FROZEN_MANIFEST":
        raise RuntimeError("Not a prelabel frozen manifest")
    archive = Path(manifest["archive"])
    if archive.stat().st_size != manifest["archive_size"]:
        raise RuntimeError("Dataset archive size changed after prelabel freeze")
    if md5(archive) != manifest["archive_md5"]:
        raise RuntimeError("Dataset archive MD5 changed after prelabel freeze")
    if "prelabel_script_sha256" in manifest and sha256(
        HERE / "batterylife_blind_prelabel_manifest.py"
    ) != manifest["prelabel_script_sha256"]:
        raise RuntimeError("Prelabel script changed after manifest freeze")
    if "evaluator_script_sha256" in manifest and sha256(Path(__file__)) != manifest[
        "evaluator_script_sha256"
    ]:
        raise RuntimeError("Evaluator script changed after manifest freeze")
    if sha256(args.label_file) != manifest["label_file_opaque_sha256"]:
        raise RuntimeError("Life-label file changed after prelabel freeze")
    if sha256(HERE / "batterylife_asymmetric_cohort_router.py") != manifest[
        "method_script_sha256"
    ]:
        raise RuntimeError("Method script changed after prelabel freeze")
    if sha256(HERE.parents[1] / "METHOD_FREEZE.md") != manifest[
        "method_freeze_sha256"
    ]:
        raise RuntimeError("Method freeze record changed after prelabel freeze")
    if sha256(HERE / "batterylife_asymmetric_cohort_router.json") != manifest[
        "development_result_sha256"
    ]:
        raise RuntimeError("Development evidence changed after prelabel freeze")
    # This is the first semantic opening of the label JSON in the blind chain.
    labels = json.load(open(args.label_file))
    if not labels:
        raise RuntimeError("Empty label file")
    loaded = {}
    for horizon in sorted({setting["horizon"] for setting in manifest["settings"]}):
        if manifest.get("feature_source") == "verified_zip_stream":
            loaded[horizon] = load_cells_archive(
                manifest["archive"], labels, horizon
            )
        else:
            loaded[horizon] = load_cells(manifest["dataset"], horizon)
    results = [
        evaluate_setting(loaded[setting["horizon"]], setting)
        for setting in manifest["settings"]
    ]
    output = {
        "phase": "ONE_SHOT_LABEL_OPENED_EVALUATION",
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "dataset": manifest["dataset"],
        "label_file_sha256": sha256(args.label_file),
        "active_settings_frozen_prelabel": manifest["active_settings"],
        "results": results,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                f"h{row['horizon']}_k{row['label_budget_k']}": {
                    "mape": [row["baseline"]["mape"], row["method"]["mape"]],
                    "relative_reduction_percent": row["relative_mape_reduction_percent"],
                    "cells": row["improved_same_worse_cells"],
                    "p": row["paired_wilcoxon_p"],
                    "route_counts": row["route_counts"],
                }
                for row in results
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
