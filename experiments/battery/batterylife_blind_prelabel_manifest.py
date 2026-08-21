"""Create a label-blind external-domain execution manifest for APACE-Cal.

This script intentionally never opens or parses a life-label JSON.  It hashes
the label file as opaque bytes, reads only protocol and first-H-cycle features,
freezes every acquisition/test split and support identity, and records whether
the frozen router actually activates.  The resulting manifest must be written
and hashed before the separate evaluator is allowed to read labels.
"""
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

from batterylife_asymmetric_cohort_router import (
    LOW_PROTOCOL_THRESHOLD,
    protocol_dispersion,
    routed_support,
)
from batterylife_curve_aware_support import curve_features, robust_scale, structured_protocol
from batterylife_transductive_pool_acquisition import (
    WEIGHTS,
    distance_matrix,
    key_number,
    select_facilities,
)


HERE = Path(__file__).parent
DEV_RESULT = HERE / "batterylife_asymmetric_cohort_router.json"


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


def top_level_json_keys(path):
    """Read only top-level JSON object keys while lexically skipping values."""
    text = path.read_text()
    decoder = json.JSONDecoder()
    index = 0

    def skip_space(position):
        while position < len(text) and text[position].isspace():
            position += 1
        return position

    index = skip_space(index)
    if index >= len(text) or text[index] != "{":
        raise RuntimeError("Label membership file is not a JSON object")
    index += 1
    keys = set()
    while True:
        index = skip_space(index)
        if index < len(text) and text[index] == "}":
            break
        if index < len(text) and text[index] == ",":
            index = skip_space(index + 1)
        key, end = decoder.raw_decode(text, index)
        if not isinstance(key, str):
            raise RuntimeError("Non-string top-level label key")
        index = skip_space(end)
        if index >= len(text) or text[index] != ":":
            raise RuntimeError("Malformed top-level label mapping")
        keys.add(key)
        index += 1
        # Skip the value as raw JSON syntax.  No numeric/string value is
        # decoded, stored, summarized, or returned.
        nested = 0
        in_string = False
        escaped = False
        while index < len(text):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char in "[{":
                nested += 1
            elif char in "]}":
                if char == "}" and nested == 0:
                    break
                nested -= 1
            elif char == "," and nested == 0:
                break
            index += 1
    return keys


def load_unlabeled(dataset_dir, horizons, allowed_names=None):
    cells = {horizon: [] for horizon in horizons}
    for path in sorted(dataset_dir.glob("*.pkl")):
        if allowed_names is not None and path.name not in allowed_names:
            continue
        with open(path, "rb") as stream:
            data = pickle.load(stream)
        for horizon in horizons:
            features = curve_features(data, horizon)
            if features is None:
                continue
            cells[horizon].append(
                {
                    "name": path.name,
                    "protocol": structured_protocol(data, path.name, horizon),
                    "curve": features,
                }
            )
    return cells


def load_unlabeled_archive(archive, horizons, allowed_names=None):
    """Stream pickle members from a verified ZIP without extracting the corpus."""
    cells = {horizon: [] for horizon in horizons}
    with zipfile.ZipFile(archive) as bundle:
        members = sorted(
            name for name in bundle.namelist()
            if name.lower().endswith(".pkl") and not name.endswith("/")
        )
        for member in members:
            name = Path(member).name
            if allowed_names is not None and name not in allowed_names:
                continue
            with bundle.open(member) as stream:
                data = pickle.load(stream)
            for horizon in horizons:
                features = curve_features(data, horizon)
                if features is None:
                    continue
                cells[horizon].append(
                    {
                        "name": name,
                        "protocol": structured_protocol(data, name, horizon),
                        "curve": features,
                    }
                )
    return cells


def external_predictor(dev, horizon, budget):
    records = [
        record
        for record in dev["results"]
        if record["horizon"] == horizon and record["label_budget_k"] == budget
    ]
    scores = defaultdict(list)
    maes = defaultdict(list)
    for record in records:
        for candidate in record["predictor_ranking"]:
            scores[candidate["predictor"]].append(candidate["development_macro_mape"])
            maes[candidate["predictor"]].append(candidate["development_macro_mae"])
    ranking = [
        {
            "predictor": predictor,
            "six_domain_mean_lodo_mape": float(np.mean(values)),
            "six_domain_mean_lodo_mae": float(np.mean(maes[predictor])),
        }
        for predictor, values in scores.items()
    ]
    return min(
        ranking,
        key=lambda row: (row["six_domain_mean_lodo_mape"], row["six_domain_mean_lodo_mae"]),
    ), ranking


def setting_manifest(cells, horizon, budget, seeds, baseline_predictor):
    names = [cell["name"] for cell in cells]
    protocol = np.asarray([cell["protocol"] for cell in cells], dtype=float)
    curve = np.asarray([cell["curve"] for cell in cells], dtype=float)
    clients = np.arange(len(cells), dtype=int)
    dp = distance_matrix(protocol, robust_scale(protocol), 1e9)
    dc = distance_matrix(curve, robust_scale(curve), 1e9)
    distances = {
        weight: dc
        if math.isinf(weight)
        else np.sqrt(np.square(dp) + weight * np.square(dc))
        for weight in WEIGHTS
    }
    spread = protocol_dispersion(dp)
    episodes, route_counts = [], defaultdict(int)
    for seed in range(1, seeds + 1):
        rng = np.random.default_rng(60_000_000 * horizon + 10_000 * budget + seed)
        permutation = rng.permutation(len(cells))
        acquisition_n = min(
            len(cells) - 2, max(int(math.ceil(0.7 * len(cells))), budget)
        )
        acquisition = np.sort(permutation[:acquisition_n])
        test = np.sort(permutation[acquisition_n:])
        k = min(budget, len(acquisition))
        shuffled = rng.permutation(len(cells))
        tie_rank = np.empty(len(cells), dtype=int)
        tie_rank[shuffled] = np.arange(len(cells))
        random_support = np.sort(rng.choice(acquisition, size=k, replace=False))
        method_support, route = routed_support(
            spread,
            k,
            random_support,
            acquisition,
            clients,
            distances,
            tie_rank,
        )
        route_counts[route] += 1
        method_predictor = baseline_predictor
        if route.startswith("active_w"):
            weight = route.split("active_w", 1)[1]
            method_predictor = f"w{weight}_bw0.5"
        episodes.append(
            {
                "seed": seed,
                "acquisition": [names[index] for index in acquisition],
                "test": [names[index] for index in test],
                "baseline_support": [names[index] for index in random_support],
                "method_support": [names[index] for index in method_support],
                "route": route,
                "baseline_predictor": baseline_predictor,
                "method_predictor": method_predictor,
            }
        )
    return {
        "horizon": horizon,
        "label_budget_k": budget,
        "n_unlabeled_cells": len(cells),
        "protocol_dispersion": spread,
        "route_counts": dict(route_counts),
        "episodes": episodes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-md5", required=True)
    parser.add_argument("--label-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument(
        "--label-membership-only",
        action="store_true",
        help="Use only top-level label JSON keys to exclude unlabeled pickle identities.",
    )
    args = parser.parse_args()

    archive_md5 = md5(args.archive)
    if archive_md5 != args.expected_md5:
        raise RuntimeError(f"Archive MD5 mismatch: {archive_md5}")
    dev = json.load(open(DEV_RESULT))
    horizons, budgets = (10, 20, 50), (1, 3, 5, 10)
    allowed_names = (
        top_level_json_keys(args.label_file) if args.label_membership_only else None
    )
    if args.dataset_dir is None:
        loaded = load_unlabeled_archive(args.archive, horizons, allowed_names)
        feature_source = "verified_zip_stream"
    else:
        loaded = load_unlabeled(args.dataset_dir, horizons, allowed_names)
        feature_source = "extracted_directory"
    settings = []
    predictor_freeze = {}
    for horizon in horizons:
        if len(loaded[horizon]) < 3:
            raise RuntimeError(f"Too few unlabeled cells at H={horizon}")
        for budget in budgets:
            choice, ranking = external_predictor(dev, horizon, budget)
            predictor_freeze[f"h{horizon}_k{budget}"] = {
                "selected": choice,
                "ranking": ranking,
            }
            settings.append(
                setting_manifest(
                    loaded[horizon],
                    horizon,
                    budget,
                    args.seeds,
                    choice["predictor"],
                )
            )
    active_settings = [
        f"h{record['horizon']}_k{record['label_budget_k']}"
        for record in settings
        if any(route.startswith("active_") for route in record["route_counts"])
    ]
    output = {
        "phase": "PRELABEL_FROZEN_MANIFEST",
        "dataset": args.dataset,
        "archive": str(args.archive),
        "archive_size": args.archive.stat().st_size,
        "archive_md5": archive_md5,
        "feature_source": feature_source,
        "label_file_opaque_sha256": sha256(args.label_file),
        "explicit_non_access_statement": (
            "This script lexically read top-level label keys only and skipped every value "
            "without decoding, storing, or summarizing it."
            if args.label_membership_only else
            "This script hashed the label file as opaque bytes and never opened or parsed it."
        ),
        "label_membership_only": args.label_membership_only,
        "label_membership_count": (
            len(allowed_names) if allowed_names is not None else None
        ),
        "method_freeze_sha256": sha256(HERE.parents[1] / "METHOD_FREEZE.md"),
        "prelabel_script_sha256": sha256(Path(__file__)),
        "evaluator_script_sha256": sha256(
            HERE / "batterylife_blind_manifest_eval.py"
        ),
        "method_script_sha256": sha256(
            HERE / "batterylife_asymmetric_cohort_router.py"
        ),
        "development_result_sha256": sha256(DEV_RESULT),
        "router_constants": {
            "low_protocol_threshold": LOW_PROTOCOL_THRESHOLD,
            "high_protocol_threshold": 0.60,
            "active_weights": [2.0, 0.5],
            "rbf_bandwidth": 0.5,
        },
        "predictor_freeze": predictor_freeze,
        "active_settings": active_settings,
        "settings": settings,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "active_settings": active_settings,
                "cells_by_horizon": {h: len(loaded[h]) for h in horizons},
                "protocol_dispersion": {
                    f"h{record['horizon']}": record["protocol_dispersion"]
                    for record in settings
                    if record["label_budget_k"] == 1
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
