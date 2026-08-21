"""Frozen-format blind chain for the SNU dynamic Markov-protocol dataset.

`prelabel` stops each CSV immediately after the first row of numeric cycle 51;
`evaluate` is the only mode that scans CSV members to EOF to derive EOL as the
maximum numeric TotCycle.  Dataset 2 is reserved for parser/evaluator dry-run;
Dataset 1 remains the independent confirmation cohort.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np

import batterylife_asymmetric_cohort_router_v2 as v2
import batterylife_blind_manifest_eval as common_eval
import batterylife_blind_prelabel_manifest as common_prelabel
from batterylife_curve_aware_support import curve_features, structured_protocol


HERE = Path(__file__).parent
FREEZE = HERE.parents[1] / "CONFIRMATION_FREEZE_SNU.md"
DEV_RESULT = HERE / "batterylife_asymmetric_cohort_router_v2.json"
EXPECTED_SHA256 = "2c58920e663fd089297ec2678a5a8ff791b6737466b9420bc2129c0cd2cde7ff"
MAX_HORIZON = 50
NOMINAL_CAPACITY_AH = 3.0
AMBIENT_TEMPERATURE_C = 25.0


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_cycle(value):
    text = str(value).strip()
    return int(text) if text.isdigit() else None


def number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def csv_members(archive, subset):
    prefix = f"Dataset {subset}/"
    with zipfile.ZipFile(archive) as bundle:
        return sorted(name for name in bundle.namelist()
                      if name.startswith(prefix) and name.lower().endswith(".csv"))


def finalize_cycle(rows):
    def finite(key, condition=None):
        values = [number(row.get(key)) for row in rows
                  if condition is None or condition(row)]
        return np.asarray([x for x in values if np.isfinite(x)], dtype=float)

    voltage = finite("Voltage(V)")
    current = finite("Current(A)")
    charge_current = finite("Current(A)", lambda r: r.get("Type") == "charging")
    discharge_current = finite("Current(A)", lambda r: r.get("Type") == "discharging")
    charge_capacity = finite("Char. Cap.(Ah)")
    discharge_capacity = finite("Dischar. Cap.(Ah)")
    impedance = finite("Imp(10s)(mOhm)")
    # The public record states a 10-second logging interval. Aggregate to the
    # exact statistics consumed by the frozen 56-D curve signature.
    cycle = {
        "voltage_in_V": ([float(np.min(voltage)), float(np.median(voltage)),
                          float(np.max(voltage))] if voltage.size else []),
        "current_in_A": [float(np.mean(np.abs(current)))] if current.size else [],
        "charge_capacity_in_Ah": [float(np.max(charge_capacity))]
            if charge_capacity.size else [],
        "discharge_capacity_in_Ah": [float(np.max(discharge_capacity))]
            if discharge_capacity.size else [],
        "time_in_s": [0.0, max(0.0, 10.0 * (len(rows) - 1))],
        "temperature_in_C": [AMBIENT_TEMPERATURE_C],
        "internal_resistance_in_ohm": [float(np.median(impedance)) / 1000.0]
            if impedance.size else [],
    }
    charge_rate = (float(np.max(np.abs(charge_current))) / NOMINAL_CAPACITY_AH
                   if charge_current.size else np.nan)
    discharge_rate = (float(np.max(np.abs(discharge_current))) / NOMINAL_CAPACITY_AH
                      if discharge_current.size else np.nan)
    return cycle, charge_rate, discharge_rate


def early_cell(bundle, member):
    grouped, current_cycle, rows = [], None, []
    with bundle.open(member) as raw:
        reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig",
                                                errors="strict", newline=""))
        for row in reader:
            cycle = numeric_cycle(row.get("TotCycle"))
            if cycle is None:
                continue
            if cycle > MAX_HORIZON:
                break  # hard information barrier: never inspect later rows
            if current_cycle is None:
                current_cycle = cycle
            if cycle != current_cycle:
                grouped.append((current_cycle, *finalize_cycle(rows)))
                current_cycle, rows = cycle, []
            rows.append(row)
        if rows and current_cycle is not None and current_cycle <= MAX_HORIZON:
            grouped.append((current_cycle, *finalize_cycle(rows)))
    grouped.sort(key=lambda x: x[0])
    if [x[0] for x in grouped] != list(range(1, MAX_HORIZON + 1)):
        raise RuntimeError(f"Non-contiguous first {MAX_HORIZON} cycles: {member}")
    cycles = [x[1] for x in grouped]
    data = {
        "cycle_data": cycles,
        "nominal_capacity_in_Ah": NOMINAL_CAPACITY_AH,
        "SOC_interval": [0.0, 1.0],
        "charge_protocol": [{"rate_in_C": x[2]} for x in grouped],
        "discharge_protocol": [{"rate_in_C": x[3]} for x in grouped],
    }
    name = Path(member).name
    return {h: {"name": name,
                "protocol": structured_protocol(
                    {**data, "cycle_data": cycles[:h],
                     "charge_protocol": data["charge_protocol"][:h],
                     "discharge_protocol": data["discharge_protocol"][:h]}, name, h),
                "curve": curve_features({**data, "cycle_data": cycles[:h]}, h)}
            for h in (10, 20, 50)}


def load_early(archive, subset):
    output = {10: [], 20: [], 50: []}
    with zipfile.ZipFile(archive) as bundle:
        members = sorted(name for name in bundle.namelist()
                         if name.startswith(f"Dataset {subset}/")
                         and name.lower().endswith(".csv"))
        for member in members:
            cell = early_cell(bundle, member)
            for horizon in output:
                output[horizon].append(cell[horizon])
    return output


def derive_eol_labels(archive, subset):
    """The sole label-opening operation: scan every selected CSV to EOF."""
    labels = {}
    with zipfile.ZipFile(archive) as bundle:
        members = sorted(name for name in bundle.namelist()
                         if name.startswith(f"Dataset {subset}/")
                         and name.lower().endswith(".csv"))
        for member in members:
            maximum = 0
            with bundle.open(member) as raw:
                reader = csv.DictReader(io.TextIOWrapper(
                    raw, encoding="utf-8-sig", errors="strict", newline=""))
                for row in reader:
                    cycle = numeric_cycle(row.get("TotCycle"))
                    if cycle is not None:
                        maximum = max(maximum, cycle)
            if maximum < MAX_HORIZON:
                raise RuntimeError(f"EOL {maximum} shorter than H50: {member}")
            labels[Path(member).name] = float(maximum)
    return labels


def prelabel(args):
    if sha256(args.archive) != EXPECTED_SHA256:
        raise RuntimeError("Official SNU archive SHA-256 mismatch")
    loaded = load_early(args.archive, args.subset)
    dev = json.loads(DEV_RESULT.read_text())
    settings, predictor_freeze = [], {}
    for h in (10, 20, 50):
        for k in (1, 3, 5, 10):
            choice, ranking = common_prelabel.external_predictor(dev, h, k)
            predictor_freeze[f"h{h}_k{k}"] = {"selected": choice,
                                               "ranking": ranking}
            settings.append(v2_manifest_setting(loaded[h], h, k, args.seeds,
                                                choice["predictor"]))
    active = [f"h{x['horizon']}_k{x['label_budget_k']}" for x in settings
              if any(r.startswith("active_") for r in x["route_counts"])]
    output = {
        "phase": "SNU_DYNAMIC_PRELABEL_FROZEN", "subset": args.subset,
        "archive": str(args.archive), "archive_size": args.archive.stat().st_size,
        "archive_sha256": EXPECTED_SHA256,
        "information_barrier": (
            "Each CSV read stopped at the first row with numeric TotCycle > 50; "
            "no file tail, maximum cycle, EOL, RUL, or performance metric was read."),
        "cell_count": len(loaded[10]), "method_freeze_sha256": sha256(FREEZE),
        "chain_script_sha256": sha256(Path(__file__)),
        "method_script_sha256": sha256(
            HERE / "batterylife_asymmetric_cohort_router_v2.py"),
        "v1_dependency_sha256": sha256(
            HERE / "batterylife_asymmetric_cohort_router.py"),
        "development_result_sha256": sha256(DEV_RESULT),
        "adapter_constants": {"nominal_capacity_Ah": NOMINAL_CAPACITY_AH,
                              "ambient_temperature_C": AMBIENT_TEMPERATURE_C,
                              "logging_interval_s": 10.0,
                              "eol_definition": "maximum numeric TotCycle"},
        "predictor_freeze": predictor_freeze, "active_settings": active,
        "settings": settings,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"subset": args.subset, "cells": len(loaded[10]),
                      "active_settings": active,
                      "unlabeled_diagnostics": {
                          f"h{x['horizon']}": {
                              "protocol_dispersion": x["protocol_dispersion"],
                              "rho": x["distance_concordance_spearman"]}
                          for x in settings if x["label_budget_k"] == 1}}, indent=2))


def v2_manifest_setting(cells, horizon, budget, seeds, baseline_predictor):
    # Reuse the already dry-run v2 setting builder without importing its CLI.
    import batterylife_blind_prelabel_manifest_v2 as builder
    return builder.setting_manifest(cells, horizon, budget, seeds, baseline_predictor)


def evaluate(args):
    manifest = json.loads(args.manifest.read_text())
    if manifest["phase"] != "SNU_DYNAMIC_PRELABEL_FROZEN":
        raise RuntimeError("Not an SNU dynamic prelabel manifest")
    frozen = [
        (args.archive.stat().st_size == manifest["archive_size"], "archive size"),
        (sha256(args.archive) == manifest["archive_sha256"], "archive hash"),
        (sha256(FREEZE) == manifest["method_freeze_sha256"], "freeze record"),
        (sha256(Path(__file__)) == manifest["chain_script_sha256"], "chain script"),
        (sha256(HERE / "batterylife_asymmetric_cohort_router_v2.py") ==
         manifest["method_script_sha256"], "method"),
        (sha256(HERE / "batterylife_asymmetric_cohort_router.py") ==
         manifest["v1_dependency_sha256"], "v1 dependency"),
        (sha256(DEV_RESULT) == manifest["development_result_sha256"],
         "development result"),
    ]
    failed = [name for ok, name in frozen if not ok]
    if failed:
        raise RuntimeError("Frozen-chain verification failed: " + ", ".join(failed))
    labels = derive_eol_labels(args.archive, manifest["subset"])
    loaded = load_early(args.archive, manifest["subset"])
    for h, cells in loaded.items():
        for cell in cells:
            cell["life"] = labels[cell["name"]]
    common_eval.predict = v2.predict
    results = [common_eval.evaluate_setting(loaded[x["horizon"]], x)
               for x in manifest["settings"]]
    output = {
        "phase": "SNU_DYNAMIC_ONE_SHOT_EVALUATION",
        "manifest": str(args.manifest), "manifest_sha256": sha256(args.manifest),
        "subset": manifest["subset"], "eol_definition": "maximum numeric TotCycle",
        "active_settings_frozen_prelabel": manifest["active_settings"],
        "results": results,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({f"h{x['horizon']}_k{x['label_budget_k']}": {
        "mape": [x["baseline"]["mape"], x["method"]["mape"]],
        "relative_reduction_percent": x["relative_mape_reduction_percent"],
        "cells": x["improved_same_worse_cells"], "p": x["paired_wilcoxon_p"],
        "route_counts": x["route_counts"]} for x in results}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("prelabel")
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--subset", type=int, choices=(1, 2, 3), required=True)
    p.add_argument("--seeds", type=int, default=100)
    p.add_argument("--output", type=Path, required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--archive", type=Path, required=True)
    e.add_argument("--manifest", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prelabel(args) if args.mode == "prelabel" else evaluate(args)


if __name__ == "__main__":
    main()
