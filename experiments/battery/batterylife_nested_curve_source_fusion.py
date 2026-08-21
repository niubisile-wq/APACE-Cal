"""Nested LODO audit of source early-curve models and PASS-Cal fusion."""
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from batterylife_curve_aware_support import DATASETS, evaluate_domain, load_cells


SUPPORT_SOURCE = Path(__file__).with_name("batterylife_curve_aware_support.json")
OUTPUT = Path(__file__).with_name("batterylife_nested_curve_source_fusion.json")
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)  # 0=calibrated source, 1=local PASS-Cal


def model_factory(name):
    if name == "ridge10":
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=10.0),
        )
    if name == "rf":
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=2,
                max_features=0.8,
                random_state=819,
                n_jobs=-1,
            ),
        )
    if name == "extra_trees":
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            ExtraTreesRegressor(
                n_estimators=500,
                min_samples_leaf=2,
                max_features=0.8,
                random_state=819,
                n_jobs=-1,
            ),
        )
    raise KeyError(name)


def train_source(cells_by_domain, train_domains, model_name):
    x, y, weights = [], [], []
    for domain in train_domains:
        cells = cells_by_domain[domain]
        for cell in cells:
            x.append(cell["curve"])
            y.append(np.log(cell["life"]))
            weights.append(1.0 / len(cells))
    model = model_factory(model_name)
    model.fit(np.asarray(x), np.asarray(y), **{"ridge__sample_weight": np.asarray(weights)} if model_name == "ridge10" else {f"{model.steps[-1][0]}__sample_weight": np.asarray(weights)})
    source_range = (float(np.min(y)), float(np.max(y)))
    return model, source_range


def evaluate_target(model, source_range, cells, support_weight):
    features = np.asarray([cell["curve"] for cell in cells])
    log_base = np.clip(model.predict(features), source_range[0], source_range[1])
    support_rows = evaluate_domain(cells, 3, support_weight, 1)
    index = {cell["name"]: i for i, cell in enumerate(cells)}
    rows = []
    for support_row in support_rows:
        held = index[support_row["held_out"]]
        chosen = [index[name] for name in support_row["selected"]]
        correction = float(np.median([np.log(cells[j]["life"]) - log_base[j] for j in chosen]))
        log_calibrated_source = log_base[held] + correction
        log_local = np.log(support_row["prediction"])
        truth = cells[held]["life"]
        predictions = {"raw_source": float(np.exp(log_base[held]))}
        for alpha in ALPHAS:
            predictions[f"blend_{alpha}"] = float(np.exp((1.0 - alpha) * log_calibrated_source + alpha * log_local))
        rows.append(
            {
                "held_out": cells[held]["name"],
                "truth": truth,
                "selected": support_row["selected"],
                "predictions": predictions,
            }
        )
    return rows


def metrics(rows, prediction_key):
    errors = np.asarray([abs(row["predictions"][prediction_key] - row["truth"]) for row in rows])
    ape = np.asarray([100.0 * error / max(row["truth"], 1.0) for row, error in zip(rows, errors)])
    return {"mae": float(np.mean(errors)), "mape": float(np.mean(ape)), "median_ae": float(np.median(errors))}


def selected_support_weight(support_json, horizon, target):
    record = next(
        r
        for r in support_json["nested_results"]
        if r["horizon"] == horizon and r["target"] == target and r["k"] == 3
    )
    value = record["selected_curve_weight"]
    return np.inf if value == "inf" else float(value)


def main():
    support_json = json.load(open(SUPPORT_SOURCE))
    output = {
        "support_source": SUPPORT_SOURCE.name,
        "protocol": "outer target excluded from source training and model/blend selection; inner LODO over remaining five domains; domain-balanced log-life training",
        "results": [],
    }
    for horizon in (10, 20, 50):
        cells_by_domain = {domain: load_cells(domain, horizon) for domain in DATASETS}
        for outer_target in DATASETS:
            development = [domain for domain in DATASETS if domain != outer_target]
            support_weight = selected_support_weight(support_json, horizon, outer_target)
            candidates = []
            for model_name in ("ridge10", "rf", "extra_trees"):
                inner_rows = {}
                for validation in development:
                    inner_train = [domain for domain in development if domain != validation]
                    model, source_range = train_source(cells_by_domain, inner_train, model_name)
                    inner_rows[validation] = evaluate_target(
                        model, source_range, cells_by_domain[validation], support_weight
                    )
                for alpha in ALPHAS:
                    key = f"blend_{alpha}"
                    domain_metrics = [metrics(inner_rows[domain], key) for domain in development]
                    candidates.append(
                        {
                            "model": model_name,
                            "alpha_local": alpha,
                            "development_macro_mape": float(np.mean([m["mape"] for m in domain_metrics])),
                            "development_macro_mae": float(np.mean([m["mae"] for m in domain_metrics])),
                        }
                    )
            chosen = min(candidates, key=lambda c: (c["development_macro_mape"], c["development_macro_mae"]))
            model, source_range = train_source(cells_by_domain, development, chosen["model"])
            target_rows = evaluate_target(model, source_range, cells_by_domain[outer_target], support_weight)
            selected_key = f"blend_{chosen['alpha_local']}"
            output["results"].append(
                {
                    "horizon": horizon,
                    "target": outer_target,
                    "support_weight": "inf" if np.isinf(support_weight) else support_weight,
                    "selected": chosen,
                    "raw_source": metrics(target_rows, "raw_source"),
                    "calibrated_source": metrics(target_rows, "blend_0.0"),
                    "local_passcal": metrics(target_rows, "blend_1.0"),
                    "selected_fusion": metrics(target_rows, selected_key),
                    "selection_candidates": candidates,
                    "per_cell": [
                        {
                            "held_out": row["held_out"],
                            "truth": row["truth"],
                            "selected_support": row["selected"],
                            "raw_source_prediction": row["predictions"]["raw_source"],
                            "calibrated_source_prediction": row["predictions"]["blend_0.0"],
                            "local_prediction": row["predictions"]["blend_1.0"],
                            "selected_fusion_prediction": row["predictions"][selected_key],
                        }
                        for row in target_rows
                    ],
                }
            )
    for horizon in (10, 20, 50):
        records = [r for r in output["results"] if r["horizon"] == horizon]
        output[f"macro_h{horizon}_k3"] = {
            method: {
                metric: float(np.mean([r[method][metric] for r in records]))
                for metric in ("mae", "mape")
            }
            for method in ("raw_source", "calibrated_source", "local_passcal", "selected_fusion")
        }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k.startswith("macro_")}, indent=2))
    for record in output["results"]:
        print(record["horizon"], record["target"], record["selected"])


if __name__ == "__main__":
    main()
