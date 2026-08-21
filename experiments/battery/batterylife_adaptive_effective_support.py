"""Nested selection of effective support size m <= available label budget K."""
import json
from pathlib import Path

import numpy as np

from batterylife_curve_aware_support import DATASETS, evaluate_domain, load_cells, paired_cell_summary, summarize


SOURCE = Path(__file__).with_name("batterylife_curve_aware_support.json")
OUTPUT = Path(__file__).with_name("batterylife_adaptive_effective_support.json")
BUDGETS = (1, 3, 5, 10)


def select_on_development(source, horizon, target, budget):
    candidates = []
    for effective_k in [k for k in BUDGETS if k <= budget]:
        record = next(
            r
            for r in source["nested_results"]
            if r["horizon"] == horizon and r["target"] == target and r["k"] == effective_k
        )
        for candidate in record["selection_candidates"]:
            candidates.append(
                {
                    "effective_k": effective_k,
                    "weight": candidate["weight"],
                    "development_macro_mape": candidate["development_macro_mape"],
                    "development_macro_mae": candidate["development_macro_mae"],
                }
            )
    return min(candidates, key=lambda c: (c["development_macro_mape"], c["development_macro_mae"])), candidates


def main():
    source = json.load(open(SOURCE))
    loaded = {(h, d): load_cells(d, h) for h in (10, 20, 50) for d in DATASETS}
    output = {
        "source": SOURCE.name,
        "protocol": "outer target excluded; choose effective m<=K and curve weight on other five domains; evaluate one prediction per held-out cell with 100 tie seeds",
        "results": [],
    }
    for horizon in (10, 20, 50):
        for target in DATASETS:
            cells = loaded[(horizon, target)]
            for budget in BUDGETS:
                chosen, candidates = select_on_development(source, horizon, target, budget)
                weight = np.inf if chosen["weight"] == "inf" else float(chosen["weight"])
                method_rows = evaluate_domain(cells, chosen["effective_k"], weight, 100)
                exact_record = next(
                    r
                    for r in source["nested_results"]
                    if r["horizon"] == horizon and r["target"] == target and r["k"] == budget
                )
                baseline_rows = evaluate_domain(cells, budget, 0.0, 100)
                output["results"].append(
                    {
                        "horizon": horizon,
                        "target": target,
                        "budget_k": budget,
                        "selected": chosen,
                        "protocol_only_exact_k": summarize(baseline_rows),
                        "curve_aware_exact_k": exact_record["nested_curve_aware"],
                        "adaptive_effective_support": summarize(method_rows),
                        "per_cell_vs_protocol": paired_cell_summary(baseline_rows, method_rows),
                        "selection_candidates": candidates,
                    }
                )
    for horizon in (10, 20, 50):
        for budget in BUDGETS:
            records = [r for r in output["results"] if r["horizon"] == horizon and r["budget_k"] == budget]
            output[f"macro_h{horizon}_k{budget}"] = {
                method: {
                    metric: float(np.mean([r[method][metric] for r in records]))
                    for metric in ("mae", "mape")
                }
                for method in ("protocol_only_exact_k", "curve_aware_exact_k", "adaptive_effective_support")
            }
            output[f"macro_h{horizon}_k{budget}"]["selected_effective_k"] = {
                str(k): sum(r["selected"]["effective_k"] == k for r in records) for k in BUDGETS if k <= budget
            }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k.startswith("macro_")}, indent=2))


if __name__ == "__main__":
    main()
