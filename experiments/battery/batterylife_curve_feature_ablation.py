"""Feature-group ablation under the already selected outer-LODO weights."""
import json
from pathlib import Path

import numpy as np

from batterylife_curve_aware_support import evaluate_domain, load_cells, summarize


SOURCE = Path(__file__).with_name("batterylife_curve_aware_support.json")
OUTPUT = Path(__file__).with_name("batterylife_curve_feature_ablation.json")

# curve_features consists of five 8-channel snapshots, eight slopes and eight
# first-to-last deltas.  Keep the feature construction and distance weight
# fixed, and remove complementary channel groups.
GROUP_CHANNELS = {
    "capacity_only": (0, 1),
    "electrical_time_only": (2, 3, 4, 5),
    "thermal_resistance_only": (6, 7),
}


def feature_mask(channels):
    return [8 * snapshot + channel for snapshot in range(5) for channel in channels] + [
        40 + channel for channel in channels
    ] + [48 + channel for channel in channels]


def main():
    source = json.load(open(SOURCE))
    output = {
        "source": SOURCE.name,
        "protocol": "outer-LODO curve weight held fixed; remove curve feature groups; 100 tie seeds",
        "results": {},
    }
    loaded = {(h, d): load_cells(d, h) for h in (10, 20, 50) for d in ("CALB", "HNEI", "MICH_EXP", "CALCE", "MICH", "SNL")}
    for record in source["nested_results"]:
        if record["k"] != 3:
            continue
        horizon, target = record["horizon"], record["target"]
        weight = record["selected_curve_weight"]
        weight = np.inf if weight == "inf" else float(weight)
        result = {
            "selected_curve_weight": record["selected_curve_weight"],
            "protocol_only": record["protocol_only"],
            "full_curve": record["nested_curve_aware"],
        }
        for name, channels in GROUP_CHANNELS.items():
            rows = evaluate_domain(loaded[(horizon, target)], 3, weight, 100, feature_mask(channels))
            result[name] = summarize(rows)
        output["results"][f"h{horizon}_{target}"] = result
    for horizon in (10, 20, 50):
        records = [v for key, v in output["results"].items() if key.startswith(f"h{horizon}_")]
        output[f"macro_h{horizon}_k3"] = {
            method: {
                metric: float(np.mean([r[method][metric] for r in records]))
                for metric in ("mae", "mape")
            }
            for method in ("protocol_only", "capacity_only", "electrical_time_only", "thermal_resistance_only", "full_curve")
        }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k.startswith("macro_")}, indent=2))


if __name__ == "__main__":
    main()
