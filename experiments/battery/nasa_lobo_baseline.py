"""NASA PCoE battery RUL data-chain baseline with leave-one-battery-out split."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


DATA = Path(__file__).resolve().parents[2] / "external/nasa_battery_aging/discharge.csv"


def make_rows(df: pd.DataFrame, window: int = 10) -> tuple[pd.DataFrame, dict]:
    rows = []
    eols = {}
    for battery, g in df.groupby("Battery", sort=True):
        cap = g.groupby("id_cycle")["Capacity"].first().sort_index()
        initial = float(cap.iloc[:3].median())
        candidates = cap[cap <= 0.70 * initial]
        eol = int(candidates.index[0]) if len(candidates) else int(cap.index.max())
        eols[battery] = {"initial_capacity": initial, "eol_cycle": eol,
                         "n_cycles": int(cap.index.max())}
        cycle = g.groupby("id_cycle").agg(
            capacity=("Capacity", "first"), voltage_mean=("Voltage_measured", "mean"),
            voltage_min=("Voltage_measured", "min"), temp_mean=("Temperature_measured", "mean"),
            current_mean=("Current_measured", "mean"), time_mean=("Time", "mean"))
        cycle = cycle.sort_index()
        for c in cycle.index:
            if c < max(5, window):
                continue
            hist = cycle.loc[:c].tail(window)
            cap_vals = hist.capacity.to_numpy()
            slope = float(np.polyfit(np.arange(len(cap_vals)), cap_vals, 1)[0]) if len(cap_vals) >= 2 else 0.0
            last = hist.iloc[-1]
            rows.append({"battery": battery, "cycle": int(c), "rul": max(eol - int(c), 0),
                         "capacity": float(last.capacity), "capacity_slope": slope,
                         "voltage_mean": float(last.voltage_mean), "voltage_min": float(last.voltage_min),
                         "temp_mean": float(last.temp_mean), "current_mean": float(last.current_mean),
                         "time_mean": float(last.time_mean)})
    return pd.DataFrame(rows), eols


def evaluate(rows: pd.DataFrame, model_name: str) -> dict:
    feats = [c for c in rows.columns if c not in ("battery", "cycle", "rul")]
    all_rows = []
    for held in sorted(rows.battery.unique()):
        tr = rows[rows.battery != held]; te = rows[rows.battery == held]
        if model_name == "ridge":
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        else:
            model = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                           random_state=819, n_jobs=-1)
        model.fit(tr[feats], tr.rul)
        pred = model.predict(te[feats])
        for actual, estimate, cyc in zip(te.rul, pred, te.cycle):
            all_rows.append({"held_battery": held, "cycle": int(cyc),
                             "actual_rul": float(actual), "pred_rul": float(estimate),
                             "abs_error": float(abs(actual - estimate))})
    out = pd.DataFrame(all_rows)
    by_battery = out.groupby("held_battery").abs_error.mean().to_dict()
    return {"model": model_name, "n_predictions": len(out),
            "mae": float(out.abs_error.mean()), "median_ae": float(out.abs_error.median()),
            "worst_battery_mae": float(max(by_battery.values())),
            "per_battery_mae": {str(k): float(v) for k, v in by_battery.items()},
            "predictions": all_rows}


def main() -> None:
    df = pd.read_csv(DATA)
    rows, battery_meta = make_rows(df, window=10)
    results = [evaluate(rows, name) for name in ("ridge", "rf")]
    out = {"source": str(DATA), "shape": list(df.shape), "batteries": sorted(df.Battery.unique()),
           "battery_meta": battery_meta, "feature_rows": len(rows), "results": results,
           "note": "Strict leave-one-battery-out baseline on four NASA batteries; not cross-chemistry or paper-level."}
    path = Path(__file__).with_name("nasa_lobo_baseline.json")
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("battery_meta", "results")}, indent=2))
    for r in results: print(r["model"], {k: r[k] for k in r if k != "predictions"})


if __name__ == "__main__":
    main()
