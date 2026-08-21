"""Auditable post-MATR development checks for the APACE-Cal v2 candidate.

This is explicitly not a blind-confirmation analysis: MATR motivated v2.  It
combines the original six development domains with the now-seen MATR failure
domain, reports rho-threshold sensitivity, and performs paired hierarchical
bootstrap inference at K=3.  No model is fitted in this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


HERE = Path(__file__).parent
DEFAULT_OUTPUT = HERE / "batterylife_apace_v2_postblind_audit.json"
THRESHOLDS = (0.25, 0.30, 0.35, 0.40, 0.45)
MATR_RHO = {10: 0.2605065534670599, 20: 0.2977465331975196,
            50: 0.2994379735304507}


def keyed(rows):
    return {(r["horizon"], r.get("target", r.get("domain")),
             r.get("label_budget_k", 3)): r for r in rows}


def summary(rows):
    rel = [(r["method"]["mape"] - r["baseline"]["mape"])
           / max(r["baseline"]["mape"], 1e-12) for r in rows]
    return {
        "n_domains": len(rows),
        "baseline_macro_mape": float(np.mean([r["baseline"]["mape"] for r in rows])),
        "method_macro_mape": float(np.mean([r["method"]["mape"] for r in rows])),
        "relative_macro_mape_reduction_percent": float(
            100 * (1 - np.mean([r["method"]["mape"] for r in rows])
                   / np.mean([r["baseline"]["mape"] for r in rows]))),
        "worst_domain_relative_mape_change": float(max(rel)),
        "improved_same_worse_domains": [
            int(sum(x < -1e-12 for x in rel)),
            int(sum(abs(x) <= 1e-12 for x in rel)),
            int(sum(x > 1e-12 for x in rel)),
        ],
    }


def bootstrap(rows, reps=20000, seed=20260820):
    rng = np.random.default_rng(seed)
    domains = []
    for r in rows:
        diffs = np.asarray([x["baseline_ape"] - x["method_ape"]
                            for x in r["per_cell"]], dtype=float)
        domains.append(diffs)
    observed = float(np.mean([x.mean() for x in domains]))
    domain_boot = np.empty(reps)
    hierarchical = np.empty(reps)
    n = len(domains)
    means = np.asarray([x.mean() for x in domains])
    for i in range(reps):
        selected = rng.integers(0, n, n)
        domain_boot[i] = means[selected].mean()
        hierarchical[i] = np.mean([
            domains[j][rng.integers(0, len(domains[j]), len(domains[j]))].mean()
            for j in selected
        ])
    statistic, pvalue = wilcoxon(means, alternative="greater", zero_method="wilcox")
    return {
        "estimand": "macro-domain mean paired MAPE reduction (percentage points)",
        "observed": observed,
        "domain_bootstrap_95_ci": np.percentile(domain_boot, [2.5, 97.5]).tolist(),
        "cell_within_domain_then_domain_bootstrap_95_ci":
            np.percentile(hierarchical, [2.5, 97.5]).tolist(),
        "one_sided_paired_domain_wilcoxon_statistic": float(statistic),
        "one_sided_paired_domain_wilcoxon_p": float(pvalue),
        "bootstrap_repetitions": reps,
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", type=Path,
                        default=HERE / "batterylife_asymmetric_cohort_router_v2.json")
    parser.add_argument("--hybrid", type=Path,
                        default=HERE / "batterylife_high_dispersion_hybrid_search.json")
    parser.add_argument("--matr-blind", type=Path,
                        default=HERE / "batterylife_matr_blind_eval.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    v2 = json.loads(args.v2.read_text())
    hybrid = json.loads(args.hybrid.read_text())
    blind = json.loads(args.matr_blind.read_text())
    v2k = keyed(v2["results"])
    hyk = keyed(hybrid["results"])
    blk = {(r["horizon"], r["label_budget_k"]): r for r in blind["results"]}

    # Reconstruct the predictor that an all-six, label-free external deployment
    # would carry into MATR. Each LODO ranking contains five of six domains, so
    # averaging the six LODO scores exactly recovers the all-six macro ranking.
    external_baselines = {}
    for h in (10, 20, 50):
        for k in (1, 3, 5, 10):
            scores = {}
            for domain in ("CALB", "HNEI", "MICH_EXP", "CALCE", "MICH", "SNL"):
                for row in v2k[(h, domain, k)]["predictor_ranking"]:
                    scores.setdefault(row["predictor"], []).append(
                        row["development_macro_mape"])
            ranking = sorted(
                ({"predictor": p, "six_domain_macro_mape": float(np.mean(x))}
                 for p, x in scores.items()),
                key=lambda x: (x["six_domain_macro_mape"], x["predictor"]),
            )
            external_baselines[f"h{h}_k{k}"] = {
                "selected": ranking[0], "ranking": ranking
            }

    seven = []
    domains = ("CALB", "HNEI", "MICH_EXP", "CALCE", "MICH", "SNL")
    for h in (10, 20, 50):
        for k in (1, 3, 5, 10):
            seven.extend(v2k[(h, d, k)] for d in domains)
            base = blk[(h, k)]
            if k == 3:
                med = hyk[(h, "MATR", 3)]["hybrid"]["a0_median"]
                method = {x["held_out"]: x for x in med["per_cell"]}
                per_cell = [{
                    "held_out": x["held_out"],
                    "baseline_ape": x["baseline_ape"],
                    "method_ape": method[x["held_out"]]["ape"],
                    "baseline_ae": x["baseline_ae"],
                    "method_ae": method[x["held_out"]]["ae"],
                } for x in base["per_cell"]]
                method_summary = {"mape": med["mape"], "mae": med["mae"],
                                  "n_cells": len(per_cell)}
                predictor = "support_median"
            else:
                per_cell = [{
                    "held_out": x["held_out"],
                    "baseline_ape": x["baseline_ape"],
                    "method_ape": x["baseline_ape"],
                    "baseline_ae": x["baseline_ae"],
                    "method_ae": x["baseline_ae"],
                } for x in base["per_cell"]]
                method_summary = dict(base["baseline"])
                predictor = "fallback_identical_to_baseline"
            seven.append({
                "horizon": h, "target": "MATR", "label_budget_k": k,
                "protocol_dispersion": base["protocol_dispersion"],
                "distance_concordance_spearman": MATR_RHO[h],
                "method_predictor": predictor,
                "baseline": base["baseline"], "method": method_summary,
                "per_cell": per_cell,
                "provenance": "post-blind development using the disclosed failed MATR trial",
            })

    macros = {}
    inference = {}
    for h in (10, 20, 50):
        for k in (1, 3, 5, 10):
            rows = [r for r in seven if r["horizon"] == h
                    and r["label_budget_k"] == k]
            macros[f"h{h}_k{k}"] = summary(rows)
            if k == 3:
                inference[f"h{h}_k{k}"] = bootstrap(rows)

    sensitivity = {}
    high_domains = ("MICH_EXP", "SNL", "MATR")
    for threshold in THRESHOLDS:
        threshold_rows = []
        for h in (10, 20, 50):
            for domain in high_domains:
                if domain == "MATR":
                    rho = MATR_RHO[h]
                    baseline = blk[(h, 3)]["baseline"]
                else:
                    row = v2k[(h, domain, 3)]
                    rho, baseline = row["distance_concordance_spearman"], row["baseline"]
                candidate = hyk[(h, domain, 3)]["hybrid"][
                    "a0_median" if rho < threshold else "a0_kernel"]
                threshold_rows.append({"horizon": h, "domain": domain, "rho": rho,
                                       "baseline": baseline, "method": candidate})
        rel = [(r["method"]["mape"] - r["baseline"]["mape"])
               / r["baseline"]["mape"] for r in threshold_rows]
        sensitivity[f"rho_lt_{threshold:g}"] = {
            "method_macro_mape": float(np.mean([r["method"]["mape"]
                                                 for r in threshold_rows])),
            "baseline_macro_mape": float(np.mean([r["baseline"]["mape"]
                                                   for r in threshold_rows])),
            "improved_same_worse_settings": [int(sum(x < -1e-12 for x in rel)),
                                               int(sum(abs(x) <= 1e-12 for x in rel)),
                                               int(sum(x > 1e-12 for x in rel))],
            "worst_setting_relative_mape_change": float(max(rel)),
            "routing": [{"horizon": r["horizon"], "domain": r["domain"],
                          "rho": r["rho"],
                          "predictor": "support_median" if r["rho"] < threshold
                          else "w0.5_bw0.5"} for r in threshold_rows],
        }

    output = {
        "status": "POST-BLIND DEVELOPMENT AUDIT; NOT INDEPENDENT CONFIRMATION",
        "fixed_problem": "asymmetric protocol-aware few-shot target-domain calibration",
        "evidence_boundary": (
            "MATR is included only as a disclosed seen failure-development domain; "
            "a different untouched domain is required for confirmation."),
        "external_baseline_reconstruction": external_baselines,
        "seven_domain_results": seven,
        "seven_domain_macro": macros,
        "k3_inference": inference,
        "rho_threshold_sensitivity_high_dispersion_k3": sensitivity,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"seven_domain_macro": macros, "k3_inference": inference,
                      "rho_sensitivity": sensitivity}, indent=2))


if __name__ == "__main__":
    main()
