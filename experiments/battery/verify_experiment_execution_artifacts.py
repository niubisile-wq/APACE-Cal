"""Machine-check the non-frozen experiment package used for the paper audit."""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).parent


def load(name):
    return json.loads((HERE / name).read_text())


def finite_tree(x):
    if isinstance(x, float):
        # Positive infinity is the explicit JSON sentinel for an unweighted
        # curve distance; NaN is never admissible.
        return not math.isnan(x)
    if isinstance(x, list):
        return all(finite_tree(v) for v in x)
    if isinstance(x, dict):
        return all(finite_tree(v) for v in x.values())
    return True


def main():
    checks = []
    required = [
        "batterylife_e0_data_audit.json",
        "batterylife_e2_rankboard.json",
        "batterylife_apace_v2_stats.json",
        "batterylife_apace_cost_audit.json",
        "batterylife_apace_v2_curve_missing20.json",
        "batterylife_apace_v2_curve_missing10.json",
        "batterylife_apace_v2_curve_missing30.json",
        "batterylife_apace_v2_curve_noise0.5pct.json",
        "batterylife_apace_v2_curve_noise2pct.json",
        "batterylife_apace_v2_protocol_missing25.json",
        "batterylife_apace_v2_protocol_missing10.json",
        "batterylife_apace_v2_protocol_missing50.json",
        "batterylife_apace_v2_curve_noise1pct.json",
        "batterylife_apace_pbt_fixed_pool_plugin.json",
        "batterylife_apace_classical_backbone_plugin.json",
        "batterylife_apace_support_stability.json",
        "snu_dynamic_dataset1_blind_eval.json",
        "batterylife_naion_prelabel_manifest_v2.json",
        "batterylife_naion_blind_eval_v2.json",
        "batterylife_apace_e4_missing_ablation.json",
        "batterylife_apace_e5_primary_sensitivity.json",
        "batterylife_apace_route_stability_full.json",
        "batterylife_apace_efficiency_audit.json",
        "batterylife_apace_label_queue_robustness.json",
        "batterylife_apace_stratified_audit.json",
        "batterylife_eol_definition_audit.json",
        "batterylife_apace_stability_gate_candidate.json",
        "batterylife_apace_stability_gate_protocol_missing25.json",
        "batterylife_apace_stability_gate_protocol_missing50.json",
        "batterylife_apace_stability_gate_curve_noise0.5pct.json",
        "batterylife_matr_v3_one_shot.json",
        "batterylife_hust_xjtu_v3_one_shot.json",
        "batterylife_hust_xjtu_oracle_search.json",
        "batterylife_gpr_sequential_baseline.json",
        "batterylife_strong_selector_baselines.json",
        "batterylife_risk_return_analysis.json",
        "batterylife_cross_predictor_audit.json",
        "batterylife_multimetric_audit.json",
        "batterylife_multimetric_hierarchical_stats.json",
        "batterylife_external_candidate_contract_audit.json",
        "batterylife_zenodo_21700_expt4_prelabel.json",
        "batterylife_zenodo_21700_postlabel_audit.json",
        "batterylife_zenodo_21700_exploratory_90eol.json",
        "batterylife_zenodo_21700_expt5_prelabel.json",
        "batterylife_zenodo_21700_expt5_postlabel_audit.json",
        "batterylife_zenodo_21700_expt5_exploratory_90eol.json",
        "batterylife_uconn_mathworks_v3_one_shot.json",
    ]
    for name in required:
        checks.append({"check": f"artifact exists:{name}", "passed": (HERE / name).is_file()})
    checks.append({"check": "paper figure manifest exists", "passed": (HERE / "paper_figures/manifest.json").is_file()})

    matr = load("batterylife_matr_v3_one_shot.json")
    matr_rows = {(r["horizon"], r["label_budget_k"]): r for r in matr.get("results", [])}
    checks.append({
        "check": "MATR 169-cell external safety audit exact",
        "passed": len(matr_rows) == 12 and all(
            r["improved_same_worse_cells"] == [0, 169, 0]
            and r["relative_mape_reduction_percent"] == 0.0
            and r["paired_wilcoxon_p"] == 1.0 for r in matr_rows.values()
        ),
    })
    hx = load("batterylife_hust_xjtu_v3_one_shot.json")
    hx_rows = {(r["horizon"], r["label_budget_k"]): r for r in hx.get("results", [])}
    checks.append({
        "check": "HUST+XJTU 100-cell external safety audit exact",
        "passed": len(hx_rows) == 12 and all(
            r["improved_same_worse_cells"] == [0, 100, 0]
            and r["relative_mape_reduction_percent"] == 0.0
            and r["paired_wilcoxon_p"] == 1.0 for r in hx_rows.values()
        ),
    })
    oracle = load("batterylife_hust_xjtu_oracle_search.json")
    checks.append({
        "check": "test-aware oracle is explicitly non-method ceiling",
        "passed": oracle.get("phase") == "TEST_AWARE_ORACLE_SEARCH"
        and "upper-bound" in oracle.get("warning", ""),
    })

    strong = load("batterylife_strong_selector_baselines.json")
    checks.append({
        "check": "strong selector audit has all 72 domain/H/K rows",
        "passed": len(strong.get("results", {})) == 72
        and all(f"macro_h{h}_k{k}" in strong for h in (10, 20, 50) for k in (1, 3, 5, 10)),
    })
    gpr = load("batterylife_gpr_sequential_baseline.json")
    checks.append({
        "check": "sequential GPR K3 negative-control values are exact",
        "passed": all(
            abs(gpr[f"macro_h{h}_k3"]["random_gpr_mape"] - random_value) < 1e-9
            and abs(gpr[f"macro_h{h}_k3"]["sequential_gpr_mape"] - sequential_value) < 1e-9
            for h, random_value, sequential_value in (
                (10, 49.47923291666522, 66.31259723544943),
                (20, 50.99766539713684, 66.67083574113788),
                (50, 56.201550188177, 61.76995394483911),
            )
        ),
    })
    risk = load("batterylife_risk_return_analysis.json")
    checks.append({
        "check": "risk-return audit has primary rows and fixed margins",
        "passed": all(f"macro_h{h}_k3" in risk for h in (10, 20, 50))
        and risk.get("margins_relative_mape") == [0.01, 0.02, 0.05],
    })
    cross = load("batterylife_cross_predictor_audit.json")
    checks.append({
        "check": "cross-predictor audit covers frozen predictor family",
        "passed": len(cross.get("records", [])) == 6
        and len(cross.get("macro_by_predictor", {})) >= 9,
    })
    multi = load("batterylife_multimetric_audit.json")
    checks.append({
        "check": "multimetric audit reproduces frozen primary MAPE",
        "passed": all(
            abs(multi[f"macro_h{h}_k3"]["method"]["mape"] - expected) < 1e-9
            for h, expected in ((10, 24.27726689895908), (20, 22.46910971874686), (50, 22.942158458396136))
        ) and all(
            metric in multi["macro_h50_k3"]["method"] for metric in ("mae", "rmse", "mape", "smape")
        ),
    })
    multi_stats = load("batterylife_multimetric_hierarchical_stats.json")
    checks.append({
        "check": "multimetric hierarchical bootstrap has all primary metrics",
        "passed": all(
            all(metric in multi_stats["results"][f"h{h}_k3"] for metric in ("mae", "rmse", "mape", "smape"))
            for h in (10, 20, 50)
        ) and multi_stats.get("bootstraps") == 10000,
    })
    candidates = load("batterylife_external_candidate_contract_audit.json")
    checks.append({
        "check": "external candidate contract audit is conservative",
        "passed": candidates.get("eligible_new_strict_active_candidates") == []
        and len(candidates.get("candidates", [])) == 6,
    })
    mathworks = load("batterylife_uconn_mathworks_v3_one_shot.json")
    mathworks_rows = {
        (r["horizon"], r["label_budget_k"]): r
        for r in mathworks.get("results", [])
    }
    expected_mathworks_k3 = {
        10: (24.571731470184197, 15.302114737076499, 37.724719335939454),
        20: (25.891552114082444, 15.945937828570765, 38.41258431201677),
        50: (28.72485963220518, 14.941236976939193, 47.98499568580078),
    }
    checks.append({
        "check": "MathWorks K3 values and fallback branches are exact",
        "passed": len(mathworks_rows) == 12
        and all(
            abs(mathworks_rows[(h, 3)]["baseline"]["mape"] - values[0]) < 1e-9
            and abs(mathworks_rows[(h, 3)]["method"]["mape"] - values[1]) < 1e-9
            and abs(mathworks_rows[(h, 3)]["relative_mape_reduction_percent"] - values[2]) < 1e-9
            for h, values in expected_mathworks_k3.items()
        )
        and all(
            mathworks_rows[(h, k)]["relative_mape_reduction_percent"] == 0.0
            and mathworks_rows[(h, k)]["improved_same_worse_cells"] == [0, 27, 0]
            for h in (10, 20, 50) for k in (1, 5, 10)
        ),
    })
    mathworks_outcome = (HERE / "UCONN_MATHWORKS_EXTERNAL_OUTCOME.md").read_text(encoding="utf-8")
    paper_tables = (HERE / "PAPER_EXPERIMENT_TABLES.md").read_text(encoding="utf-8")
    checks.append({
        "check": "MathWorks K3 markdown tables use K3 rather than K1 baselines",
        "passed": "| 10 | 3 | 24.572 | 15.302 |" in mathworks_outcome
        and "| 20 | 3 | 25.892 | 15.946 |" in mathworks_outcome
        and "| 10 | 24.5717% | 15.3021% |" in paper_tables
        and "| 20 | 25.8916% | 15.9459% |" in paper_tables,
    })
    zen_pre = load("batterylife_zenodo_21700_expt4_prelabel.json")
    zen_post = load("batterylife_zenodo_21700_postlabel_audit.json")
    zen_exp = load("batterylife_zenodo_21700_exploratory_90eol.json")
    checks.append({
        "check": "21700 prelabel route and postlabel contract are explicit",
        "passed": zen_pre.get("phase") == "PRELABEL_FROZEN_MANIFEST"
        and len(zen_pre.get("active_settings", [])) == 9
        and zen_post.get("decision") == "technical_failure_no_common_80_percent_eol"
        and zen_exp.get("phase") == "EXPLORATORY_NONBLIND_90_PERCENT_EOL"
        and "Not an independent blind confirmation" in zen_exp.get("warning", ""),
    })
    ex5_pre = load("batterylife_zenodo_21700_expt5_prelabel.json")
    ex5_post = load("batterylife_zenodo_21700_expt5_postlabel_audit.json")
    ex5_exp = load("batterylife_zenodo_21700_expt5_exploratory_90eol.json")
    checks.append({
        "check": "21700 Expt5 frozen episodes and EOL boundary are explicit",
        "passed": ex5_pre.get("phase") == "PRELABEL_FROZEN_MANIFEST_V2"
        and len(ex5_pre.get("settings", [])) == 12
        and len(ex5_pre.get("active_settings", [])) == 9
        and ex5_post.get("common_80_percent_eol") is False
        and ex5_post.get("common_90_percent_eol") is True
        and ex5_exp.get("phase") == "EXPLORATORY_NONBLIND_90_PERCENT_EOL"
        and "not independent blind" in ex5_exp.get("warning", "").lower(),
    })

    e0 = load("batterylife_e0_data_audit.json")
    checks.append({"check": "E0 audit passed", "passed": e0.get("passed") is True})

    na = load("batterylife_naion_blind_eval_v2.json")
    na_rows = na.get("results", [])
    checks.append({"check": "NA-ion has twelve blind settings", "passed": len(na_rows) == 12})
    checks.append({
        "check": "NA-ion zero-dispersion fallback is exact",
        "passed": all(r["route_counts"] == {"fallback_zero_protocol_dispersion": 100}
                       and r["improved_same_worse_cells"] == [0, 34, 0]
                       and r["relative_mape_reduction_percent"] == 0.0
                       for r in na_rows),
    })

    stats = load("batterylife_apace_v2_stats.json")
    main_rows = [stats["results"][f"h{h}_k3"] for h in (10, 20, 50)]
    checks.append({"check": "three primary K3 statistical rows", "passed": len(main_rows) == 3})
    checks.append({"check": "primary bootstrap lower bounds positive", "passed": all(r["hierarchical_ci95_percent"][0] > 0 for r in main_rows)})
    checks.append({"check": "primary paired tests significant", "passed": all(r["pooled_cell_wilcoxon_p"] < 0.001 for r in main_rows)})

    rank = load("batterylife_e2_rankboard.json")
    checks.append({"check": "E2 rankboard has rows", "passed": len(rank.get("rows", [])) >= 15})
    checks.append({"check": "all rankboard values finite", "passed": finite_tree(rank)})

    gate = load("batterylife_apace_stability_gate_candidate.json")
    gate_primary = gate["macro_h50_k3"]
    checks.append({
        "check": "stability-gate clean H50/K3 preserves v2 primary result",
        "passed": (
            abs(gate_primary["baseline_mape"] - 48.53519919661475) < 1e-9
            and abs(gate_primary["method_mape"] - 22.942158458396136) < 1e-9
            and gate_primary["improved_same_worse_domains"] == [3, 3, 0]
        ),
    })

    for name in required:
        try:
            value = load(name)
            checks.append({"check": f"JSON numeric values valid:{name}", "passed": finite_tree(value)})
        except Exception as exc:
            checks.append({"check": f"read JSON:{name}", "passed": False, "error": repr(exc)})

    output = {"passed": all(c["passed"] for c in checks), "checks": checks}
    print(json.dumps(output, indent=2))
    raise SystemExit(0 if output["passed"] else 1)


if __name__ == "__main__":
    main()
