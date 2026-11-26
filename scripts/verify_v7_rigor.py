"""V7 审稿反例数值复现与真实数据异质性统计推断重算脚本。

执行内容（全离线，依赖本地已有 data/ 缓存）：
1. 复现审稿指出的四类数学反例定量行为并验证修复后函数输出：
   - P0-01 窄峰反例 sup R_h(N) >= 1/2 及 all-beta>0 统一正则类；
   - P0-03 4x4 块对角含负相关 SPD 矩阵 eta = 6/23 > 1/4；
   - P0-04 3x3 零乘子弱激活矩阵（I_+={0,1}, I_0={2}）精确 2D 数值积分比对与 Copula 次阶判据；
   - P0-05 三层混合 kappa_Z ~ 0.3103 < 1 反例（右尾依然跌向 0）与单内点充分条件。
2. 以严格不相交内部留出（inner_fit 80% / inner_val 20%）及题目级 500x Paired Cluster Bootstrap
   重算并刷新 Stage 2 五组真实异质性实验：
   - S1a RewardBench-2 Skywork 6 域异质性
   - S2 OlympiadBench uPRM-70B 跨学科子域异质性
   - S3 3-Domain (GSM8K + GPQA + AIME) 辛普森混杂与自适应切换
   - S4 AceMath RM-B / RM-A 7 数学子域异质性
3. 导出至 results/v7_rigor_verification.json，并同步更新 data/stage2/cache/*、
   results/stage2_real_results.json、results/stage2_summary.csv 及图 10。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy import integrate, special

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import plot_stage2_figures as ps2  # noqa: E402
import run_stage2_experiments as rs2  # noqa: E402

from tts_theory import (  # noqa: E402
    GaussianCopulaHeavyTailSpec,
    GeneralGaussianMultiVerifierSpec,
    diagnose_zscore_induced_collapse,
    evaluate_real_prompt_heterogeneity,
    gaussian_copula_heavy_tail_criterion,
    general_min_ensemble_qp_criterion,
    general_min_tail_asymptotic_constants,
    solve_positive_quadrant_qp,
    tight_joint_minimax_scaling,
)


def verify_counterexamples() -> dict[str, Any]:
    """定量复现并检验全部审稿反例。"""
    # P0-01: 窄峰反例 q_h(s) = 1/4 + 1/2 exp(-s^2/h^2)
    n_budget = 10
    p0_01_rows = []
    for h in (1.0, 0.2, 0.05, 0.005):
        def _integrand(r: float) -> float:
            sf_abs = 2.0 * float(special.ndtr(-r))
            pdf_abs = 2.0 * math.exp(-0.5 * r * r) / math.sqrt(2.0 * math.pi)
            pdf_r = float(n_budget) * pdf_abs * (sf_abs ** (n_budget - 1))
            return math.exp(-(r * r) / (h * h)) * pdf_r

        ex_exp, _ = integrate.quad(_integrand, 0.0, min(10.0, 12.0 * h + 1.0), epsabs=1e-12, epsrel=1e-10)
        reg = 0.5 * (1.0 - ex_exp)
        lb = 0.5 - (float(n_budget) * h) / (2.0 * math.sqrt(h * h + 2.0))
        p0_01_rows.append({"h": h, "exact_regret_n10": reg, "analytic_lb": lb})

    bound_b05 = tight_joint_minimax_scaling(m_total=10_000, n_budget=64, beta=0.5)
    bound_b20 = tight_joint_minimax_scaling(m_total=10_000, n_budget=64, beta=2.0)

    # P0-03: 4x4 块对角含负相关 SPD 矩阵
    r_block = np.array(
        [
            [1.0, -0.2, 0.0, 0.0],
            [-0.2, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.5, 1.0],
        ],
        dtype=np.float64,
    )
    cop_b = gaussian_copula_heavy_tail_criterion(
        GaussianCopulaHeavyTailSpec(p=0.25, alpha1=2.0, alpha0=2.0, corr1=np.eye(4), corr0=r_block)
    )

    # P0-04: 3x3 零乘子弱激活矩阵
    r_deg = np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.5],
            [0.5, 0.5, 1.0],
        ],
        dtype=np.float64,
    )
    qp_deg = solve_positive_quadrant_qp(r_deg)
    c_deg = general_min_tail_asymptotic_constants(np.zeros(3), r_deg)
    inv_2pi = 1.0 / (2.0 * math.pi)
    t_checks = []
    for t_val in (20.0, 100.0, 500.0):
        def _scaled(x2: float, x1: float, t: float = t_val) -> float:
            return (
                inv_2pi
                * math.exp(-(x1 + x2) - 0.5 * ((x1 * x1 + x2 * x2) / (t * t)))
                * float(special.ndtr((x1 + x2) / (t * math.sqrt(2.0))))
            )
        scaled_sf, _ = integrate.dblquad(_scaled, 0.0, 18.0, 0.0, 18.0, epsabs=0.0, epsrel=1e-9)
        t_checks.append(
            {
                "t": t_val,
                "scaled_exact_sf_t2_exp_t2": scaled_sf,
                "c_sf_v7": float(c_deg["c_sf"]),
                "ratio": scaled_sf / float(c_deg["c_sf"]),
            }
        )
    r_equi = np.full((3, 3), 0.25, dtype=np.float64)
    np.fill_diagonal(r_equi, 1.0)
    cop_deg = gaussian_copula_heavy_tail_criterion(
        GaussianCopulaHeavyTailSpec(p=0.25, alpha1=2.0, alpha0=2.0, corr1=r_equi, corr0=r_deg)
    )
    g_deg = general_min_ensemble_qp_criterion(
        GeneralGaussianMultiVerifierSpec(p=0.25, mu1=np.zeros(3), cov1=r_equi, mu0=np.zeros(3), cov0=r_deg)
    )

    # P0-05: 审稿人 kappa_Z ~ 0.3103 < 1 反例 + 单内点充分条件
    z_g = np.linspace(-3.5, 9.0, 1251)
    d_ce = diagnose_zscore_induced_collapse(
        prompt_weights=[0.01, 0.495, 0.495],
        prompt_p=[0.0, 0.1, 0.2],
        mu1_base=100.0,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
        z_grid=z_g,
    )
    d_suff = diagnose_zscore_induced_collapse(
        prompt_weights=[0.55, 0.45],
        prompt_p=[0.0, 0.50],
        mu1_base=1.2,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
        z_grid=z_g,
    )

    return {
        "p0_01_narrow_peak": {
            "n_budget": n_budget,
            "regrets_by_h": p0_01_rows,
            "uniform_class_bound_beta05": bound_b05,
            "uniform_class_bound_beta20": bound_b20,
        },
        "p0_03_block_negative_corr": {
            "gamma": float(1.0 / float(cop_b["eta0_copula"])),
            "eta0_copula": cop_b["eta0_copula"],
            "is_nonneg_corr0": cop_b["is_nonneg_corr0"],
            "independent_floor": 0.25,
        },
        "p0_04_degenerate_zero_multiplier": {
            "gamma": qp_deg["gamma"],
            "multipliers": qp_deg["multipliers"],
            "active_set": qp_deg["active_set"],
            "essential_set": qp_deg["essential_set"],
            "weakly_active_set": qp_deg["weakly_active_set"],
            "strict_complementarity": qp_deg["strict_complementarity"],
            "orthant_prob": c_deg["orthant_prob"],
            "c_sf": c_deg["c_sf"],
            "c_pdf": c_deg["c_pdf"],
            "exact_2d_quadrature_ratios": t_checks,
            "copula_log_tiebreak_regime": cop_deg["regime"],
            "copula_log_factor_exp1": cop_deg["log_factor_exp1"],
            "copula_log_factor_exp0": cop_deg["log_factor_exp0"],
            "gaussian_min_log_tiebreak_regime": g_deg["regime"],
        },
        "p0_05_zscore_kappa_lt_1": {
            "counterexample": {
                "var_z_pos": d_ce["var_z_pos"],
                "within_var_z_pos": d_ce["within_var_z_pos"],
                "between_var_z_pos": d_ce["between_var_z_pos"],
                "var_z_neg": d_ce["var_z_neg"],
                "kappa_z": d_ce["kappa_z"],
                "z_peak": d_ce["z_peak"],
                "q_z_peak": d_ce["q_z_peak"],
                "z_tail_value": d_ce["z_tail_value"],
                "z_tail_posterior_limit": d_ce["z_tail_posterior_limit"],
                "tail_turn_down": d_ce["tail_turn_down"],
                "raw_bon_asymptotic_acc": d_ce["raw_bon_asymptotic_acc"],
                "oracle_z_cal_asymptotic_acc": d_ce["oracle_z_cal_asymptotic_acc"],
                "asymptotic_ceiling_loss": d_ce["asymptotic_ceiling_loss"],
            },
            "single_interior_sufficient_case": {
                "between_var_z_pos": d_suff["between_var_z_pos"],
                "kappa_z": d_suff["kappa_z"],
                "z_tail_posterior_limit": d_suff["z_tail_posterior_limit"],
                "asymptotic_ceiling_loss": d_suff["asymptotic_ceiling_loss"],
            },
        },
    }


def _recompute_rb2_het() -> dict[str, Any]:
    rb2_root = rs2.STAGE2_DIR / "reward_bench_2"
    loaded = {name: rs2._load_rb2_model(rb2_root / rel) for name, rel in rs2.RB2_HETERO6.items()}
    _, h6_mats, h6_ys, h6_strata = rs2._align_rb2_group(loaded)
    pools_strata = [(st, mat[:, 0], y) for st, mat, y in zip(h6_strata, h6_mats, h6_ys)]
    return evaluate_real_prompt_heterogeneity(
        pools_strata,
        n_values=[1, 2, 3, 4],
        n_folds=5,
        n_trials_per_problem=20,
        n_bins=15,
        n_simpson_bins=10,
        seed=107,
        n_bootstrap=500,
    )


def _recompute_olympiad_het() -> dict[str, Any]:
    oly_root = rs2.STAGE2_DIR / "olympiadbench"
    repo_70b = "sibasmarakp/Qwen2.5-7B-Instruct-uPRM-70B-T80-olympiadbench-best_of_n-completions"
    repo_32b = "sibasmarakp/Qwen2.5-7B-Instruct-uPRM-32B-T80-olympiadbench-best_of_n-completions"
    p70, _ = rs2._parse_olympiad_repo(repo_70b, oly_root / "uprm_70b")
    p32, _ = rs2._parse_olympiad_repo(repo_32b, oly_root / "uprm_32b")
    map70 = {p["id"]: p for p in p70}
    map32 = {p["id"]: p for p in p32}
    common_ids = sorted(set(map70.keys()) & set(map32.keys()))
    aligned_70 = [map70[pid] for pid in common_ids]
    return evaluate_real_prompt_heterogeneity(
        [(p["subfield"], p["prm_last"], p["is_correct"]) for p in aligned_70],
        n_values=[1, 2, 4, 8, 16, 32, 64, 128],
        n_folds=5,
        n_trials_per_problem=20,
        n_bins=15,
        n_simpson_bins=10,
        seed=613,
        n_bootstrap=500,
    )


def _recompute_s3_3domain_het() -> dict[str, Any]:
    parsed = {tag: rs2._stream_parse_strategic_ttc(tag, rel) for tag, rel in rs2.TTC_FILES.items()}
    pooled_3domain: list[tuple[str, np.ndarray, np.ndarray]] = []
    for dom_name, tag in (("GSM8K", "gsm8k_qwen7b"), ("GPQA", "gpqa_qwen7b"), ("AIME", "aime_qwen7b")):
        for x in parsed[tag]:
            pooled_3domain.append((
                dom_name,
                np.asarray(x["rewards"], dtype=np.float64),
                np.asarray(x["correct"], dtype=bool),
            ))
    return evaluate_real_prompt_heterogeneity(
        pooled_3domain,
        n_values=[1, 2, 4, 8, 16, 32, 64, 128],
        n_folds=5,
        n_trials_per_problem=20,
        n_bins=15,
        n_simpson_bins=12,
        seed=760,
        n_bootstrap=500,
    )


def _recompute_acemath_het() -> tuple[dict[str, Any], dict[str, Any]]:
    ace_dst = rs2.DATA_DIR / "acemath_rewardbench" / "default_test_0.parquet"
    tbl = pq.read_table(ace_dst)
    rm_all = tbl["rm_score"].to_pylist()
    corr_all = tbl["score"].to_pylist()
    offset_b = 8987
    all_probs = []
    for sub_name, s0, s1 in rs2.ACEMATH_SEGMENTS:
        for idx in range(s0, s1):
            all_probs.append({
                "subset": sub_name,
                "rm_a": np.asarray(rm_all[idx], dtype=np.float64),
                "rm_b": np.asarray(rm_all[offset_b + idx], dtype=np.float64),
                "is_correct": np.asarray(corr_all[idx], dtype=bool),
            })
    n_values = [1, 2, 4, 8, 16, 32, 64]
    het_b = evaluate_real_prompt_heterogeneity(
        [(p["subset"], p["rm_b"], p["is_correct"]) for p in all_probs],
        n_values=n_values,
        n_folds=5,
        n_trials_per_problem=12,
        n_bins=15,
        n_simpson_bins=12,
        seed=830,
        n_bootstrap=500,
    )
    het_a = evaluate_real_prompt_heterogeneity(
        [(p["subset"], p["rm_a"], p["is_correct"]) for p in all_probs],
        n_values=n_values,
        n_folds=5,
        n_trials_per_problem=12,
        n_bins=15,
        n_simpson_bins=12,
        seed=831,
        n_bootstrap=500,
    )
    return het_b, het_a


def refresh_real_heterogeneity_and_stage2_outputs() -> dict[str, Any]:
    """重算 5 组真实异质性基准并同步刷新 Stage 2 缓存、JSON/CSV 汇总及图 10。"""
    print("[V7] Recomputing S1a RB2 Skywork heterogeneity (disjoint inner split + 500x cluster bootstrap)...")
    het_rb2 = rs2.to_json_safe(_recompute_rb2_het())
    print("[V7] Recomputing S2 OlympiadBench heterogeneity...")
    het_oly = rs2.to_json_safe(_recompute_olympiad_het())
    print("[V7] Recomputing S3 3-Domain heterogeneity...")
    het_3dom = rs2.to_json_safe(_recompute_s3_3domain_het())
    print("[V7] Recomputing S4 AceMath 7-strata heterogeneity (RM-B & RM-A)...")
    het_b_raw, het_a_raw = _recompute_acemath_het()
    het_b = rs2.to_json_safe(het_b_raw)
    het_a = rs2.to_json_safe(het_a_raw)

    # 更新本地阶段缓存
    cache_updates = [
        ("s1a_rb2", [("rb2_skywork_heterogeneity", het_rb2)]),
        ("s2_olympiadbench", [("olympiad_subfield_heterogeneity", het_oly)]),
        ("s3_strategic_ttc", [("cross_3domain_simpson_het", het_3dom)]),
        ("s4_acemath", [("het_7strata_rm_b", het_b), ("het_7strata_rm_a", het_a)]),
    ]
    for cname, pairs in cache_updates:
        cpath = rs2.CACHE_DIR / f"{cname}.json"
        if cpath.exists():
            cobj = json.loads(cpath.read_text(encoding="utf-8"))
            for key, val in pairs:
                cobj[key] = val
            cpath.write_text(json.dumps(cobj, indent=2, ensure_ascii=False), encoding="utf-8")

    # 同步刷新 results/stage2_real_results.json、stage2_summary.csv 及图 10
    s2_json_path = rs2.RESULTS_DIR / "stage2_real_results.json"
    if s2_json_path.exists():
        payload = json.loads(s2_json_path.read_text(encoding="utf-8"))
        payload["s1a_rb2"]["rb2_skywork_heterogeneity"] = het_rb2
        payload["s2_olympiadbench"]["olympiad_subfield_heterogeneity"] = het_oly
        payload["s3_strategic_ttc"]["cross_3domain_simpson_het"] = het_3dom
        payload["s4_acemath"]["het_7strata_rm_b"] = het_b
        payload["s4_acemath"]["het_7strata_rm_a"] = het_a
        s2_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        rs2.export_summary_csv(payload, rs2.RESULTS_DIR / "stage2_summary.csv")
        ps2.plot_fig10_simpson_and_cross_domain(
            payload, rs2.RESULTS_DIR / "figures/stage2_real_simpson_and_cross_domain.png"
        )

    return {
        "rb2_skywork_heterogeneity": het_rb2,
        "olympiad_subfield_heterogeneity": het_oly,
        "cross_3domain_simpson_het": het_3dom,
        "het_7strata_rm_b": het_b,
        "het_7strata_rm_a": het_a,
    }


def main() -> None:
    ce = rs2.to_json_safe(verify_counterexamples())
    real_het = refresh_real_heterogeneity_and_stage2_outputs()
    out = {
        "counterexamples": ce,
        "real_heterogeneity_v7": real_het,
    }
    out_path = PROJECT_ROOT / "results/v7_rigor_verification.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved:", out_path)

    print("[P0-01 Narrow Peak N=10]", ce["p0_01_narrow_peak"]["regrets_by_h"])
    print("[P0-03 Block Neg Corr]", ce["p0_03_block_negative_corr"])
    print("[P0-04 Degenerate 3x3]", ce["p0_04_degenerate_zero_multiplier"])
    print("[P0-05 kappa_Z < 1]", ce["p0_05_zscore_kappa_lt_1"]["counterexample"])

    for label, blk in real_het.items():
        nv = blk["n_values"][-1]
        algs = blk["algorithms"]
        paired = blk["paired_bootstrap_vs_pooled"]
        p_z = paired["Within-Prompt-Z-KL-LCB"]
        p_h = paired["Hybrid-Adaptive-KL-LCB"]
        print(
            f"[Real {label} N={nv}] "
            f"Std={algs['Standard-BoN']['accuracy'][-1]:.4f}+/-{algs['Standard-BoN']['se'][-1]:.4f} | "
            f"Pool={algs['Pooled-KL-LCB']['accuracy'][-1]:.4f}+/-{algs['Pooled-KL-LCB']['se'][-1]:.4f} | "
            f"Z={algs['Within-Prompt-Z-KL-LCB']['accuracy'][-1]:.4f}+/-{algs['Within-Prompt-Z-KL-LCB']['se'][-1]:.4f} "
            f"(diff={p_z['mean_diff'][-1]:+.4f}, 95%CI=[{p_z['ci_low'][-1]:+.4f},{p_z['ci_high'][-1]:+.4f}], p={p_z['p_value'][-1]:.4f}) | "
            f"Hyb={algs['Hybrid-Adaptive-KL-LCB']['accuracy'][-1]:.4f}+/-{algs['Hybrid-Adaptive-KL-LCB']['se'][-1]:.4f} "
            f"(diff={p_h['mean_diff'][-1]:+.4f}, p={p_h['p_value'][-1]:.4f})"
        )


if __name__ == "__main__":
    main()
