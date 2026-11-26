"""V8 二审数学边界核验与真实数据三分支异质性统计推断重算脚本。

执行内容（全离线，依赖本地已有 data/ 缓存）：
1. 数值核验 V8 全部数学修复项：
   - P0-01 定理 7B 同方差单调高斯 N=2 纯秩等号反例（0.630125）与异方差严格不等号；
   - P0-02 定理 6C beta=0.25 <= 0.5 下样本分割 Uniform 顺序统计量 Gamma 间距界 vs DKW 发散；
   - P1-01 定理 7A 显式有限参数二层高斯模型在闭区间 [3.5, 4.2] 上的逐层负斜率与池化正斜率；
   - P1-02 定理 2B(4) kappa < 1, delta_mu = 0 边界初值平局后严格递增；
   - P1-04 定理 5B-II K=1 与 K>=2 正定相关矩阵 eta_c 精确边界分类；
   - P1-05 定理 7C(4) 在线样本题内标准化 Z_{i,N} 一致极值收敛 O_p(sqrt((ln N)/N)) 与渐近上限损失。
2. 以三分支内部留出门控 {Standard-BoN, Pooled-KL-LCB, Within-Prompt-Z-KL-LCB}、
   2000x 题目级 Paired Cluster Bootstrap 及 10000x Phipson-Smyth 对称符号翻转置换检验
   重算并刷新 Stage 2 五组真实异质性实验：
   - S1a RewardBench-2 Skywork 6 域异质性
   - S2 OlympiadBench uPRM-70B 跨学科子域异质性
   - S3 3-Domain (GSM8K + GPQA + AIME) 辛普森混杂与三分支自适应切换
   - S4 AceMath RM-B / RM-A 7 数学子域异质性
3. 导出至 results/v8_rigor_verification.json，并同步更新 data/stage2/cache/*、
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
from scipy import special

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import plot_stage2_figures as ps2  # noqa: E402
import run_stage2_experiments as rs2  # noqa: E402

from tts_theory import (  # noqa: E402
    GaussianCopulaHeavyTailSpec,
    GaussianVerifierSpec,
    PromptHeterogeneitySpec,
    calibration_minimax_scaling,
    check_collapse_criterion,
    diagnose_simpson_confounding,
    diagnose_within_prompt_residual,
    empirical_quantile_spacings_bounds,
    evaluate_real_prompt_heterogeneity,
    exact_bon_curve,
    gaussian_copula_heavy_tail_criterion,
    verify_sample_zscore_extremal_convergence,
)


def verify_v8_math_checks() -> dict[str, Any]:
    """定量复现并检验二审全部数学修复项。"""
    # P0-01: N=2 单调同方差高斯等号反例 vs 异方差严格不等号
    spec_mono = PromptHeterogeneitySpec(
        prompt_weights=[1.0],
        prompt_p=[0.5],
        prompt_shift=[0.0],
        mu1_base=1.0,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
    )
    d_mono = diagnose_within_prompt_residual(spec_mono)
    analytic_mono_n2 = 0.5 + 0.25 * (2.0 * float(special.ndtr(1.0 / math.sqrt(2.0))) - 1.0)

    spec_het = PromptHeterogeneitySpec(
        prompt_weights=[1.0],
        prompt_p=[0.5],
        prompt_shift=[0.0],
        mu1_base=0.5,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=2.0,
    )
    d_het = diagnose_within_prompt_residual(spec_het)

    # P0-02: beta = 0.25 <= 0.5 下 Gamma 间距界 vs DKW 相对误差界（K = (M / ln M)^{2/3}）
    spacings_rows = []
    for m_val in (24_000, 250_000, 2_500_000):
        k_opt = int(round((float(m_val) / math.log(float(m_val))) ** (2.0 / 3.0)))
        sp = empirical_quantile_spacings_bounds(m_total=m_val, n_bins=k_opt, delta=0.01)
        spacings_rows.append(
            {
                "m_total": m_val,
                "k_opt_beta025": k_opt,
                "d_min": sp["d_min"],
                "d_min_required": sp["d_min_required"],
                "spacings_failure_prob_upper": sp["spacings_failure_prob_upper"],
                "spacings_condition_holds": sp["spacings_condition_holds"],
                "dkw_relative_error_bound": sp["dkw_relative_error_bound"],
                "dkw_non_vacuous": sp["dkw_non_vacuous"],
            }
        )

    # P1-01: 定理 7A 显式有限参数二层高斯在 [3.5, 4.2] 上的辛普森遮蔽
    spec_simp = PromptHeterogeneitySpec(
        prompt_weights=[0.85, 0.15],
        prompt_p=[0.05, 0.95],
        prompt_shift=[0.0, 3.0],
        prompt_scale=[1.0, 1.0],
        mu1_base=0.3,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=2.0,
    )
    s_grid = np.linspace(3.5, 4.2, 701)
    st_simp = spec_simp.stratum_stats(s_grid)
    diag_simp = diagnose_simpson_confounding(spec_simp, s_grid=s_grid)

    # P1-02: kappa < 1, delta_mu = 0 边界
    spec_k_lt1 = GaussianVerifierSpec(p=0.35, mu1=0.0, sigma1=2.0, mu0=0.0, sigma0=1.0)
    crit_k_lt1 = check_collapse_criterion(spec_k_lt1)
    curve_k_lt1 = exact_bon_curve(spec_k_lt1, [1, 2, 3, 4, 8, 16])

    # P1-04: eta_c 精确边界分类
    r_high = np.full((3, 3), 0.99, dtype=np.float64)
    np.fill_diagonal(r_high, 1.0)
    cop_k1 = gaussian_copula_heavy_tail_criterion(
        GaussianCopulaHeavyTailSpec(p=0.3, alpha1=4.0, alpha0=2.0, corr1=[[1.0]], corr0=[[1.0]])
    )
    cop_high = gaussian_copula_heavy_tail_criterion(
        GaussianCopulaHeavyTailSpec(p=0.3, alpha1=4.0, alpha0=2.0, corr1=r_high, corr0=r_high)
    )

    # P1-05: 在线样本 Z_{i,N} 一致极值收敛（定理 7C(4)）
    sample_z_conv = verify_sample_zscore_extremal_convergence(
        prompt_weights=[0.55, 0.45],
        prompt_p=[0.0, 0.40],
        n_values=[16, 64, 256, 1024, 4096],
        mu1_base=1.1,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
        n_eval_prompts=3_000,
        seed=808,
    )

    return {
        "p0_01_theorem7b_n2_boundary": {
            "homoscedastic_monotone_analytic": analytic_mono_n2,
            "homoscedastic_raw_n2": d_mono["raw_bon_n2"],
            "homoscedastic_rank_ceiling_n2": d_mono["rank_only_n2_ceiling"],
            "homoscedastic_oracle_n2": d_mono["oracle_cal_n2"],
            "homoscedastic_strict_gap": d_mono["strict_gap_over_rank_n2"],
            "heteroscedastic_rank_ceiling_n2": d_het["rank_only_n2_ceiling"],
            "heteroscedastic_oracle_n2": d_het["oracle_cal_n2"],
            "heteroscedastic_strict_gap": d_het["strict_gap_over_rank_n2"],
        },
        "p0_02_spacings_beta025": spacings_rows,
        "p1_01_explicit_two_stratum_simpson": {
            "interval": [3.5, 4.2],
            "x_peak_stratum_1": 0.4,
            "x_peak_stratum_2": 3.4,
            "max_dq_stratum_1": float(np.max(st_simp["dq_g"][0])),
            "max_dq_stratum_2": float(np.max(st_simp["dq_g"][1])),
            "max_within_slope_term": float(np.max(diag_simp["within_slope_term"])),
            "min_w1_w2_product": float(np.min(st_simp["w_post"][0] * st_simp["w_post"][1])),
            "min_between_cov_term": float(np.min(diag_simp["between_cov_term"])),
            "min_q_pool_deriv": float(np.min(diag_simp["q_pool_deriv"])),
            "q_pool_start": float(diag_simp["q_pool"][0]),
            "q_pool_end": float(diag_simp["q_pool"][-1]),
        },
        "p1_02_kappa_lt1_zero_dmu": {
            "criterion": crit_k_lt1,
            "n_values": [1, 2, 3, 4, 8, 16],
            "exact_bon_curve": curve_k_lt1,
        },
        "p1_04_copula_eta_boundaries": {
            "k1_eta0": cop_k1["eta0_copula"],
            "k1_class": cop_k1["eta0_boundary_class"],
            "k3_rho099_eta0": cop_high["eta0_copula"],
            "k3_rho099_strictly_below_one": cop_high["eta_strictly_below_one"],
            "k3_rho099_class": cop_high["eta0_boundary_class"],
        },
        "p1_05_sample_zscore_extremal_convergence": sample_z_conv,
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
        n_bootstrap=2_000,
        n_permutations=10_000,
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
        n_bootstrap=2_000,
        n_permutations=10_000,
    )


def _recompute_s3_3domain_het() -> dict[str, Any]:
    parsed = {tag: rs2._stream_parse_strategic_ttc(tag, rel) for tag, rel in rs2.TTC_FILES.items()}
    pooled_3domain: list[tuple[str, np.ndarray, np.ndarray]] = []
    for dom_name, tag in (("GSM8K", "gsm8k_qwen7b"), ("GPQA", "gpqa_qwen7b"), ("AIME", "aime_qwen7b")):
        for x in parsed[tag]:
            pooled_3domain.append(
                (
                    dom_name,
                    np.asarray(x["rewards"], dtype=np.float64),
                    np.asarray(x["correct"], dtype=bool),
                )
            )
    return evaluate_real_prompt_heterogeneity(
        pooled_3domain,
        n_values=[1, 2, 4, 8, 16, 32, 64, 128],
        n_folds=5,
        n_trials_per_problem=20,
        n_bins=15,
        n_simpson_bins=12,
        seed=760,
        n_bootstrap=2_000,
        n_permutations=10_000,
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
            all_probs.append(
                {
                    "subset": sub_name,
                    "rm_a": np.asarray(rm_all[idx], dtype=np.float64),
                    "rm_b": np.asarray(rm_all[offset_b + idx], dtype=np.float64),
                    "is_correct": np.asarray(corr_all[idx], dtype=bool),
                }
            )
    n_values = [1, 2, 4, 8, 16, 32, 64]
    het_b = evaluate_real_prompt_heterogeneity(
        [(p["subset"], p["rm_b"], p["is_correct"]) for p in all_probs],
        n_values=n_values,
        n_folds=5,
        n_trials_per_problem=12,
        n_bins=15,
        n_simpson_bins=12,
        seed=830,
        n_bootstrap=2_000,
        n_permutations=10_000,
    )
    het_a = evaluate_real_prompt_heterogeneity(
        [(p["subset"], p["rm_a"], p["is_correct"]) for p in all_probs],
        n_values=n_values,
        n_folds=5,
        n_trials_per_problem=12,
        n_bins=15,
        n_simpson_bins=12,
        seed=831,
        n_bootstrap=2_000,
        n_permutations=10_000,
    )
    return het_b, het_a


def refresh_real_heterogeneity_v8() -> dict[str, Any]:
    """重算 5 组真实异质性基准并同步刷新 Stage 2 缓存、JSON/CSV 汇总及图 10。"""
    print("[V8] Recomputing S1a RB2 Skywork heterogeneity (3-branch hybrid + 2000x boot + 10000x perm)...")
    het_rb2 = rs2.to_json_safe(_recompute_rb2_het())
    print("[V8] Recomputing S2 OlympiadBench heterogeneity...")
    het_oly = rs2.to_json_safe(_recompute_olympiad_het())
    print("[V8] Recomputing S3 3-Domain heterogeneity...")
    het_3dom = rs2.to_json_safe(_recompute_s3_3domain_het())
    print("[V8] Recomputing S4 AceMath 7-strata heterogeneity (RM-B & RM-A)...")
    het_b_raw, het_a_raw = _recompute_acemath_het()
    het_b = rs2.to_json_safe(het_b_raw)
    het_a = rs2.to_json_safe(het_a_raw)

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
    math_checks = rs2.to_json_safe(verify_v8_math_checks())
    real_het = refresh_real_heterogeneity_v8()
    out = {
        "v8_math_checks": math_checks,
        "real_heterogeneity_v8": real_het,
    }
    out_path = PROJECT_ROOT / "results/v8_rigor_verification.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved:", out_path)

    print("[P0-01 Thm7B N=2]", math_checks["p0_01_theorem7b_n2_boundary"])
    print("[P0-02 Spacings beta=0.25]", math_checks["p0_02_spacings_beta025"])
    print("[P1-01 Explicit Simpson]", math_checks["p1_01_explicit_two_stratum_simpson"])
    print("[P1-05 Sample Z Conv]", math_checks["p1_05_sample_zscore_extremal_convergence"])

    for label, blk in real_het.items():
        nv = str(blk["n_values"][-1])
        algs = blk["algorithms"]
        vs_pool = blk["paired_bootstrap_vs_pooled"]
        vs_std = blk["paired_bootstrap_vs_standard_bon"]
        p_z = vs_pool["Within-Prompt-Z-KL-LCB"]
        p_h_pool = vs_pool["Hybrid-Adaptive-KL-LCB"]
        p_h_std = vs_std["Hybrid-Adaptive-KL-LCB"]
        counts = blk["hybrid_branch_selection_counts"][nv]
        print(
            f"[Real {label} N={nv} folds={counts}] "
            f"Std={algs['Standard-BoN']['accuracy'][-1]:.4f}+/-{algs['Standard-BoN']['se'][-1]:.4f} | "
            f"Pool={algs['Pooled-KL-LCB']['accuracy'][-1]:.4f}+/-{algs['Pooled-KL-LCB']['se'][-1]:.4f} | "
            f"Z={algs['Within-Prompt-Z-KL-LCB']['accuracy'][-1]:.4f}+/-{algs['Within-Prompt-Z-KL-LCB']['se'][-1]:.4f} "
            f"(vs Pool: {p_z['mean_diff'][-1]:+.4f}, p={p_z['p_value'][-1]:.5f}) | "
            f"Hyb={algs['Hybrid-Adaptive-KL-LCB']['accuracy'][-1]:.4f}+/-{algs['Hybrid-Adaptive-KL-LCB']['se'][-1]:.4f} "
            f"(vs Pool: {p_h_pool['mean_diff'][-1]:+.4f}, p={p_h_pool['p_value'][-1]:.5f}; "
            f"vs Std: {p_h_std['mean_diff'][-1]:+.4f}, p={p_h_std['p_value'][-1]:.5f})"
        )


if __name__ == "__main__":
    main()
