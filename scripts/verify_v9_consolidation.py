"""三审收敛定稿（V9 Consolidation）全套数学、分桶协议与独立留出测试集验证脚本。

执行内容（全离线，基于本地 data/ 缓存）：
1. 核验三审全部数学修复项：
   - P0-01 定理 7C 原始打分与基底打分仿射变换恒等式及大偏移（B=1000, a=8）不变性；
   - P1-01 命题 6.1 完整置信水平 delta = 1/(4M) 下的紧 Gamma 间距浓缩界、常数 2.0 与预算平衡点 N_balance；
   - P1-02 定理 7A 闭区间 [3.5, 4.2] 上的无网格纯解析上下界链条。
2. 真实数据实验 I（修复 P0-02）：在 AceMath RM-B（P_test=4,494, N=64）与 GPQA-Qwen7B（P_test=273, N=128）
   上严格按同一总预算 M 对比三种等频分桶协议：
   - 'sample_split'（命题 6.1 严格 M/2 分位边界 + M/2 估计，零额外无标签数据）
   - 'in_sample'（同批 M 条样本同时估计边界与桶均值，零额外无标签数据）
   - 'semi_supervised'（全池 U 条无标签分数边界 + M 条有标签样本）
   以及同预算 Fixed-60 + MLE 基线。
3. 真实数据实验 II（修复 P1-03 & P1-04）：在全部 5 组真实异质性基准上执行互不相交单次划分
   Train (50%) / Val (20%) / Held-Out Test (30%)，评估零重拟合误差的 Hybrid-Fixed-KL-LCB、
   重拟合漂移模量 epsilon_refit 及双侧配对学生化 t 检验主 p 值（同步刷新 5-Fold 辅助对照）。
4. 真实数据实验 III（同等预算与算力下的受控重错尾污染对照）：在 AceMath RM-B 真实候选池上注入
   6% 错误轨迹高分假阳性污染（匹配定理 2A(b) Reward Hacking 机制），在严格相同单次划分预算下对比
   Standard-BoN、Pooled-KL-LCB、Within-Prompt-Z-KL-LCB 与 Hybrid-Fixed-KL-LCB。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import norm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import plot_stage2_figures as ps2  # noqa: E402
import run_stage2_experiments as rs2  # noqa: E402

from tts_theory import (  # noqa: E402
    calibration_minimax_scaling,
    empirical_quantile_spacings_bounds,
    evaluate_finite_calibration_on_real_pools,
    evaluate_real_prompt_heterogeneity,
    evaluate_single_split_prompt_heterogeneity,
    tight_joint_minimax_scaling,
    verify_sample_zscore_extremal_convergence,
)


def verify_math_and_analytical_bounds() -> dict[str, Any]:
    """核验 P0-01、P1-01 与 P1-02 解析界。"""
    # 1. P0-01: 大偏移仿射不变性
    conv_zero = verify_sample_zscore_extremal_convergence(
        prompt_weights=[0.55, 0.45],
        prompt_p=[0.0, 0.40],
        n_values=[16, 64, 256, 1024, 4096],
        mu1_base=1.1,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
        prompt_shift=[0.0, 0.0],
        prompt_scale=[1.0, 1.0],
        n_eval_prompts=3_000,
        seed=808,
    )
    conv_affine = verify_sample_zscore_extremal_convergence(
        prompt_weights=[0.55, 0.45],
        prompt_p=[0.0, 0.40],
        n_values=[16, 64, 256, 1024, 4096],
        mu1_base=1.1,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
        prompt_shift=[-250.0, 1000.0],
        prompt_scale=[0.25, 6.0],
        n_eval_prompts=3_000,
        seed=808,
    )
    max_shift_diff = float(np.max(np.abs(conv_zero["sample_z_oracle_acc"] - conv_affine["sample_z_oracle_acc"])))

    # 2. P1-01: 完整命题置信度 delta = 1/(4M) 与预算平衡点 N_balance
    spacings_full_delta = []
    for m_val in (250_000, 1_000_000, 2_500_000):
        k_opt = int(round((float(m_val) / math.log(float(m_val))) ** (2.0 / 3.0)))
        delta_prop = 1.0 / float(4 * m_val)
        sp = empirical_quantile_spacings_bounds(m_total=m_val, n_bins=k_opt, delta=delta_prop)
        spacings_full_delta.append(
            {
                "m_total": m_val,
                "k_opt_beta025": k_opt,
                "delta_proposition": delta_prop,
                "d_min": sp["d_min"],
                "d_min_required": sp["d_min_required"],
                "spacings_failure_prob_upper": sp["spacings_failure_prob_upper"],
                "spacings_condition_holds": sp["spacings_condition_holds"],
                "lcb_penalty_inflation_factor": sp["lcb_penalty_inflation_factor"],
                "dkw_relative_error_bound": sp["dkw_relative_error_bound"],
            }
        )

    balance_table = []
    for m_val in (500, 5_000, 50_000, 500_000):
        tj = tight_joint_minimax_scaling(m_total=m_val, n_budget=64, beta=2.0)
        balance_table.append(
            {
                "m_total": m_val,
                "beta": 2.0,
                "n_balance_rate": tj["calibration_compute_balance_n"],
                "k_opt": tj["n_agnostic_k_opt"],
                "regime_at_n64": tj["bottleneck_regime"],
            }
        )

    # 3. P1-02: [3.5, 4.2] 无网格纯解析区间上下界
    def _ell1(s: float) -> float:
        return math.log(2.0) - 0.5 * ((s - 0.3) ** 2) + 0.125 * (s * s)

    def _ell2(s: float) -> float:
        return math.log(2.0) - 0.5 * ((s - 3.3) ** 2) + 0.125 * ((s - 3.0) ** 2)

    def _sig(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    q1_lo = _sig(math.log(0.05 / 0.95) + _ell1(4.2))
    q1_hi = _sig(math.log(0.05 / 0.95) + _ell1(3.5))
    q2_lo = _sig(math.log(0.95 / 0.05) + _ell2(4.2))
    q2_hi = _sig(math.log(0.95 / 0.05) + _ell2(3.5))
    within_abs_ub = max(q1_hi * 2.85, (1.0 - q2_lo) * 0.60)

    f1_lo = 0.05 * float(norm.pdf(4.2, 0.3, 1.0)) + 0.95 * float(norm.pdf(4.2, 0.0, 2.0))
    f1_hi = 0.05 * float(norm.pdf(3.5, 0.3, 1.0)) + 0.95 * float(norm.pdf(3.5, 0.0, 2.0))
    f2_lo = 0.95 * float(norm.pdf(4.2, 3.3, 1.0)) + 0.05 * float(norm.pdf(4.2, 3.0, 2.0))
    f2_hi = 0.95 * float(norm.pdf(3.5, 3.3, 1.0)) + 0.05 * float(norm.pdf(3.5, 3.0, 2.0))
    r_lo = (0.15 * f2_lo) / (0.85 * f1_hi)
    r_hi = (0.15 * f2_hi) / (0.85 * f1_lo)
    w_prod_lb = min(r_lo / ((1.0 + r_lo) ** 2), r_hi / ((1.0 + r_hi) ** 2))
    g_diff_lb = 3.3 - 0.75 * 4.2
    cov_lb = w_prod_lb * (q2_lo - q1_hi) * g_diff_lb

    return {
        "p0_01_affine_invariance": {
            "max_affine_posterior_discrepancy": conv_affine["max_affine_posterior_discrepancy"],
            "max_sample_z_acc_diff_under_extreme_affine": max_shift_diff,
            "sample_z_oracle_acc_n4096": float(conv_affine["sample_z_oracle_acc"][-1]),
            "theoretical_z_oracle_ceiling": conv_affine["theoretical_z_oracle_ceiling"],
            "theoretical_raw_ceiling": conv_affine["theoretical_raw_ceiling"],
        },
        "p1_01_spacings_full_delta_1_over_4m": spacings_full_delta,
        "calibration_compute_balance_examples": balance_table,
        "p1_02_analytical_simpson_bounds": {
            "interval": [3.5, 4.2],
            "q1_exact_endpoint_range": [q1_lo, q1_hi],
            "q2_exact_endpoint_range": [q2_lo, q2_hi],
            "analytical_within_abs_upper_bound": within_abs_ub,
            "analytical_w1_w2_lower_bound": w_prod_lb,
            "analytical_g2_minus_g1_lower_bound": g_diff_lb,
            "analytical_cov_lower_bound": cov_lb,
            "analytical_q_pool_deriv_lower_bound": cov_lb - within_abs_ub,
        },
    }


def run_three_quantile_protocols_real_comparison() -> dict[str, Any]:
    """修复 P0-02：在真实 AceMath RM-B 与 GPQA-Qwen7B 上同场对比三种等频分桶协议。"""
    # 1. AceMath RM-B (50/50 题级划分与 Stage 2 完全一致)
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
                    "rm_b": np.asarray(rm_all[offset_b + idx], dtype=np.float64),
                    "rm_a": np.asarray(rm_all[idx], dtype=np.float64),
                    "is_correct": np.asarray(corr_all[idx], dtype=bool),
                }
            )
    rng = np.random.default_rng(2026)
    perm = rng.permutation(len(all_probs))
    half = len(all_probs) // 2
    cal_p = [all_probs[i] for i in perm[:half]]
    test_p = [all_probs[i] for i in perm[half:]]
    ace_cal_s = np.concatenate([p["rm_b"] for p in cal_p])
    ace_cal_y = np.concatenate([p["is_correct"] for p in cal_p])
    ace_test = [(p["rm_b"], p["is_correct"]) for p in test_p]

    # 2. GPQA-Qwen7B (50/50 题级划分与 Stage 2 完全一致)
    gpqa_items = rs2._stream_parse_strategic_ttc("gpqa_qwen7b", rs2.TTC_FILES["gpqa_qwen7b"])
    rng_g = np.random.default_rng(750)
    perm_g = rng_g.permutation(len(gpqa_items))
    half_g = len(gpqa_items) // 2
    cal_g = [gpqa_items[i] for i in perm_g[:half_g]]
    test_g = [gpqa_items[i] for i in perm_g[half_g:]]
    gpqa_cal_s = np.concatenate([np.asarray(x["rewards"], dtype=np.float64) for x in cal_g])
    gpqa_cal_y = np.concatenate([np.asarray(x["correct"], dtype=bool) for x in cal_g])
    gpqa_test = [(np.asarray(x["rewards"], dtype=np.float64), np.asarray(x["correct"], dtype=bool)) for x in test_g]

    def _eval_dataset_protocols(
        c_s: np.ndarray,
        c_y: np.ndarray,
        t_pools: list[tuple[np.ndarray, np.ndarray]],
        m_list: list[int],
        n_eval: int,
        n_rep: int,
        base_seed: int,
    ) -> dict[str, Any]:
        rows = []
        raw_ref = None
        for i_m, mv in enumerate(m_list):
            if mv > c_s.size:
                continue
            k_1_5 = max(3, int(calibration_minimax_scaling(mv, delta=0.05, beta=2.0)["k_opt_theory"]))
            # 在样本分割协议下估计集大小为 M/2，同时评估 K*(M) 与适配 M/2 的 K*(M/2)
            k_split = max(3, int(calibration_minimax_scaling(max(10, mv // 2), delta=0.05, beta=2.0)["k_opt_theory"]))

            r_split = evaluate_finite_calibration_on_real_pools(
                c_s,
                c_y,
                t_pools,
                m_cal=mv,
                n_bins=k_split,
                n_values=[n_eval],
                binning="quantile",
                methods=("kl_lcb",),
                n_repeats=n_rep,
                delta=0.05,
                seed=base_seed + i_m,
                quantile_protocol="sample_split",
            )
            r_in = evaluate_finite_calibration_on_real_pools(
                c_s,
                c_y,
                t_pools,
                m_cal=mv,
                n_bins=k_1_5,
                n_values=[n_eval],
                binning="quantile",
                methods=("kl_lcb",),
                n_repeats=n_rep,
                delta=0.05,
                seed=base_seed + i_m,
                quantile_protocol="in_sample",
            )
            r_semi = evaluate_finite_calibration_on_real_pools(
                c_s,
                c_y,
                t_pools,
                m_cal=mv,
                n_bins=k_1_5,
                n_values=[n_eval],
                binning="quantile",
                methods=("kl_lcb",),
                n_repeats=n_rep,
                delta=0.05,
                seed=base_seed + i_m,
                quantile_protocol="semi_supervised",
            )
            r_fix60 = evaluate_finite_calibration_on_real_pools(
                c_s,
                c_y,
                t_pools,
                m_cal=mv,
                n_bins=60,
                n_values=[n_eval],
                binning="quantile",
                methods=("mle",),
                n_repeats=n_rep,
                delta=0.05,
                seed=base_seed + i_m,
                quantile_protocol="sample_split",
            )
            if raw_ref is None:
                raw_ref = float(r_semi["standard_bon_accuracy"][0])
            rows.append(
                {
                    "m_total": mv,
                    "k_opt_full": k_1_5,
                    "k_opt_split": k_split,
                    "sample_split_kl_lcb_acc": float(r_split["methods"]["kl_lcb"]["accuracy"][0]),
                    "sample_split_kl_lcb_se": float(r_split["methods"]["kl_lcb"]["se"][0]),
                    "in_sample_kl_lcb_acc": float(r_in["methods"]["kl_lcb"]["accuracy"][0]),
                    "in_sample_kl_lcb_se": float(r_in["methods"]["kl_lcb"]["se"][0]),
                    "semi_supervised_kl_lcb_acc": float(r_semi["methods"]["kl_lcb"]["accuracy"][0]),
                    "semi_supervised_kl_lcb_se": float(r_semi["methods"]["kl_lcb"]["se"][0]),
                    "sample_split_fixed60_mle_acc": float(r_fix60["methods"]["mle"]["accuracy"][0]),
                    "sample_split_fixed60_mle_se": float(r_fix60["methods"]["mle"]["se"][0]),
                }
            )
        return {
            "n_budget": n_eval,
            "n_test_problems": len(t_pools),
            "standard_bon_acc": raw_ref,
            "rows": rows,
        }

    print("[V9] Evaluating 3 quantile protocols on AceMath RM-B (N=64) & GPQA-Qwen7B (N=128)...")
    ace_proto = _eval_dataset_protocols(
        ace_cal_s, ace_cal_y, ace_test, [100, 200, 400, 800, 2000, 5000, 15000, 40000], 64, 50, 1100
    )
    gpqa_proto = _eval_dataset_protocols(
        gpqa_cal_s, gpqa_cal_y, gpqa_test, [100, 200, 400, 800, 2000, 5000, 12000], 128, 50, 1200
    )
    return {
        "acemath_rm_b_protocols": ace_proto,
        "gpqa_qwen7b_protocols": gpqa_proto,
        "_acemath_all_probs": all_probs,
    }


def run_disjoint_single_split_and_stress_experiments(all_probs_ace: list[dict[str, Any]]) -> dict[str, Any]:
    """执行五组真实基准的单次互不相交划分（50/20/30）主检验及同预算重错尾受控对照实验。"""
    # 1. RB2 Skywork
    rb2_root = rs2.STAGE2_DIR / "reward_bench_2"
    loaded = {name: rs2._load_rb2_model(rb2_root / rel) for name, rel in rs2.RB2_HETERO6.items()}
    _, h6_mats, h6_ys, h6_strata = rs2._align_rb2_group(loaded)
    rb2_pools = [(st, mat[:, 0], y) for st, mat, y in zip(h6_strata, h6_mats, h6_ys)]

    # 2. OlympiadBench uPRM-70B
    oly_root = rs2.STAGE2_DIR / "olympiadbench"
    repo_70b = "sibasmarakp/Qwen2.5-7B-Instruct-uPRM-70B-T80-olympiadbench-best_of_n-completions"
    repo_32b = "sibasmarakp/Qwen2.5-7B-Instruct-uPRM-32B-T80-olympiadbench-best_of_n-completions"
    p70, _ = rs2._parse_olympiad_repo(repo_70b, oly_root / "uprm_70b")
    p32, _ = rs2._parse_olympiad_repo(repo_32b, oly_root / "uprm_32b")
    map70 = {p["id"]: p for p in p70}
    map32 = {p["id"]: p for p in p32}
    common_ids = sorted(set(map70.keys()) & set(map32.keys()))
    oly_pools = [(map70[pid]["subfield"], map70[pid]["prm_last"], map70[pid]["is_correct"]) for pid in common_ids]

    # 3. S3 3-Domain
    parsed_ttc = {tag: rs2._stream_parse_strategic_ttc(tag, rel) for tag, rel in rs2.TTC_FILES.items()}
    s3_pools: list[tuple[str, np.ndarray, np.ndarray]] = []
    for dom_name, tag in (("GSM8K", "gsm8k_qwen7b"), ("GPQA", "gpqa_qwen7b"), ("AIME", "aime_qwen7b")):
        for x in parsed_ttc[tag]:
            s3_pools.append((dom_name, np.asarray(x["rewards"], dtype=np.float64), np.asarray(x["correct"], dtype=bool)))

    # 4. AceMath RM-B & RM-A
    ace_b_pools = [(p["subset"], p["rm_b"], p["is_correct"]) for p in all_probs_ace]
    ace_a_pools = [(p["subset"], p["rm_a"], p["is_correct"]) for p in all_probs_ace]

    print("[V9] Running disjoint single-split (50/20/30) paired t-test primary inference on 5 real benchmarks...")
    ss_rb2 = evaluate_single_split_prompt_heterogeneity(
        rb2_pools, n_values=[1, 2, 3, 4], n_trials_per_problem=20, n_bins=15, seed=107, n_bootstrap=2_000
    )
    ss_oly = evaluate_single_split_prompt_heterogeneity(
        oly_pools, n_values=[1, 2, 4, 8, 16, 32, 64, 128], n_trials_per_problem=20, n_bins=15, seed=613, n_bootstrap=2_000
    )
    ss_s3 = evaluate_single_split_prompt_heterogeneity(
        s3_pools, n_values=[1, 2, 4, 8, 16, 32, 64, 128], n_trials_per_problem=20, n_bins=15, seed=760, n_bootstrap=2_000
    )
    ss_ace_b = evaluate_single_split_prompt_heterogeneity(
        ace_b_pools, n_values=[1, 2, 4, 8, 16, 32, 64], n_trials_per_problem=12, n_bins=15, seed=830, n_bootstrap=2_000
    )
    ss_ace_a = evaluate_single_split_prompt_heterogeneity(
        ace_a_pools, n_values=[1, 2, 4, 8, 16, 32, 64], n_trials_per_problem=12, n_bins=15, seed=831, n_bootstrap=2_000
    )

    # 5. 同等标签预算与计算预算下的受控 Reward Hacking（定理 2A(b)）对照实验（基于真实 AceMath RM-B 候选池）
    print("[V9] Running controlled equal-budget reward-hacking stress test on real AceMath RM-B pools...")
    hack_rng = np.random.default_rng(202609)
    all_rm_b = np.concatenate([p["rm_b"] for p in all_probs_ace])
    q995 = float(np.quantile(all_rm_b, 0.995))
    sd_b = float(np.std(all_rm_b))
    ace_hacked_pools: list[tuple[str, np.ndarray, np.ndarray]] = []
    for p in all_probs_ace:
        s_clean = p["rm_b"].copy()
        y_clean = p["is_correct"].copy()
        wrong_mask = ~y_clean
        if np.any(wrong_mask):
            spike_draw = (hack_rng.random(s_clean.size) < 0.06) & wrong_mask
            if np.any(spike_draw):
                s_clean[spike_draw] = q995 + 0.8 * sd_b + np.abs(hack_rng.normal(0.0, 0.5 * sd_b, size=int(np.sum(spike_draw))))
        ace_hacked_pools.append((p["subset"], s_clean, y_clean))

    ss_ace_hacked = evaluate_single_split_prompt_heterogeneity(
        ace_hacked_pools,
        n_values=[1, 2, 4, 8, 16, 32, 64],
        n_trials_per_problem=12,
        n_bins=20,
        seed=835,
        n_bootstrap=2_000,
    )

    # 6. 同步刷新 5-Fold 辅助统计（现已包含 N=2 全三分支竞争）
    print("[V9] Refreshing 5-fold CV auxiliary tables (with uniform N>=2 3-branch ERM)...")
    cv_rb2 = rs2.to_json_safe(
        evaluate_real_prompt_heterogeneity(rb2_pools, n_values=[1, 2, 3, 4], n_trials_per_problem=20, n_bins=15, seed=107)
    )
    cv_oly = rs2.to_json_safe(
        evaluate_real_prompt_heterogeneity(oly_pools, n_values=[1, 2, 4, 8, 16, 32, 64, 128], n_trials_per_problem=20, n_bins=15, seed=613)
    )
    cv_s3 = rs2.to_json_safe(
        evaluate_real_prompt_heterogeneity(s3_pools, n_values=[1, 2, 4, 8, 16, 32, 64, 128], n_trials_per_problem=20, n_bins=15, n_simpson_bins=12, seed=760)
    )
    cv_ace_b = rs2.to_json_safe(
        evaluate_real_prompt_heterogeneity(ace_b_pools, n_values=[1, 2, 4, 8, 16, 32, 64], n_trials_per_problem=12, n_bins=15, n_simpson_bins=12, seed=830)
    )
    cv_ace_a = rs2.to_json_safe(
        evaluate_real_prompt_heterogeneity(ace_a_pools, n_values=[1, 2, 4, 8, 16, 32, 64], n_trials_per_problem=12, n_bins=15, n_simpson_bins=12, seed=831)
    )

    s2_json_path = rs2.RESULTS_DIR / "stage2_real_results.json"
    if s2_json_path.exists():
        payload = json.loads(s2_json_path.read_text(encoding="utf-8"))
        payload["s1a_rb2"]["rb2_skywork_heterogeneity"] = cv_rb2
        payload["s2_olympiadbench"]["olympiad_subfield_heterogeneity"] = cv_oly
        payload["s3_strategic_ttc"]["cross_3domain_simpson_het"] = cv_s3
        payload["s4_acemath"]["het_7strata_rm_b"] = cv_ace_b
        payload["s4_acemath"]["het_7strata_rm_a"] = cv_ace_a
        s2_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        rs2.export_summary_csv(payload, rs2.RESULTS_DIR / "stage2_summary.csv")
        ps2.plot_fig10_simpson_and_cross_domain(
            payload, rs2.RESULTS_DIR / "figures/stage2_real_simpson_and_cross_domain.png"
        )

    return {
        "disjoint_single_split_primary": {
            "rb2_skywork": ss_rb2,
            "olympiadbench_uprm70b": ss_oly,
            "s3_3domain": ss_s3,
            "acemath_rm_b": ss_ace_b,
            "acemath_rm_a": ss_ace_a,
        },
        "equal_budget_reward_hacking_stress_acemath": ss_ace_hacked,
    }


def main() -> None:
    math_blk = rs2.to_json_safe(verify_math_and_analytical_bounds())
    proto_raw = run_three_quantile_protocols_real_comparison()
    all_probs_ace = proto_raw.pop("_acemath_all_probs")
    proto_blk = rs2.to_json_safe(proto_raw)
    split_blk = rs2.to_json_safe(run_disjoint_single_split_and_stress_experiments(all_probs_ace))

    out = {
        "math_and_analytical_bounds": math_blk,
        "quantile_protocols_real_comparison": proto_blk,
        **split_blk,
    }
    out_path = PROJECT_ROOT / "results/v9_consolidation_verification.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved:", out_path)

    print("[P0-01 Affine Invariance]", math_blk["p0_01_affine_invariance"])
    print("[P1-01 Full delta=1/(4M) Spacings]", math_blk["p1_01_spacings_full_delta_1_over_4m"])
    print("[P1-02 Analytical Simpson]", math_blk["p1_02_analytical_simpson_bounds"])

    for p_key in ("acemath_rm_b_protocols", "gpqa_qwen7b_protocols"):
        b = proto_blk[p_key]
        print(f"=== {p_key} (N={b['n_budget']}, Raw={b['standard_bon_acc']:.4f}) ===")
        for r in b["rows"]:
            print(
                f"  M={r['m_total']:5d} | Split(K={r['k_opt_split']:2d})={r['sample_split_kl_lcb_acc']:.4f} | "
                f"InSample(K={r['k_opt_full']:2d})={r['in_sample_kl_lcb_acc']:.4f} | "
                f"SemiSup(K={r['k_opt_full']:2d})={r['semi_supervised_kl_lcb_acc']:.4f} | "
                f"SplitFix60MLE={r['sample_split_fixed60_mle_acc']:.4f}"
            )

    for label, ss in split_blk["disjoint_single_split_primary"].items():
        nv = str(ss["n_values"][-1])
        algs = ss["algorithms"]
        vp = ss["paired_test_vs_pooled"]["Hybrid-Fixed-KL-LCB"]
        vs = ss["paired_test_vs_standard_bon"]["Hybrid-Fixed-KL-LCB"]
        print(
            f"[SingleSplit {label} P_test={ss['n_test_problems']} N={nv} sel={ss['selected_branch_by_n'][nv]} "
            f"eps_refit={ss['empirical_refit_shift_by_n'][-1]:.4f}] "
            f"Std={algs['Standard-BoN']['accuracy'][-1]:.4f} | "
            f"Pool={algs['Pooled-KL-LCB']['accuracy'][-1]:.4f} | "
            f"Z={algs['Within-Prompt-Z-KL-LCB']['accuracy'][-1]:.4f} | "
            f"HybFixed={algs['Hybrid-Fixed-KL-LCB']['accuracy'][-1]:.4f} "
            f"(vs Pool: {vp['mean_diff'][-1]:+.4f}, p_t={vp['p_value_ttest'][-1]:.4g}; "
            f"vs Std: {vs['mean_diff'][-1]:+.4f}, p_t={vs['p_value_ttest'][-1]:.4g})"
        )

    hk = split_blk["equal_budget_reward_hacking_stress_acemath"]
    hk_a = hk["algorithms"]
    hk_vs = hk["paired_test_vs_standard_bon"]["Hybrid-Fixed-KL-LCB"]
    print(
        f"[Stress AceMath 6% False-Positive Spike N=64 sel={hk['selected_branch_by_n']['64']}] "
        f"Std={hk_a['Standard-BoN']['accuracy'][-1]:.4f} | "
        f"Pool={hk_a['Pooled-KL-LCB']['accuracy'][-1]:.4f} | "
        f"Z={hk_a['Within-Prompt-Z-KL-LCB']['accuracy'][-1]:.4f} | "
        f"HybFixed={hk_a['Hybrid-Fixed-KL-LCB']['accuracy'][-1]:.4f} "
        f"(vs Std: {hk_vs['mean_diff'][-1]:+.4f}, 95%CI=[{hk_vs['ci_low'][-1]:+.4f},{hk_vs['ci_high'][-1]:+.4f}], p_t={hk_vs['p_value_ttest'][-1]:.4g})"
    )


if __name__ == "__main__":
    main()
