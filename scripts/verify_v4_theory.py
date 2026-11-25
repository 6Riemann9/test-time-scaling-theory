"""V4 严格理论定量交叉验证脚本：定理 5（异构与 Copula）、定理 6（Fano 1/5、KL-LCB、N^{-2}）、定理 7（辛普森与题内标准化）。

运行方式：
    python -u scripts/verify_v4_theory.py
输出：
    results/v4_theory_verification.json
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tts_theory.calibration_theory import (
    bernstein_zero_collapse_threshold,
    calibration_minimax_scaling,
    joint_calibration_compute_scaling,
    simulate_finite_calibration_scaling,
)
from tts_theory.ensemble_theory import (
    GaussianCopulaHeavyTailSpec,
    GeneralGaussianMultiVerifierSpec,
    gaussian_copula_heavy_tail_criterion,
    general_min_ensemble_qp_criterion,
    optimal_linear_ensemble_criterion,
    simulate_multi_verifier_bon,
    solve_positive_quadrant_qp,
)
from tts_theory.heterogeneity_theory import (
    PromptHeterogeneitySpec,
    diagnose_simpson_confounding,
    evaluate_pooled_vs_within_prompt_calibration,
)
from tts_theory.integrals import exact_bon_curve, exact_calibrated_bon_accuracy
from tts_theory.theorems import (
    GaussianVerifierSpec,
    calibrated_bon_asymptotic_limit,
    gaussian_peak_threshold,
)


def _to_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_to_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return None if (np.isnan(val) or np.isinf(val)) else val
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def main() -> None:
    t0 = time.perf_counter()
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. 定理 5A：一般非对称线性负权逆转 + 一般 Active-Set Min 胜出 Mean
    cov1_neg = np.eye(2, dtype=np.float64)
    cov0_neg = (1.2**2) * np.array([[1.0, 0.85], [0.85, 1.0]], dtype=np.float64)
    spec_neg = GeneralGaussianMultiVerifierSpec(
        p=0.25, mu1=np.array([1.6, 0.2]), cov1=cov1_neg, mu0=np.array([0.0, 0.0]), cov0=cov0_neg
    )
    lin_crit = optimal_linear_ensemble_criterion(spec_neg)
    n_t5 = [1, 4, 16, 64, 256, 2000]
    t5_lin_mean = exact_bon_curve(spec_neg.mean_verifier_spec(), n_t5)
    t5_lin_opt = exact_bon_curve(spec_neg.linear_verifier_spec(lin_crit["w_star"]), n_t5)
    t5_lin_mc = simulate_multi_verifier_bon(
        spec_neg, n_t5, aggregation="linear", weights=lin_crit["w_star"], n_trials=40_000, seed=101
    )

    cov1_qp = np.array([[1.0, 0.8, 0.8], [0.8, 1.0, 0.8], [0.8, 0.8, 1.0]], dtype=np.float64)
    cov0_qp = np.array([[0.4, 0.7, 0.7], [0.7, 4.6, 0.2], [0.7, 0.2, 4.6]], dtype=np.float64)
    spec_qp = GeneralGaussianMultiVerifierSpec(
        p=0.25, mu1=np.array([1.0, 1.0, 0.6]), cov1=cov1_qp, mu0=np.zeros(3), cov0=cov0_qp
    )
    qp_crit = general_min_ensemble_qp_criterion(spec_qp)
    mc_qp_mean = simulate_multi_verifier_bon(spec_qp, n_t5, aggregation="mean", n_trials=40_000, seed=102)
    mc_qp_min = simulate_multi_verifier_bon(spec_qp, n_t5, aggregation="min", n_trials=40_000, seed=103)

    # 2. 定理 5B：Gaussian-Copula + Student-t 统一剩余尾相依
    r1_cop = np.full((4, 4), 0.85, dtype=np.float64)
    np.fill_diagonal(r1_cop, 1.0)
    r0_cop = np.eye(4, dtype=np.float64)
    spec_cop_ok = GaussianCopulaHeavyTailSpec(
        p=0.25, alpha1=4.5, alpha0=2.0, corr1=r1_cop, corr0=r0_cop, epsilon_shared=0.0
    )
    spec_cop_brk = GaussianCopulaHeavyTailSpec(
        p=0.25, alpha1=4.5, alpha0=2.0, corr1=r1_cop, corr0=r0_cop, epsilon_shared=0.06
    )
    cop_crit_ok = gaussian_copula_heavy_tail_criterion(spec_cop_ok)
    cop_crit_brk = gaussian_copula_heavy_tail_criterion(spec_cop_brk)
    mc_cop_mean = simulate_multi_verifier_bon(spec_cop_ok, n_t5, aggregation="mean", n_trials=30_000, seed=104)
    mc_cop_min = simulate_multi_verifier_bon(spec_cop_ok, n_t5, aggregation="min", n_trials=30_000, seed=105)
    mc_cop_brk = simulate_multi_verifier_bon(spec_cop_brk, n_t5, aggregation="min", n_trials=30_000, seed=106)

    # 3. 定理 6A/6B/6C：1/5 标度律、KL-LCB vs Bernstein、与 N^{-2} 精确收敛检验
    spec_cal = GaussianVerifierSpec(p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.5)
    q_star = calibrated_bon_asymptotic_limit(spec_cal)
    x_peak = gaussian_peak_threshold(spec_cal)
    f_star = float(spec_cal.mix_pdf(x_peak))
    inv_tau2 = 1.0 / (spec_cal.sigma1**2) - 1.0 / (spec_cal.sigma0**2)
    c_quad = 0.5 * q_star * (1.0 - q_star) * inv_tau2
    theory_n2_prefactor = c_quad / (2.0 * (f_star**2))

    n_rate_list = [200, 500, 1_000, 2_500, 5_000, 10_000, 20_000]
    rate_records = []
    for nv in n_rate_list:
        acc_val = exact_calibrated_bon_accuracy(spec_cal, nv, n_grid=16_384)
        reg = q_star - acc_val
        pred = theory_n2_prefactor * (float(nv) ** (-2.0))
        rate_records.append({
            "n": nv,
            "regret": reg,
            "theory_n2": pred,
            "ratio_regret_to_theory": reg / pred,
        })

    m_grid = [500, 5_000, 50_000, 500_000, 5_000_000]
    k_sweep_records = []
    m_base = 500
    base_k = 12
    for idx, mv in enumerate(m_grid):
        sc = (mv / math.log(mv)) / (m_base / math.log(m_base))
        k_15 = max(4, int(round(base_k * (sc**0.2))))
        k_25 = max(4, int(round(base_k * (sc**0.4))))
        r1 = simulate_finite_calibration_scaling(
            spec_cal, mv, k_15, [50_000], binning="quantile", methods=("kl_lcb",), n_repeats=500, seed=200 + idx
        )
        r2 = simulate_finite_calibration_scaling(
            spec_cal, mv, k_25, [50_000], binning="quantile", methods=("kl_lcb",), n_repeats=500, seed=300 + idx
        )
        k_sweep_records.append({
            "m_cal": mv,
            "k_one_fifth": k_15,
            "k_two_fifth_wrong": k_25,
            "regret_one_fifth": q_star - float(r1["methods"]["kl_lcb"]["accuracy"][0]),
            "regret_two_fifth_wrong": q_star - float(r2["methods"]["kl_lcb"]["accuracy"][0]),
        })

    x_log = np.log([r["m_cal"] / math.log(r["m_cal"]) for r in k_sweep_records])
    slope_15 = float(np.polyfit(x_log, np.log([r["regret_one_fifth"] for r in k_sweep_records]), 1)[0])
    slope_25 = float(np.polyfit(x_log, np.log([r["regret_two_fifth_wrong"] for r in k_sweep_records]), 1)[0])

    # M=400, K=40 上比较 MLE, KL-LCB, Wilson-LCB, Empirical-Bernstein LCB
    bern_thresh_k40 = bernstein_zero_collapse_threshold(40, q_star, delta=0.05)
    sim_lcb_cmp = simulate_finite_calibration_scaling(
        spec_cal, m_cal=400, n_bins=40, n_values=[1, 16, 64, 512, 5000], binning="quantile",
        methods=("mle", "kl_lcb", "wilson_lcb", "empirical_bernstein_lcb"),
        n_repeats=500, delta=0.05, seed=350
    )

    # 4. 定理 7：辛普森假单调遮蔽 + 题内 Z-标准化双向交叉
    spec_simpson = PromptHeterogeneitySpec(
        prompt_weights=[0.40, 0.45, 0.15],
        prompt_p=[0.15, 0.48, 0.92],
        prompt_shift=[-2.0, 0.2, 3.5],
        mu1_base=0.7, sigma1_base=1.0, mu0_base=0.0, sigma0_base=1.35,
    )
    simp_diag = diagnose_simpson_confounding(spec_simpson)
    spec_strong = PromptHeterogeneitySpec(
        prompt_weights=[0.35, 0.35, 0.30],
        prompt_p=[0.20, 0.35, 0.75],
        prompt_shift=[-4.5, 0.0, 4.5],
        mu1_base=1.0, sigma1_base=1.0, mu0_base=0.0, sigma0_base=1.65,
    )
    crossover_strong = evaluate_pooled_vs_within_prompt_calibration(
        spec_strong, [1, 2, 8, 32, 64, 256], m_cal_prompts=600, n_cal_per_prompt=32, n_eval_prompts=3_000, n_bins=12, seed=400
    )
    spec_noshift = PromptHeterogeneitySpec(
        prompt_weights=[1.0], prompt_p=[0.30], prompt_shift=[0.0],
        mu1_base=0.4, sigma1_base=1.0, mu0_base=0.0, sigma0_base=2.2,
    )
    crossover_noshift = evaluate_pooled_vs_within_prompt_calibration(
        spec_noshift, [1, 2, 4, 16, 64], m_cal_prompts=500, n_cal_per_prompt=20, n_eval_prompts=4_000, n_bins=14, seed=401
    )

    elapsed = time.perf_counter() - t0
    report = {
        "metadata": {"elapsed_seconds": elapsed},
        "theorem5a_negative_weight": {
            "criterion": lin_crit,
            "n_values": n_t5,
            "exact_mean": t5_lin_mean,
            "exact_linear_opt": t5_lin_opt,
            "mc_linear_opt": t5_lin_mc["accuracy"],
        },
        "theorem5a_general_qp_min": {
            "criterion": qp_crit,
            "n_values": n_t5,
            "mc_mean": mc_qp_mean["accuracy"],
            "mc_min": mc_qp_min["accuracy"],
        },
        "theorem5b_gaussian_copula": {
            "criterion_reversible": cop_crit_ok,
            "criterion_shared_breakdown": cop_crit_brk,
            "n_values": n_t5,
            "mc_mean": mc_cop_mean["accuracy"],
            "mc_min_reversible": mc_cop_min["accuracy"],
            "mc_min_shared_breakdown": mc_cop_brk["accuracy"],
        },
        "theorem6_minimax_and_kl_lcb": {
            "q_star": q_star,
            "theory_n2_prefactor": theory_n2_prefactor,
            "oracle_n_minus_2_rate": rate_records,
            "k_exponent_sweep": k_sweep_records,
            "slope_one_fifth": slope_15,
            "slope_two_fifth_wrong": slope_25,
            "bernstein_zero_threshold_k40": bern_thresh_k40,
            "lcb_comparison_m400_k40": sim_lcb_cmp,
            "joint_scaling_examples": {
                "small_n": joint_calibration_compute_scaling(50_000, 10, beta=2.0),
                "large_n": joint_calibration_compute_scaling(500, 5_000, beta=2.0),
            },
        },
        "theorem7_simpson_and_within_prompt": {
            "decomposition_max_abs_err": simp_diag["decomposition_max_abs_err"],
            "within_prompt_collapses": simp_diag["within_prompt_collapses"],
            "pooled_appears_monotone": simp_diag["pooled_appears_monotone"],
            "false_monotonicity_masking": simp_diag["false_monotonicity_masking"],
            "masked_point_count": simp_diag["masked_point_count"],
            "strong_shift_crossover": crossover_strong,
            "no_shift_crossover": crossover_noshift,
        },
    }

    out_path = results_dir / "v4_theory_verification.json"
    out_path.write_text(json.dumps(_to_json_safe(report), indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 80)
    print(f"V4 理论定量交叉验证全部完成！保存至: {out_path} (耗时 {elapsed:.2f}s)")
    print(
        f"[5A 负权差值逆转] kappa_mean={lin_crit['kappa_mean']:.3f}, "
        f"kappa_lin={lin_crit['kappa_linear_min']:.3f} | N=2000: Mean={t5_lin_mean[-1]:.4f} vs OptLin={t5_lin_opt[-1]:.4f}"
    )
    print(
        f"[5A 异构高斯 QP Min] active_set0={qp_crit['active_set0']}, gamma0={qp_crit['gamma0']:.3f} > gamma1={qp_crit['gamma1']:.3f} | "
        f"N=2000: Mean={mc_qp_mean['accuracy'][-1]:.4f} vs Min={mc_qp_min['accuracy'][-1]:.4f}"
    )
    print(
        f"[5B Copula 统一重尾] eff_alpha1={cop_crit_ok['effective_alpha1']:.3f}, eff_alpha0={cop_crit_ok['effective_alpha0_min']:.3f} | "
        f"N=2000: Mean={mc_cop_mean['accuracy'][-1]:.4f}, Min={mc_cop_min['accuracy'][-1]:.4f}, SharedBrk={mc_cop_brk['accuracy'][-1]:.4f}"
    )
    print(
        f"[6C 完美校准 N^-2 常数检验] N={n_rate_list[-1]:,} 比值 (实测/闭式) = {rate_records[-1]['ratio_regret_to_theory']:.4f} (理论极限 1.0000)"
    )
    print(
        f"[6A 分桶指数] slope(1/5)={slope_15:.4f} vs slope(错2/5)={slope_25:.4f} | "
        f"M=5M遗憾: 1/5={k_sweep_records[-1]['regret_one_fifth']:.5f} < 2/5={k_sweep_records[-1]['regret_two_fifth_wrong']:.5f}"
    )
    lcb_m = sim_lcb_cmp["methods"]
    print(
        f"[6B KL-LCB vs Bernstein (M=400, K=40, N=5000)] MLE={lcb_m['mle']['accuracy'][-1]:.4f} | "
        f"KL-LCB={lcb_m['kl_lcb']['accuracy'][-1]:.4f} | Wilson={lcb_m['wilson_lcb']['accuracy'][-1]:.4f} | "
        f"Bernstein={lcb_m['empirical_bernstein_lcb']['accuracy'][-1]:.4f} (归零阈值 m_k={bern_thresh_k40:.1f})"
    )
    print(
        f"[7A 辛普森遮蔽] err={simp_diag['decomposition_max_abs_err']:.2e} | masking={simp_diag['false_monotonicity_masking']} ({simp_diag['masked_point_count']} pts)"
    )
    print(
        f"[7B 强偏移 N=256] Raw={crossover_strong['raw_bon_mc'][-1]:.4f}, Pooled-KL={crossover_strong['pooled_kl_lcb_mc'][-1]:.4f}, Within-Z-KL={crossover_strong['within_prompt_z_kl_lcb_mc'][-1]:.4f}"
    )
    print(
        f"[7B 无偏移 N=2  ] Pooled-KL={crossover_noshift['pooled_kl_lcb_mc'][1]:.4f} vs Within-Z-KL={crossover_noshift['within_prompt_z_kl_lcb_mc'][1]:.4f}"
    )


if __name__ == "__main__":
    main()
