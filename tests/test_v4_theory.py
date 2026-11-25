"""V4 严格理论重构单元测试：检验定理 5–7 的全部推广、修补与定量预言。"""

from __future__ import annotations

import math

import numpy as np

from tts_theory.algorithms import EmpiricalScoreCalibrator
from tts_theory.calibration_theory import (
    bernstein_zero_collapse_threshold,
    build_score_bins,
    calibration_minimax_scaling,
    compute_bin_calibrator_scores,
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


def test_v4_theorem5a_asymmetric_linear_negative_weight_reversal() -> None:
    """验证高错相关 rho0 > rho1 下等权 Mean 及全非负权重均坍塌，而负权差值组合可逆转。"""
    cov1 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    cov0 = (1.2**2) * np.array([[1.0, 0.85], [0.85, 1.0]], dtype=np.float64)
    spec = GeneralGaussianMultiVerifierSpec(
        p=0.25,
        mu1=np.array([1.6, 0.2]),
        cov1=cov1,
        mu0=np.array([0.0, 0.0]),
        cov0=cov0,
    )
    crit = optimal_linear_ensemble_criterion(spec)
    assert crit["mean_collapses"] is True
    assert crit["kappa_nonneg_min"] > 1.0
    assert crit["linear_reversible"] is True
    assert crit["negative_weight_rescues_mean"] is True
    assert crit["kappa_linear_min"] < 0.5

    n_values = [1, 16, 256, 2000]
    exact_mean = exact_bon_curve(spec.mean_verifier_spec(), n_values)
    exact_lin = exact_bon_curve(spec.linear_verifier_spec(crit["w_star"]), n_values)
    assert float(exact_mean[-1]) < 0.35
    assert float(exact_lin[-1]) > 0.99

    mc_lin = simulate_multi_verifier_bon(
        spec,
        n_values,
        aggregation="linear",
        weights=crit["w_star"],
        n_trials=30_000,
        seed=401,
        batch_size=3_000,
    )
    assert float(np.max(np.abs(exact_lin - mc_lin["accuracy"]))) < 0.01


def test_v4_theorem5a_general_min_active_set_qp_beats_mean_in_pure_gaussian() -> None:
    """验证一般正定协方差 QP KKT 精度、激活集剔除高噪裁判，以及纯高斯下 Min 逆转 Mean。"""
    cov1 = np.array(
        [
            [1.0, 0.8, 0.8],
            [0.8, 1.0, 0.8],
            [0.8, 0.8, 1.0],
        ],
        dtype=np.float64,
    )
    cov0 = np.array(
        [
            [0.4, 0.7, 0.7],
            [0.7, 4.6, 0.2],
            [0.7, 0.2, 4.6],
        ],
        dtype=np.float64,
    )
    spec = GeneralGaussianMultiVerifierSpec(
        p=0.25,
        mu1=np.array([1.0, 1.0, 0.6]),
        cov1=cov1,
        mu0=np.array([0.0, 0.0, 0.0]),
        cov0=cov0,
    )

    qp0 = solve_positive_quadrant_qp(cov0)
    # 校验全部 KKT 条件与严格激活集缩减 |I0| = 1 < 3
    assert qp0["active_set"] == [0]
    assert abs(qp0["gamma"] - 2.5) < 1e-10
    assert float(np.min(qp0["u_star"])) >= 1.0 - 1e-10
    assert float(np.min(qp0["multipliers"])) >= -1e-10
    assert float(np.max(np.abs(qp0["multipliers"] * (qp0["u_star"] - 1.0)))) < 1e-10
    assert qp0["strict_complementarity"] is True

    crit = general_min_ensemble_qp_criterion(spec)
    assert crit["gamma0"] >= crit["gamma0_cauchy_lb"] - 1e-12
    assert crit["mean_collapses"] is True
    assert crit["min_will_collapse"] is False
    assert crit["min_rescues_mean_in_gaussian"] is True

    ns = [1, 16, 256, 2000]
    mc_mean = simulate_multi_verifier_bon(
        spec, ns, aggregation="mean", n_trials=30_000, seed=402, batch_size=3_000
    )
    mc_min = simulate_multi_verifier_bon(
        spec, ns, aggregation="min", n_trials=30_000, seed=403, batch_size=3_000
    )
    assert float(mc_mean["accuracy"][-1]) < float(np.max(mc_mean["accuracy"])) - 0.05
    assert float(mc_min["accuracy"][-1]) > 0.90


def test_v4_theorem5b_gaussian_copula_unified_eta_criterion() -> None:
    """验证 Gaussian-Copula + Student-t 重尾下统一相界 alpha0 / eta0 > alpha1 / eta1。"""
    k = 4
    r1 = np.full((k, k), 0.85, dtype=np.float64)
    np.fill_diagonal(r1, 1.0)
    r0 = np.eye(k, dtype=np.float64)

    spec_rev = GaussianCopulaHeavyTailSpec(
        p=0.25, alpha1=4.5, alpha0=2.0, corr1=r1, corr0=r0, epsilon_shared=0.0
    )
    crit_rev = gaussian_copula_heavy_tail_criterion(spec_rev)
    assert crit_rev["mean_will_collapse"] is True
    assert crit_rev["min_will_collapse"] is False
    assert crit_rev["effective_alpha0_min"] > crit_rev["effective_alpha1"]

    ns = [1, 16, 256, 2048]
    mc_mean = simulate_multi_verifier_bon(
        spec_rev, ns, aggregation="mean", n_trials=20_000, seed=404, batch_size=2_500
    )
    mc_min = simulate_multi_verifier_bon(
        spec_rev, ns, aggregation="min", n_trials=20_000, seed=405, batch_size=2_500
    )
    assert float(mc_mean["accuracy"][-1]) < 0.25
    assert float(mc_min["accuracy"][-1]) > 0.88

    # 注入 6% 同源上尾相依漏洞 -> eta0_eff 跃升为 1.0，击穿 Min
    spec_brk = GaussianCopulaHeavyTailSpec(
        p=0.25, alpha1=4.5, alpha0=2.0, corr1=r1, corr0=r0, epsilon_shared=0.06
    )
    crit_brk = gaussian_copula_heavy_tail_criterion(spec_brk)
    assert crit_brk["eta0_eff"] == 1.0
    assert crit_brk["min_will_collapse"] is True
    assert crit_brk["regime"] == "shared_upper_tail_breakdown"
    mc_brk = simulate_multi_verifier_bon(
        spec_brk, ns, aggregation="min", n_trials=20_000, seed=406, batch_size=2_500
    )
    assert float(mc_brk["accuracy"][-1]) < float(np.max(mc_brk["accuracy"])) - 0.20


def test_v4_theorem6a_corrected_k_exponent_1_over_5_beats_2_over_5() -> None:
    """验证修正后的最优分桶指数 1/(2*beta+1)（beta=2 为 1/5），且大 M 遗憾斜率击败错误 2/5 指数。"""
    mm1 = calibration_minimax_scaling(500, beta=2.0)
    mm2 = calibration_minimax_scaling(500_000, beta=2.0)
    assert abs(mm1["k_exponent"] - 0.2) < 1e-12
    assert abs(mm1["regret_exponent"] - 0.4) < 1e-12
    ratio_obs = mm2["k_opt_float"] / mm1["k_opt_float"]
    ratio_th = ((500_000.0 / math.log(500_000.0)) / (500.0 / math.log(500.0))) ** 0.2
    assert abs(ratio_obs - ratio_th) < 1e-10

    mm_b1 = calibration_minimax_scaling(5_000, beta=1.0)
    assert abs(mm_b1["k_exponent"] - 1.0 / 3.0) < 1e-12
    assert abs(mm_b1["regret_exponent"] - 1.0 / 3.0) < 1e-12

    # 检验大校准样本区间（M >= 5e4 至 5e6）下 1/5 标度律随 M 的渐近遗憾优于过细 2/5 标度
    spec = GaussianVerifierSpec(p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.5)
    q_star = calibrated_bon_asymptotic_limit(spec)
    m_base = 500
    base_k = 12
    m_list = [500, 5_000, 50_000, 500_000, 5_000_000]
    reg_one_fifth = []
    reg_two_fifth = []
    for idx, m_val in enumerate(m_list):
        scale = (m_val / math.log(m_val)) / (m_base / math.log(m_base))
        k1 = max(4, int(round(base_k * (scale**0.2))))
        k2 = max(4, int(round(base_k * (scale**0.4))))
        res1 = simulate_finite_calibration_scaling(
            spec, m_cal=m_val, n_bins=k1, n_values=[50_000], binning="quantile",
            methods=("kl_lcb",), n_repeats=400, delta=0.05, seed=500 + idx
        )
        res2 = simulate_finite_calibration_scaling(
            spec, m_cal=m_val, n_bins=k2, n_values=[50_000], binning="quantile",
            methods=("kl_lcb",), n_repeats=400, delta=0.05, seed=600 + idx
        )
        reg_one_fifth.append(max(1e-7, q_star - float(res1["methods"]["kl_lcb"]["accuracy"][0])))
        reg_two_fifth.append(max(1e-7, q_star - float(res2["methods"]["kl_lcb"]["accuracy"][0])))

    # 大 M=5,000,000 下正确 1/5 指数避免方差膨胀，遗憾严格更低
    assert reg_one_fifth[-1] < reg_two_fifth[-1]
    x_log = np.log(np.array([m / math.log(m) for m in m_list]))
    slope_1_5, _ = np.polyfit(x_log, np.log(reg_one_fifth), 1)
    slope_2_5, _ = np.polyfit(x_log, np.log(reg_two_fifth), 1)
    assert -0.50 <= slope_1_5 <= -0.32
    assert slope_1_5 < slope_2_5


def test_v4_theorem6b_kl_lcb_nonasymptotic_coverage_and_bernstein_threshold() -> None:
    """验证 KL-LCB 对极端小样本 m_k=1,2 的严格非渐近覆盖率及 Empirical-Bernstein 归零阈值。"""
    rng = np.random.default_rng(707)
    delta = 0.05
    k_bins = 20
    # 对单桶单侧联合置信度 1 - delta/K = 0.9975 做 40,000 次精确二项抽样检验
    for m_val in (1, 2, 5, 15):
        for q_true in (0.1, 0.45, 0.85):
            pos = rng.binomial(m_val, q_true, size=40_000).astype(np.float64)
            # 利用内部单桶函数接口构建长度为 K 的等效阵列或单桶传参检查（注意 delta/K）
            # compute_bin_calibrator_scores 按 K 个桶除以 K；若单桶检验需设 delta_single = delta / k_bins
            counts_k = np.full(k_bins, float(m_val), dtype=np.float64)
            lcb_vals = np.empty(40_000, dtype=np.float64)
            # 每批 20 个抽样一起调用向量化 K=20 桶即可精确测试 rhs = ln(20 / 0.05)
            for b in range(0, 40_000, k_bins):
                lcb_vals[b : b + k_bins] = compute_bin_calibrator_scores(
                    counts_k, pos[b : b + k_bins], method="kl_lcb", prior_p=q_true, delta=delta
                )
            fail_rate = float(np.mean(lcb_vals > q_true + 1e-12))
            assert fail_rate <= delta / k_bins + 0.002

    # 验证 Bernstein 归零阈值在 (M=400, K=40) 必崩溃，而 KL-LCB 稳健消除坍塌
    spec = GaussianVerifierSpec(p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.5)
    q_star = calibrated_bon_asymptotic_limit(spec)
    thresh = bernstein_zero_collapse_threshold(n_bins=40, q_peak=q_star, delta=0.05)
    assert thresh > 30.0  # 每桶仅 10 个样本远低于 37.7 的归零线！
    sim = simulate_finite_calibration_scaling(
        spec, m_cal=400, n_bins=40, n_values=[1, 64, 5000], binning="quantile",
        methods=("mle", "kl_lcb", "wilson_lcb", "empirical_bernstein_lcb"),
        n_repeats=300, delta=0.05, seed=708
    )
    assert float(sim["methods"]["empirical_bernstein_lcb"]["accuracy"][-1]) < 0.28
    assert float(sim["methods"]["kl_lcb"]["accuracy"][-1]) > 0.42
    assert float(sim["methods"]["kl_lcb"]["accuracy"][-1]) > float(sim["methods"]["mle"]["accuracy"][-1])


def test_v4_theorem6c_exact_oracle_peak_rate_n_minus_2() -> None:
    """验证内点二次峰下完美校准遗憾 q* - P_Cal-BoN(N) 精确收敛于 [C / (2 f*^2)] N^{-2}。"""
    spec = GaussianVerifierSpec(p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.5)
    x_peak = gaussian_peak_threshold(spec)
    q_star = float(spec.posterior(x_peak))
    f_star = float(spec.mix_pdf(x_peak))
    inv_tau2 = 1.0 / (spec.sigma1**2) - 1.0 / (spec.sigma0**2)
    c_quad = 0.5 * q_star * (1.0 - q_star) * inv_tau2
    exact_prefactor = c_quad / (2.0 * (f_star**2))

    n_list = [200, 500, 1_000, 5_000, 10_000]
    ratios = []
    for n_val in n_list:
        acc_cal = exact_calibrated_bon_accuracy(spec, n_val, n_grid=16_384)
        reg = q_star - acc_cal
        pred = exact_prefactor * (float(n_val) ** (-2.0))
        ratios.append(reg / pred)

    # 在 N = 5,000 与 10,000 处实测值与闭式主项之比高度趋近 1.0（误差 < 2%）
    assert 0.98 <= ratios[-1] <= 1.02
    assert abs(ratios[-1] - 1.0) < abs(ratios[0] - 1.0)

    # 验证联合算力-校准标度相区判定
    j_small_n = joint_calibration_compute_scaling(m_total=50_000, n_budget=10, beta=2.0)
    j_large_n = joint_calibration_compute_scaling(m_total=500, n_budget=5_000, beta=2.0)
    assert j_small_n["regime"] == "compute_limited"
    assert j_large_n["regime"] == "calibration_limited"
    assert j_small_n["k_joint_star"] < j_small_n["k_cal_star"]


def test_v4_theorem7_simpson_masking_and_within_prompt_crossover() -> None:
    """验证定理 7 辛普森导数分解、假单调遮蔽，以及题内标准化在大 N/小 N 的双向交叉。"""
    spec_masked = PromptHeterogeneitySpec(
        prompt_weights=[0.40, 0.45, 0.15],
        prompt_p=[0.15, 0.48, 0.92],
        prompt_shift=[-2.0, 0.2, 3.5],
        mu1_base=0.7,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.35,
    )
    diag = diagnose_simpson_confounding(spec_masked)
    assert diag["decomposition_max_abs_err"] < 1e-8
    assert diag["within_prompt_collapses"] is True
    assert diag["pooled_appears_monotone"] is True
    assert diag["false_monotonicity_masking"] is True
    assert diag["masked_point_count"] >= 10

    # 强跨题偏移 + 较高方差下，全局池化校准等价于 Raw-BoN 随大 N 坍塌，而题内标准化反转击败
    spec_strong_shift = PromptHeterogeneitySpec(
        prompt_weights=[0.35, 0.35, 0.30],
        prompt_p=[0.20, 0.35, 0.75],
        prompt_shift=[-4.5, 0.0, 4.5],
        mu1_base=1.0,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.65,
    )
    res_shift = evaluate_pooled_vs_within_prompt_calibration(
        spec_strong_shift,
        n_values=[2, 64, 256],
        m_cal_prompts=600,
        n_cal_per_prompt=32,
        n_eval_prompts=3_000,
        n_bins=12,
        seed=801,
    )
    # 在大预算 N=256 下，Raw-BoN 与 Pooled-KL-LCB 双双坍塌，而 Within-Prompt-Z 显著胜出
    assert float(res_shift["within_prompt_z_kl_lcb_mc"][-1]) > float(res_shift["pooled_kl_lcb_mc"][-1]) + 0.04
    assert float(res_shift["within_prompt_z_kl_lcb_mc"][-1]) > float(res_shift["raw_bon_mc"][-1]) + 0.04

    # 零偏移 + 极小预算 N=2（非单调异方差）：样本 Z 标准化退化为只比大小，Pooled-KL-LCB 显著击败 Within-Prompt-Z
    spec_noshift = PromptHeterogeneitySpec(
        prompt_weights=[1.0],
        prompt_p=[0.30],
        prompt_shift=[0.0],
        mu1_base=0.4,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=2.2,
    )
    res_noshift = evaluate_pooled_vs_within_prompt_calibration(
        spec_noshift,
        n_values=[2, 4],
        m_cal_prompts=500,
        n_cal_per_prompt=20,
        n_eval_prompts=4_000,
        n_bins=14,
        seed=802,
    )
    assert float(res_noshift["pooled_kl_lcb_mc"][0]) > float(res_noshift["within_prompt_z_kl_lcb_mc"][0]) + 0.04
