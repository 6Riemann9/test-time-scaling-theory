"""V2 原创扩展理论单元测试：定理 5（多裁判相变逆转）与定理 6（有限校准二次坍塌）。"""

from __future__ import annotations

import numpy as np

from tts_theory.algorithms import EmpiricalScoreCalibrator
from tts_theory.calibration_theory import (
    build_score_bins,
    calibration_minimax_scaling,
    compute_bin_calibrator_scores,
    exact_conditional_bin_bon_accuracy,
    simulate_finite_calibration_scaling,
)
from tts_theory.ensemble_theory import (
    HeavyTailMultiVerifierSpec,
    HeterogeneousVulnerabilitySpec,
    MultiVerifierGaussianSpec,
    ensemble_phase_reversal_criterion,
    heavy_tail_separation_criterion,
    simulate_multi_verifier_bon,
)
from tts_theory.integrals import exact_bon_accuracy, exact_bon_curve
from tts_theory.theorems import GaussianVerifierSpec


def test_theorem5a_min_ensemble_exact_integral_vs_100k_mc() -> None:
    """验证 MinEnsemble 一维鞍点积分 vs 100,000 次蒙特卡洛及自适应 quad 一致性。"""
    n_values = [1, 4, 16, 64, 256, 1000]
    for k_ver in (2, 3):
        spec = MultiVerifierGaussianSpec(
            p=0.25,
            mu1=1.2,
            sigma1=1.0,
            mu0=0.0,
            sigma0=1.3,
            num_verifiers=k_ver,
            rho1=0.8,
            rho0=0.1,
        )
        min_spec = spec.min_verifier_spec()
        exact_gl = exact_bon_curve(min_spec, n_values, method="gauss_legendre", n_nodes=256)
        mc_res = simulate_multi_verifier_bon(
            spec, n_values, aggregation="min", n_trials=100_000, seed=42 + k_ver, batch_size=5_000
        )
        abs_err = np.abs(exact_gl - mc_res["accuracy"])
        z_scores = abs_err / np.maximum(mc_res["se"], 1e-12)

        assert float(np.mean(abs_err)) < 1.2e-3
        assert float(np.max(z_scores)) < 3.5

        # 交叉核对 quad 与 Gauss-Legendre
        for qn, g_val in zip((16, 256), (exact_gl[2], exact_gl[4])):
            q_val = exact_bon_accuracy(min_spec, qn, method="quad")
            assert abs(q_val - float(g_val)) < 1e-6


def test_theorem5a_phase_reversal_and_irreversible_regime() -> None:
    """验证高斯多裁判相变逆转临界人数 K_c 与不可逆坍塌区判据。"""
    # 算例 1: kappa=1.25, rho1=0.8, rho0=0.1 => K_c = ceil(1 + 0.5625 / 0.64375) = 2
    spec_k2_crit = MultiVerifierGaussianSpec(
        p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.25,
        num_verifiers=2, rho1=0.8, rho0=0.1,
    )
    crit1 = ensemble_phase_reversal_criterion(spec_k2_crit)
    assert crit1["reversible_by_scaling_k"] is True
    assert crit1["critical_k"] == 2
    assert crit1["will_collapse"] is False

    # 算例 2: kappa=1.65, rho1=0.85, rho0=0.10 => K_c = ceil(1 + 1.7225 / 0.57775) = 4
    spec_k2 = MultiVerifierGaussianSpec(
        p=0.25, mu1=0.9, sigma1=1.0, mu0=0.0, sigma0=1.65,
        num_verifiers=2, rho1=0.85, rho0=0.10,
    )
    spec_k4 = MultiVerifierGaussianSpec(
        p=0.25, mu1=0.9, sigma1=1.0, mu0=0.0, sigma0=1.65,
        num_verifiers=4, rho1=0.85, rho0=0.10,
    )
    c2 = ensemble_phase_reversal_criterion(spec_k2)
    c4 = ensemble_phase_reversal_criterion(spec_k4)
    assert c2["critical_k"] == 4 and c2["will_collapse"] is True
    assert c4["critical_k"] == 4 and c4["will_collapse"] is False

    ns = [1, 16, 64, 512, 5000]
    acc_k2 = exact_bon_curve(spec_k2.min_verifier_spec(), ns)
    acc_k4 = exact_bon_curve(spec_k4.min_verifier_spec(), ns)
    # K=2 < K_c=4：必过峰显著坍塌
    assert float(acc_k2[-1]) < float(np.max(acc_k2)) - 0.05
    # K=4 >= K_c=4：跨过相界逆转为单调逼近 1
    assert np.all(np.diff(acc_k4) > 0.0)
    assert float(acc_k4[-1]) > 0.90

    # 不可逆区：rho0=0.7 >= rho1 / kappa^2
    spec_irrev = MultiVerifierGaussianSpec(
        p=0.25, mu1=0.9, sigma1=1.0, mu0=0.0, sigma0=1.55,
        num_verifiers=10, rho1=0.7, rho0=0.7,
    )
    cirr = ensemble_phase_reversal_criterion(spec_irrev)
    assert cirr["reversible_by_scaling_k"] is False
    assert cirr["critical_k"] is None
    assert cirr["will_collapse"] is True
    acc_irrev = exact_bon_curve(spec_irrev.mean_verifier_spec(), ns)
    assert float(acc_irrev[-1]) < float(np.max(acc_irrev)) - 0.15


def test_theorem5b_heavy_tail_mean_vs_min_separation_and_shared_breakdown() -> None:
    """验证 Student-t 重尾下 Mean 必崩、Min 逆转以及同源漏洞击穿机理。"""
    ns = [1, 8, 64, 512, 4096]
    # alpha1=5.0, alpha0=1.8 => K_c_heavy = floor(5.0 / 1.8) + 1 = 3
    spec_clean_min = HeavyTailMultiVerifierSpec(
        p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.0,
        num_verifiers=4, alpha1=5.0, alpha0=1.8, eta1=0.15, eta0=0.15, epsilon_shared=0.0,
    )
    crit = heavy_tail_separation_criterion(spec_clean_min)
    assert crit["critical_k_heavy"] == 3
    assert crit["mean_will_collapse"] is True
    assert crit["min_will_collapse"] is False

    res_mean = simulate_multi_verifier_bon(
        spec_clean_min, ns, aggregation="mean", n_trials=25_000, seed=101, batch_size=2_500
    )
    res_min = simulate_multi_verifier_bon(
        spec_clean_min, ns, aggregation="min", n_trials=25_000, seed=102, batch_size=2_500
    )
    # Mean 随 N 增大严重坍塌；Min 跨过 K_c 逆转冲向高准确率
    assert float(res_mean["accuracy"][-1]) < 0.15
    assert float(res_min["accuracy"][-1]) > 0.85

    # 注入同源漏洞 epsilon_shared=0.05 -> Min 被击穿重新坍塌
    spec_shared = HeavyTailMultiVerifierSpec(
        p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.0,
        num_verifiers=4, alpha1=5.0, alpha0=1.8, eta1=0.15, eta0=0.15, epsilon_shared=0.05,
    )
    crit_s = heavy_tail_separation_criterion(spec_shared)
    assert crit_s["min_will_collapse"] is True
    res_min_shared = simulate_multi_verifier_bon(
        spec_shared, ns, aggregation="min", n_trials=25_000, seed=103, batch_size=2_500
    )
    assert float(res_min_shared["accuracy"][-1]) < float(np.max(res_min_shared["accuracy"])) - 0.25
    assert float(res_min_shared["accuracy"][-1]) < 0.25

    # 异质漏洞（3 健康 + 1 污染）：Mean 坍塌而 Min 保持单调增长
    spec_het = HeterogeneousVulnerabilitySpec(num_verifiers=4, n_vulnerable=1)
    res_het_mean = simulate_multi_verifier_bon(
        spec_het, ns, aggregation="mean", n_trials=25_000, seed=104, batch_size=2_500
    )
    res_het_min = simulate_multi_verifier_bon(
        spec_het, ns, aggregation="min", n_trials=25_000, seed=105, batch_size=2_500
    )
    assert float(res_het_mean["accuracy"][-1]) < 0.10
    assert float(res_het_min["accuracy"][-1]) > 0.85


def test_theorem6a_exact_conditional_bin_formula_vs_mc() -> None:
    """验证定理 6A 精确闭式平局分组条件公式与 200,000 次蒙特卡洛抽样的一致性。"""
    spec = GaussianVerifierSpec(p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.5)
    bins = build_score_bins(spec, n_bins=12, binning="uniform")
    pi_k = bins["bin_probs"]
    q_k = bins["true_bin_q"]

    # 人为构造含两对平局组的校准估值数组
    est_scores = np.array([0.1, 0.3, 0.3, 0.55, 0.72, 0.72, 0.65, 0.4, 0.25, 0.8, 0.15, 0.8])
    est_scores = est_scores[: len(pi_k)]
    n_values = [1, 4, 16, 64, 256]

    exact_acc = exact_conditional_bin_bon_accuracy(q_k, pi_k, est_scores, n_values)

    # 在离散桶分布上直接做 500,000 次蒙特卡洛（随机平局打破）验证公式无偏
    rng = np.random.default_rng(2026)
    n_trials = 500_000
    n_max = max(n_values)
    counts = np.zeros(len(n_values), dtype=np.int64)
    cdf_pi = np.cumsum(pi_k)
    cdf_pi[-1] = 1.0

    batch = 10_000
    for _ in range(n_trials // batch):
        u = rng.random((batch, n_max))
        b_idx = np.searchsorted(cdf_pi, u, side="right")
        s_drawn = est_scores[b_idx]
        y_drawn = rng.random((batch, n_max)) < q_k[b_idx]
        for k_idx, nv in enumerate(n_values):
            sub_s = s_drawn[:, :nv]
            sub_y = y_drawn[:, :nv]
            row_max = np.max(sub_s, axis=1, keepdims=True)
            is_max = np.abs(sub_s - row_max) <= 1e-13
            tie_noise = np.where(is_max, rng.random(sub_s.shape), -1.0)
            chosen = np.argmax(tie_noise, axis=1)
            counts[k_idx] += int(np.count_nonzero(sub_y[np.arange(batch), chosen]))

    mc_acc = counts.astype(np.float64) / float(n_trials)
    abs_err = np.abs(exact_acc - mc_acc)
    assert float(np.mean(abs_err)) < 1.2e-3
    assert float(np.max(abs_err)) < 3.0e-3


def test_theorem6_secondary_collapse_and_lcb_mitigation() -> None:
    """验证有限校准集 M 引发的 MLE 二次坍塌及 Wilson-LCB / Bernstein-LCB 的根治效果。"""
    spec = GaussianVerifierSpec(p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.5)
    n_values = [1, 4, 16, 64, 256, 1024, 5000]

    # 1) 非均匀等宽分桶（M=300, K=20）：MLE 出现典型过峰二次坍塌，Wilson-LCB 有效抑制
    sim_uni = simulate_finite_calibration_scaling(
        spec,
        m_cal=300,
        n_bins=20,
        n_values=n_values,
        binning="uniform",
        methods=("mle", "wilson_lcb"),
        n_repeats=500,
        delta=0.05,
        seed=97,
    )
    mle_u = sim_uni["methods"]["mle"]["accuracy"]
    wil_u = sim_uni["methods"]["wilson_lcb"]["accuracy"]
    assert float(mle_u[-1]) < float(np.max(mle_u)) - 0.03
    assert float(wil_u[-1]) >= float(np.max(wil_u)) - 0.015
    assert float(wil_u[-1]) > float(mle_u[-1]) + 0.03

    # 2) 过细等频分桶（M=300, K=80）：MLE 陷入赢家诅咒严重退化，Wilson-LCB 单调稳健胜出
    sim_q = simulate_finite_calibration_scaling(
        spec,
        m_cal=300,
        n_bins=80,
        n_values=n_values,
        binning="quantile",
        methods=("mle", "wilson_lcb"),
        n_repeats=500,
        delta=0.05,
        seed=77,
    )
    mle_q = sim_q["methods"]["mle"]["accuracy"]
    wil_q = sim_q["methods"]["wilson_lcb"]["accuracy"]
    assert float(mle_q[-1]) < float(np.max(mle_q))
    assert float(wil_q[-1]) >= float(np.max(wil_q)) - 0.005
    assert float(wil_q[-1]) > float(mle_q[-1]) + 0.03

    # 充足主峰采样区（q_peak 较高、M=2000 等宽 K=20）：Wilson-LCB 与 Empirical-Bernstein LCB 均击败 MLE
    spec_high = GaussianVerifierSpec(p=0.40, mu1=1.5, sigma1=1.0, mu0=0.0, sigma0=1.5)
    sim_bern = simulate_finite_calibration_scaling(
        spec_high,
        m_cal=2000,
        n_bins=24,
        n_values=n_values,
        binning="uniform",
        methods=("mle", "wilson_lcb", "empirical_bernstein_lcb"),
        n_repeats=400,
        delta=0.05,
        seed=99,
    )
    mle_h = sim_bern["methods"]["mle"]["accuracy"]
    wil_h = sim_bern["methods"]["wilson_lcb"]["accuracy"]
    bern_h = sim_bern["methods"]["empirical_bernstein_lcb"]["accuracy"]
    assert float(mle_h[-1]) < float(np.max(mle_h)) - 0.03
    assert float(wil_h[-1]) >= float(np.max(wil_h)) - 0.015
    assert float(bern_h[-1]) >= float(np.max(bern_h)) - 0.015
    assert float(wil_h[-1]) > float(mle_h[-1]) + 0.04
    assert float(bern_h[-1]) > float(mle_h[-1]) + 0.04

    # 测试 Minimax 最优分桶公式单调性以及 EmpiricalScoreCalibrator(method="quantile_lcb")
    mm_small = calibration_minimax_scaling(500)
    mm_large = calibration_minimax_scaling(50_000)
    assert mm_large["k_opt_theory"] > mm_small["k_opt_theory"]
    assert mm_large["regret_rate_theory"] < mm_small["regret_rate_theory"]

    rng = np.random.default_rng(123)
    y_cal = rng.random(400) < 0.25
    s_cal = np.where(y_cal, rng.normal(1.2, 1.0, 400), rng.normal(0.0, 1.5, 400))
    cal_lcb = EmpiricalScoreCalibrator(n_bins=20, method="quantile_lcb", lcb_delta=0.05).fit(
        s_cal, y_cal
    )
    cal_mle = EmpiricalScoreCalibrator(n_bins=20, prior_strength=0.0, method="quantile_hist").fit(
        s_cal, y_cal
    )
    assert float(np.max(cal_lcb.knots_q_)) < float(np.max(cal_mle.knots_q_))
    probs = cal_lcb.predict_proba(np.array([-2.0, 1.0, 6.0]))
    assert probs.shape == (3,) and np.all((probs > 0.0) & (probs < 1.0))
