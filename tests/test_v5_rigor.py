"""V5 审稿修补与严格性单元测试：针对 9 处核心数学修补点逐一做数值与边界验证。"""

from __future__ import annotations

import math

import numpy as np
from scipy import integrate, optimize, special
from scipy.stats import norm

from tts_theory.calibration_theory import (
    compute_bin_calibrator_scores,
    exact_conditional_bin_bon_accuracy,
    joint_calibration_compute_scaling,
)
from tts_theory.ensemble_theory import (
    GaussianCopulaHeavyTailSpec,
    GeneralGaussianMultiVerifierSpec,
    gaussian_copula_heavy_tail_criterion,
    general_min_ensemble_qp_criterion,
    general_min_tail_asymptotic_constants,
    solve_positive_quadrant_qp,
)
from tts_theory.heterogeneity_theory import (
    PromptHeterogeneitySpec,
    diagnose_within_prompt_residual,
)
from tts_theory.integrals import exact_bon_accuracy, exact_calibrated_bon_accuracy
from tts_theory.theorems import (
    BinaryVerifierSpec,
    GaussianVerifierSpec,
    _gaussian_mix_log_sf_scalar,
    binary_bon_accuracy,
    binary_bon_divergence_bounds,
    gaussian_peak_threshold,
    weighted_majority_convergence_bound,
)


def test_v5_thm2b_c0_continuous_counterexample_two_intervals() -> None:
    """定理 2B(2)：检验纯 C^0（含非 C^1 尖点）正密度在双闭子区间并 A=I1 U I2 上打破单调性。"""
    p = 0.4
    mu1, b1 = 0.6, 0.8
    mu0, b0 = 0.0, 1.5

    def pdf1(x: float) -> float:
        return math.exp(-abs(x - mu1) / b1) / (2.0 * b1)

    def pdf0(x: float) -> float:
        return math.exp(-abs(x - mu0) / b0) / (2.0 * b0)

    def f_mix(x: float) -> float:
        return p * pdf1(x) + (1.0 - p) * pdf0(x)

    def post(x: float) -> float:
        return (p * pdf1(x)) / f_mix(x)

    # x1 = 0.6 为 pdf1 的不可微尖点；对 x >= 0.6，似然比随 x 严格递减
    x1, x2 = 0.6, 1.5
    delta = post(x1) - post(x2)
    assert delta > 0.05

    w1_lim = f_mix(x1) / (f_mix(x1) + f_mix(x2))
    w2_lim = f_mix(x2) / (f_mix(x1) + f_mix(x2))
    expected_asym_drop = -w1_lim * w2_lim * delta

    drops = []
    for eps in (0.15, 0.04, 0.01):
        i1 = (x1 - eps, x1 + eps)
        i2 = (x2 - eps, x2 + eps)
        m1, _ = integrate.quad(f_mix, i1[0], i1[1], epsabs=1e-13, epsrel=1e-12)
        m2, _ = integrate.quad(f_mix, i2[0], i2[1], epsabs=1e-13, epsrel=1e-12)
        mass_a = m1 + m2

        p1, _ = integrate.quad(lambda x: p * pdf1(x), i1[0], i1[1], epsabs=1e-13, epsrel=1e-12)
        p2, _ = integrate.quad(lambda x: p * pdf1(x), i2[0], i2[1], epsabs=1e-13, epsrel=1e-12)
        bon1_a = (p1 + p2) / mass_a

        # 计算 A = I1 U I2 上两个独立样本最大值的期望后验
        def integrand_bon2_i1(x: float) -> float:
            sub, _ = integrate.quad(f_mix, i1[0], x, epsabs=1e-12, epsrel=1e-11)
            return 2.0 * post(x) * (f_mix(x) / mass_a) * (sub / mass_a)

        def integrand_bon2_i2(x: float) -> float:
            sub, _ = integrate.quad(f_mix, i2[0], x, epsabs=1e-12, epsrel=1e-11)
            cdf_a = (m1 + sub) / mass_a
            return 2.0 * post(x) * (f_mix(x) / mass_a) * cdf_a

        b2_1, _ = integrate.quad(integrand_bon2_i1, i1[0], i1[1], epsabs=1e-12, epsrel=1e-11)
        b2_2, _ = integrate.quad(integrand_bon2_i2, i2[0], i2[1], epsabs=1e-12, epsrel=1e-11)
        bon2_a = b2_1 + b2_2
        drops.append(bon2_a - bon1_a)

    # 所有尺度下均出现严格负增量，且趋近理论值 -w1* w2* delta
    assert all(d < -1e-3 for d in drops)
    assert abs(drops[-1] - expected_asym_drop) < 2e-3


def test_v5_thm2c_fixed_peak_height_gumbel_limit() -> None:
    """定理 2C：在固定峰值高度渐近机制下验证 N*_exact / N*_0 -> e^{-gamma} ~ 0.561459。"""
    e_minus_gamma = math.exp(-0.5772156649015329)
    u_nodes, u_weights = np.polynomial.legendre.leggauss(96)
    v_weights = 0.5 * u_weights
    log_v = np.log(0.5 * (u_nodes + 1.0))

    def bon_at_log_n(spec: GaussianVerifierSpec, log_n: float) -> float:
        n_val = math.exp(log_n) if log_n < 700.0 else None
        if n_val is not None and n_val < 1e12:
            log_q = np.log(-np.expm1(log_v / n_val))
        else:
            log_q = np.log(-log_v) - log_n
        lp, lq = math.log(spec.p), math.log1p(-spec.p)
        x = spec.mu0 + spec.sigma0 * np.sqrt(np.maximum(1.0, -2.0 * log_q))
        for _ in range(30):
            z1 = (x - spec.mu1) / spec.sigma1
            z0 = (x - spec.mu0) / spec.sigma0
            ls1 = lp + special.log_ndtr(-z1)
            ls0 = lq + special.log_ndtr(-z0)
            hi = np.maximum(ls1, ls0)
            lsf = hi + np.log1p(np.exp(np.minimum(ls1, ls0) - hi))
            lp1 = lp - 0.5 * math.log(2.0 * math.pi) - math.log(spec.sigma1) - 0.5 * z1 * z1
            lp0 = lq - 0.5 * math.log(2.0 * math.pi) - math.log(spec.sigma0) - 0.5 * z0 * z0
            hp = np.maximum(lp1, lp0)
            lpdf = hp + np.log1p(np.exp(np.minimum(lp1, lp0) - hp))
            fval = lsf - log_q
            grad = -np.exp(np.clip(lpdf - lsf, -700.0, 700.0))
            x = x - np.clip(fval / grad, -5.0, 5.0)
            if np.max(np.abs(fval)) < 1e-13:
                break
        return float(np.dot(v_weights, np.asarray(spec.posterior(x), dtype=np.float64)))

    ratios = []
    peak_qs = []
    for kappa in (1.10, 1.05, 1.02, 1.01):
        dmu = math.sqrt(4.0 * (kappa * kappa - 1.0))
        spec = GaussianVerifierSpec(p=0.3, mu1=dmu, sigma1=1.0, mu0=0.0, sigma0=kappa)
        x_pk = gaussian_peak_threshold(spec)
        peak_qs.append(float(spec.posterior(x_pk)))
        log_n0 = -_gaussian_mix_log_sf_scalar(spec, x_pk)
        opt = optimize.minimize_scalar(
            lambda d: -bon_at_log_n(spec, log_n0 + float(d)),
            bounds=(-1.2, 0.2),
            method="bounded",
            options={"xatol": 1e-6},
        )
        ratios.append(math.exp(float(opt.x)))

    # 峰值后验保持有界远离 1，且比值单调收敛到 e^{-gamma}
    assert all(0.70 < q < 0.85 for q in peak_qs)
    assert ratios[0] > ratios[1] > ratios[2] > ratios[3] > e_minus_gamma
    assert abs(ratios[-1] - e_minus_gamma) / e_minus_gamma < 0.015


def test_v5_thm3_binary_ppv_one_boundary() -> None:
    """定理 3：校验 FPR=0 (PPV=1) 边界下无除零错误、前向散度为 +inf 且 TV 与反向 KL 精确匹配。"""
    spec_perf = BinaryVerifierSpec(p=0.3, tpr=0.7, fpr=0.0)
    assert spec_perf.ppv == 1.0
    q1 = spec_perf.p * spec_perf.tpr

    for n_val in (1, 3, 10):
        p_n = float(binary_bon_accuracy(spec_perf, n_val))
        expected_p_n = 1.0 - ((1.0 - spec_perf.p) / (1.0 - q1)) * ((1.0 - q1) ** n_val)
        assert abs(p_n - expected_p_n) < 1e-13

        bounds = binary_bon_divergence_bounds(spec_perf, n_val)
        assert bounds["is_ppv_one_boundary"] is True
        assert math.isinf(bounds["kl_forward"])
        assert math.isinf(bounds["chi2_upper_bound"])
        assert abs(bounds["tv_distance"] - (1.0 - expected_p_n)) < 1e-13
        assert abs(bounds["kl_reverse"] - (-math.log(expected_p_n))) < 1e-13

    spec_noisy = BinaryVerifierSpec(p=0.3, tpr=0.8, fpr=0.1)
    bounds_noisy = binary_bon_divergence_bounds(spec_noisy, 5)
    assert bounds_noisy["is_ppv_one_boundary"] is False
    assert 0.0 < bounds_noisy["kl_forward"] <= bounds_noisy["chi2_upper_bound"] < math.inf
    assert math.isfinite(bounds_noisy["kl_reverse"])


def test_v5_thm4b_heterogeneous_wrong_weight_counterexample() -> None:
    """定理 4B：验证单错类高权反例会推翻全错类平均权重判据，新接口准确检出并计算超级尾桶阈值。"""
    p = 0.20
    mu_w_pos = 0.60
    w_max = 1.0
    wrong_probs = [0.15, 0.30, 0.35]
    wrong_weights = [0.95, 0.10, 0.10]

    # 错类混合条件均权仅 0.259375，若仅传混合均权会误报 0.12 - 0.35*0.259375 > 0 一致
    mu_w_neg_agg = sum(pj * wj for pj, wj in zip(wrong_probs, wrong_weights)) / sum(wrong_probs)
    old_naive = weighted_majority_convergence_bound(
        p=p, p_wrong_max=max(wrong_probs), mu_w_pos=mu_w_pos, mu_w_neg=mu_w_neg_agg, w_max=w_max
    )
    assert old_naive["is_consistent"] is True

    # 传入各错类各自期望权重后：诱饵 0 得分 0.15*0.95=0.1425 > 0.12，精确检出失效
    exact_res = weighted_majority_convergence_bound(
        p=p,
        p_wrong_max=max(wrong_probs),
        mu_w_pos=mu_w_pos,
        mu_w_neg=mu_w_neg_agg,
        w_max=w_max,
        wrong_answer_probs=wrong_probs,
        wrong_mean_weights=wrong_weights,
    )
    assert exact_res["is_consistent"] is False
    assert abs(exact_res["margin"] - (0.12 - 0.1425)) < 1e-12
    assert abs(exact_res["effective_max_wrong_prob"] - 0.15) < 1e-12

    # 压低诱饵 0 权重后恢复一致，并检查超级尾桶概率阈值 p*mu_{w,+}/(2*w_max)
    safe_res = weighted_majority_convergence_bound(
        p=p,
        p_wrong_max=max(wrong_probs),
        mu_w_pos=mu_w_pos,
        mu_w_neg=mu_w_neg_agg,
        w_max=w_max,
        wrong_answer_probs=wrong_probs,
        wrong_mean_weights=[0.50, 0.10, 0.10],
    )
    assert safe_res["is_consistent"] is True
    assert abs(safe_res["margin"] - (0.12 - 0.075)) < 1e-12
    assert abs(safe_res["tail_bucket_prob_threshold"] - (p * mu_w_pos) / (2.0 * w_max)) < 1e-12


def test_v5_thm5a2_exact_pdf_constant_gamma_factor() -> None:
    """定理 5A-II：验证非齐次高斯最小值密度精确前因子 C_pdf = gamma * C_sf（补全遗漏 gamma 因子）。"""
    # 1. 2D 全激活情况精确边界积分比对
    mu2 = np.array([0.8, -0.3])
    cov2 = np.array([[1.4, 0.5], [0.5, 1.1]])
    c2 = general_min_tail_asymptotic_constants(mu2, cov2)
    assert c2["active_set"] == [0, 1]
    assert c2["strict_complementarity"] is True
    assert abs(c2["c_pdf"] - c2["gamma"] * c2["c_sf"]) < 1e-14 * c2["c_pdf"]

    s1, s2 = math.sqrt(cov2[0, 0]), math.sqrt(cov2[1, 1])
    rho = cov2[0, 1] / (s1 * s2)
    ratios_2d = []
    for t in (15.0, 30.0, 100.0):
        m2_1 = mu2[1] + rho * (s2 / s1) * (t - mu2[0])
        v2_1 = s2 * math.sqrt(1.0 - rho * rho)
        m1_2 = mu2[0] + rho * (s1 / s2) * (t - mu2[1])
        v1_2 = s1 * math.sqrt(1.0 - rho * rho)
        lt1 = -0.5 * math.log(2.0 * math.pi) - math.log(s1) - 0.5 * ((t - mu2[0]) / s1) ** 2 + float(special.log_ndtr(-(t - m2_1) / v2_1))
        lt2 = -0.5 * math.log(2.0 * math.pi) - math.log(s2) - 0.5 * ((t - mu2[1]) / s2) ** 2 + float(special.log_ndtr(-(t - m1_2) / v1_2))
        hi = max(lt1, lt2)
        log_exact_pdf = hi + math.log1p(math.exp(min(lt1, lt2) - hi))
        log_asym_pdf = c2["log_c_pdf"] - math.log(t) - 0.5 * c2["gamma"] * t * t + c2["beta"] * t
        ratios_2d.append(math.exp(log_exact_pdf - log_asym_pdf))

    assert 0.99 <= ratios_2d[-1] <= 1.01
    assert abs(ratios_2d[-1] - 1.0) < abs(ratios_2d[0] - 1.0)

    # 2. 3D 部分激活情况（|I|=2 < K=3），用真 3D 条件一维积分核验 C_pdf = gamma * C_sf
    cov3 = np.array([[1.0, 0.2, 0.9], [0.2, 1.0, 0.9], [0.9, 0.9, 2.5]])
    mu3 = np.array([0.4, -0.2, 0.7])
    c3 = general_min_tail_asymptotic_constants(mu3, cov3)
    assert c3["active_set"] == [0, 1]
    assert abs(c3["gamma"] - 5.0 / 3.0) < 1e-12
    assert abs(c3["c_pdf"] - c3["gamma"] * c3["c_sf"]) < 1e-14 * c3["c_pdf"]

    def exact_3d_min_pdf(t: float) -> float:
        total = 0.0
        for k in range(3):
            others = [j for j in range(3) if j != k]
            # S_others | (S_k = t) ~ N(m_cond, S_cond)
            c_ok = cov3[np.ix_(others, [k])][:, 0] / cov3[k, k]
            m_cond = mu3[others] + c_ok * (t - mu3[k])
            s_cond = cov3[np.ix_(others, others)] - np.outer(c_ok, cov3[k, others])
            # 对 (U0, U1) | (S_k=t) 沿 u0 in [t, t+12] 将 u1 解析积出 sf
            sd0 = math.sqrt(s_cond[0, 0])
            rho_c = s_cond[0, 1] / (sd0 * math.sqrt(s_cond[1, 1]))
            sd1_given0 = math.sqrt(s_cond[1, 1] * (1.0 - rho_c * rho_c))

            def inner_prob(u0: float) -> float:
                m1_g0 = m_cond[1] + rho_c * (math.sqrt(s_cond[1, 1]) / sd0) * (u0 - m_cond[0])
                return float(norm.pdf(u0, m_cond[0], sd0) * norm.sf(t, m1_g0, sd1_given0))

            prob_others, _ = integrate.quad(inner_prob, t, t + 12.0, epsabs=0.0, epsrel=1e-9)
            total += float(norm.pdf(t, mu3[k], math.sqrt(cov3[k, k]))) * prob_others
        return total

    t_test = 18.0
    pdf_3d = exact_3d_min_pdf(t_test)
    asym_3d = c3["c_pdf"] * (t_test ** (-1)) * math.exp(-0.5 * c3["gamma"] * t_test * t_test + c3["beta"] * t_test)
    assert 0.97 <= pdf_3d / asym_3d <= 1.03

    # 校验 general_min_ensemble_qp_criterion 暴露了对应常数字段
    spec3 = GeneralGaussianMultiVerifierSpec(p=0.25, mu1=mu3, cov1=cov3, mu0=np.zeros(3), cov0=np.eye(3))
    crit3 = general_min_ensemble_qp_criterion(spec3)
    assert abs(crit3["c_pdf1"] - c3["c_pdf"]) < 1e-14


def test_v5_thm5b2_negative_correlation_eta_below_one_over_k() -> None:
    """定理 5B-II：验证审稿负相关反例 eta = 1/4 < 1/K，及非负相关 Slepian 界 eta in [1/K, 1]。"""
    # 1. 审稿人反例 R = [[1, -0.5], [-0.5, 1]] => gamma=4, eta=1/4 < 1/2
    r_neg = np.array([[1.0, -0.5], [-0.5, 1.0]])
    qp = solve_positive_quadrant_qp(r_neg)
    assert abs(qp["gamma"] - 4.0) < 1e-12
    eta_neg = 1.0 / qp["gamma"]
    assert abs(eta_neg - 0.25) < 1e-12
    assert eta_neg < 0.5  # 突破 1/K 下界

    # 更强负相关 rho=-0.6 => gamma=5, eta=0.2 => 无需 K=3 即可在 K=2 下逆转 alpha0=2, alpha1=4
    r_strong_neg = np.array([[1.0, -0.6], [-0.6, 1.0]])
    spec_neg_rescue = GaussianCopulaHeavyTailSpec(
        p=0.25, alpha1=4.0, alpha0=2.0, corr1=np.eye(2), corr0=r_strong_neg, epsilon_shared=0.0
    )
    crit = gaussian_copula_heavy_tail_criterion(spec_neg_rescue)
    assert crit["is_nonneg_corr0"] is False
    assert crit["is_nonneg_corr1"] is True
    assert crit["eta0_below_independent_floor"] is True
    assert abs(crit["eta0_copula"] - 0.20) < 1e-12
    assert crit["effective_alpha0_min"] > crit["effective_alpha1"]
    assert crit["min_will_collapse"] is False

    # 2. 随机抽取 20 组 K in {2,3,4,5} 非负相关 SPD 矩阵验证 Slepian 界 eta in [1/K, 1]
    rng = np.random.default_rng(505)
    for k_dim in (2, 3, 4, 5):
        for _ in range(5):
            # 非负因子构造确保 R_{ij} >= 0 且严格正定
            f = rng.uniform(0.0, 1.0, size=(k_dim, k_dim + 2))
            gram = f @ f.T + 0.1 * np.eye(k_dim)
            d_inv = np.diag(1.0 / np.sqrt(np.diag(gram)))
            r_pos = d_inv @ gram @ d_inv
            assert np.all(r_pos >= -1e-12)
            eta_k = 1.0 / solve_positive_quadrant_qp(r_pos)["gamma"]
            assert (1.0 / float(k_dim) - 1e-11) <= eta_k <= (1.0 + 1e-11)


def test_v5_thm6b_lcb_safe_floor_vs_non_perfect_ranking() -> None:
    """定理 6B：复现 A(0.90, LCB) vs B(0.60, LCB) + 低质稀疏诱饵 C(0.10, 1/1) 对比，验证安全保底但非完美排序。"""
    q_true = np.array([0.90, 0.60, 0.10], dtype=np.float64)
    pi_true = np.array([0.10, 0.80, 0.10], dtype=np.float64)
    counts = np.array([1.0, 100.0, 1.0], dtype=np.float64)
    positives = np.array([1.0, 60.0, 1.0], dtype=np.float64)

    est_mle = compute_bin_calibrator_scores(counts, positives, method="mle", prior_p=0.5, delta=0.05)
    est_kl = compute_bin_calibrator_scores(counts, positives, method="kl_lcb", prior_p=0.5, delta=0.05)

    # MLE 把 C(q=0.10) 和 A(q=0.90) 并列排第一 (1.0)，导致大 N 落到均权 0.50 < 0.60
    assert est_mle[2] == est_mle[0] == 1.0 > est_mle[1]
    mle_large_n = float(exact_conditional_bin_bon_accuracy(q_true, pi_true, est_mle, [10_000])[0])
    assert abs(mle_large_n - 0.50) < 1e-3

    # KL-LCB 不在高估算的任何桶超过 q_true（覆盖保证），把 B 放在首位而非 A（不保证完美排序！），
    # 但严格封杀 C 进入 G1 且大 N 正确率 (0.60) 严格高于 max(est_kl)
    assert bool(np.all(est_kl <= q_true + 1e-12))
    assert est_kl[1] > est_kl[0] == est_kl[2]
    kl_large_n = float(exact_conditional_bin_bon_accuracy(q_true, pi_true, est_kl, [10_000])[0])
    assert abs(kl_large_n - 0.60) < 1e-3
    assert kl_large_n >= float(np.max(est_kl))


def test_v5_thm6c_joint_minimax_rates_and_separated_peak() -> None:
    """定理 6C：校验联合 (M,N) 上下界字段切换，以及双峰等高违反唯一分离假设时收敛率缩放 2^{-beta}。"""
    # 1. 联合上下界诊断检验
    j_cal = joint_calibration_compute_scaling(m_total=400, n_budget=10_000, beta=2.0)
    assert j_cal["regime"] == "calibration_limited"
    assert j_cal["is_minimax_tight"] is True
    assert abs(j_cal["failure_prob_delta_opt"] - 1.0 / 400.0) < 1e-15
    assert j_cal["kl_lcb_upper_rate"] >= j_cal["minimax_lower_rate"]

    j_comp = joint_calibration_compute_scaling(m_total=200_000, n_budget=25, beta=2.0)
    assert j_comp["regime"] == "compute_limited"
    assert j_comp["is_minimax_tight"] is False
    assert j_comp["compute_log_gap_factor"] > 1.0

    # 2. 全局唯一分离峰 vs 双等高分离峰（在均匀边际 [-2, 2] 上解析对比，具精确闭式 cdf1/sf1）
    class _UnimodalVsBimodalSpec:
        def __init__(self, bimodal: bool) -> None:
            self.bimodal = bimodal
            self.c_quad = 0.08
            self.q_peak = 0.80
            # 计算严格归一化先验 p = int_{-2}^2 q(x) * 0.25 dx
            self.p = float(self._antideriv(np.array([2.0]))[0] - self._antideriv(np.array([-2.0]))[0]) * 0.25

        def _q(self, x: np.ndarray) -> np.ndarray:
            arr = np.asarray(x, dtype=np.float64)
            if not self.bimodal:
                return self.q_peak - self.c_quad * arr * arr
            dist = np.abs(arr) - 1.0
            return self.q_peak - self.c_quad * dist * dist

        def _antideriv(self, x: np.ndarray) -> np.ndarray:
            arr = np.clip(np.asarray(x, dtype=np.float64), -2.0, 2.0)
            if not self.bimodal:
                return self.q_peak * arr - (self.c_quad / 3.0) * (arr**3)
            return np.where(
                arr <= 0.0,
                self.q_peak * arr - (self.c_quad / 3.0) * ((arr + 1.0) ** 3),
                self.q_peak * arr - (self.c_quad / 3.0) * ((arr - 1.0) ** 3) - (2.0 * self.c_quad / 3.0),
            )

        def pdf1(self, x):
            arr = np.asarray(x, dtype=np.float64)
            in_supp = (arr >= -2.0) & (arr <= 2.0)
            return np.where(in_supp, self._q(arr) * 0.25 / self.p, 0.0)

        def pdf0(self, x):
            arr = np.asarray(x, dtype=np.float64)
            in_supp = (arr >= -2.0) & (arr <= 2.0)
            return np.where(in_supp, (1.0 - self._q(arr)) * 0.25 / (1.0 - self.p), 0.0)

        def cdf1(self, x):
            arr = np.asarray(x, dtype=np.float64)
            base = float(self._antideriv(np.array([-2.0]))[0])
            mass = np.clip((self._antideriv(arr) - base) * 0.25 / self.p, 0.0, 1.0)
            return np.where(arr <= -2.0, 0.0, np.where(arr >= 2.0, 1.0, mass))

        def cdf0(self, x):
            return np.clip((self.mix_cdf(x) - self.p * self.cdf1(x)) / (1.0 - self.p), 0.0, 1.0)

        def sf1(self, x):
            return 1.0 - self.cdf1(x)

        def sf0(self, x):
            return 1.0 - self.cdf0(x)

        def mix_pdf(self, x):
            arr = np.asarray(x, dtype=np.float64)
            return np.where((arr >= -2.0) & (arr <= 2.0), 0.25, 0.0)

        def mix_cdf(self, x):
            arr = np.asarray(x, dtype=np.float64)
            return np.clip((arr + 2.0) / 4.0, 0.0, 1.0)

        def mix_sf(self, x):
            return 1.0 - self.mix_cdf(x)

        def mix_isf(self, q):
            return 2.0 - 4.0 * np.asarray(q, dtype=np.float64)

        def mix_ppf(self, u):
            return -2.0 + 4.0 * np.asarray(u, dtype=np.float64)

        def posterior(self, x):
            return self._q(x)

        def log_likelihood_ratio(self, x):
            q = np.clip(self._q(x), 1e-12, 1.0 - 1e-12)
            return np.log(q / (1.0 - q)) - math.log(self.p / (1.0 - self.p))

    uni = _UnimodalVsBimodalSpec(bimodal=False)
    bi = _UnimodalVsBimodalSpec(bimodal=True)
    n_large = 2000
    reg_uni = uni.q_peak - exact_calibrated_bon_accuracy(uni, n_large, n_grid=16_000)
    reg_bi = bi.q_peak - exact_calibrated_bon_accuracy(bi, n_large, n_grid=16_000)

    pred_uni = (uni.c_quad / (2.0 * (0.25**2))) * (float(n_large) ** (-2.0))
    assert 0.97 <= reg_uni / pred_uni <= 1.03
    # 双峰等高时主项精确缩减至 2^{-beta} = 1/4
    assert 0.24 <= reg_bi / reg_uni <= 0.26


def test_v5_thm7b_n2_rank_only_rules_and_prior_residual() -> None:
    """定理 7B：验证 N=2 纯秩天花板 max(Raw, Anti, p) < Oracle，及 p(X) 恒定 vs 异质残差对比。"""
    # 1. N=2 三类纯秩规则与非单调 Oracle 对比
    single_spec = GaussianVerifierSpec(p=0.30, mu1=0.3, sigma1=1.0, mu0=0.0, sigma0=2.3)
    raw_n2 = float(exact_bon_accuracy(single_spec, 2))
    # Anti-BoN(2) 精确积分：令 w = sqrt(1 - v) in (0, 1) => dv = 2 w dw，消除 v=1 处根号奇点
    u_g, w_g = np.polynomial.legendre.leggauss(128)
    w_node = 0.5 * (u_g + 1.0)
    w_weight = 0.5 * w_g
    x_min = np.asarray(single_spec.mix_isf(w_node), dtype=np.float64)
    anti_n2_integral = float(np.dot(2.0 * w_node * w_weight, np.asarray(single_spec.posterior(x_min), dtype=np.float64)))
    assert abs(anti_n2_integral - (2.0 * single_spec.p - raw_n2)) < 1e-12

    rank_ceiling = max(raw_n2, anti_n2_integral, single_spec.p)
    oracle_n2 = float(exact_calibrated_bon_accuracy(single_spec, 2, n_grid=12_000))
    assert oracle_n2 > rank_ceiling + 0.015

    # 2. 恒定先验 vs 异质先验 Z-Score 残差诊断
    spec_const_p = PromptHeterogeneitySpec(
        prompt_weights=[0.3, 0.4, 0.3],
        prompt_p=[0.30, 0.30, 0.30],
        prompt_shift=[-4.0, 0.0, 4.0],
        prompt_scale=[0.8, 1.0, 1.5],
        mu1_base=0.8,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=2.0,
    )
    res_const = diagnose_within_prompt_residual(spec_const_p)
    assert res_const["is_constant_prior"] is True
    assert res_const["prior_induced_shift_var"] < 1e-12
    assert res_const["peak_z_span"] < 1e-12
    assert res_const["affine_shift_var"] > 5.0
    assert res_const["oracle_cal_n2"] > res_const["rank_only_n2_ceiling"] + 0.01

    spec_het_p = PromptHeterogeneitySpec(
        prompt_weights=[0.3, 0.4, 0.3],
        prompt_p=[0.15, 0.45, 0.85],
        prompt_shift=[-4.0, 0.0, 4.0],
        prompt_scale=[0.8, 1.0, 1.5],
        mu1_base=0.8,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=2.0,
    )
    res_het = diagnose_within_prompt_residual(spec_het_p)
    assert res_het["is_constant_prior"] is False
    assert res_het["prior_induced_shift_var"] > 0.03
    assert res_het["peak_z_span"] > 0.15
