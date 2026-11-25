"""定理 1-4 的精确数值、解析公式与蒙特卡洛交叉验证单元测试。"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from tts_theory.integrals import (
    exact_bon_accuracy,
    exact_bon_curve,
    exact_calibrated_bon_accuracy,
    numerical_optimal_n,
)
from tts_theory.simulator import (
    DiscreteAnswerTaskSpec,
    simulate_algorithms_comparison,
    simulate_binary_bon,
    simulate_bon,
)
from tts_theory.theorems import (
    BinaryVerifierSpec,
    ContaminatedVerifierSpec,
    GaussianVerifierSpec,
    binary_asymptotic_ceiling,
    binary_bon_accuracy,
    binary_marginal_gain,
    binary_optimal_budget,
    calibrated_bon_asymptotic_limit,
    check_collapse_criterion,
    closed_form_optimal_n,
    gaussian_peak_threshold,
    weighted_majority_convergence_bound,
)


class _ShiftedLogisticSpec:
    """自定义非高斯 ContinuousVerifierSpec，用于验证积分器与通用抽样器的多态兼容性。"""

    def __init__(self, p: float = 0.3, loc1: float = 1.5, scale1: float = 1.0, loc0: float = 0.0, scale0: float = 1.0) -> None:
        self.p = float(p)
        self.loc1 = float(loc1)
        self.scale1 = float(scale1)
        self.loc0 = float(loc0)
        self.scale0 = float(scale0)

    def pdf1(self, x: np.ndarray | float) -> np.ndarray | float:
        return stats.logistic.pdf(x, loc=self.loc1, scale=self.scale1)

    def pdf0(self, x: np.ndarray | float) -> np.ndarray | float:
        return stats.logistic.pdf(x, loc=self.loc0, scale=self.scale0)

    def cdf1(self, x: np.ndarray | float) -> np.ndarray | float:
        return stats.logistic.cdf(x, loc=self.loc1, scale=self.scale1)

    def cdf0(self, x: np.ndarray | float) -> np.ndarray | float:
        return stats.logistic.cdf(x, loc=self.loc0, scale=self.scale0)

    def sf1(self, x: np.ndarray | float) -> np.ndarray | float:
        return stats.logistic.sf(x, loc=self.loc1, scale=self.scale1)

    def sf0(self, x: np.ndarray | float) -> np.ndarray | float:
        return stats.logistic.sf(x, loc=self.loc0, scale=self.scale0)

    def mix_pdf(self, x: np.ndarray | float) -> np.ndarray | float:
        return self.p * self.pdf1(x) + (1.0 - self.p) * self.pdf0(x)

    def mix_cdf(self, x: np.ndarray | float) -> np.ndarray | float:
        return self.p * self.cdf1(x) + (1.0 - self.p) * self.cdf0(x)

    def mix_sf(self, x: np.ndarray | float) -> np.ndarray | float:
        return self.p * self.sf1(x) + (1.0 - self.p) * self.sf0(x)

    def mix_isf(self, q: np.ndarray | float) -> np.ndarray | float:
        is_scalar = np.ndim(q) == 0
        q_arr = np.atleast_1d(np.asarray(q, dtype=np.float64))
        lo = np.full_like(q_arr, -40.0)
        hi = np.full_like(q_arr, 40.0)
        for _ in range(10):
            need_lo = self.mix_sf(lo) < q_arr
            need_hi = self.mix_sf(hi) > q_arr
            lo = np.where(need_lo, lo * 2.0, lo)
            hi = np.where(need_hi, hi * 2.0, hi)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            go_right = self.mix_sf(mid) > q_arr
            lo = np.where(go_right, mid, lo)
            hi = np.where(go_right, hi, mid)
        res = 0.5 * (lo + hi)
        return float(res[0]) if is_scalar else res

    def posterior(self, x: np.ndarray | float) -> np.ndarray | float:
        num = self.p * self.pdf1(x)
        den = num + (1.0 - self.p) * self.pdf0(x)
        return np.where(den > 0.0, num / den, 0.0) if np.ndim(x) > 0 else (float(num / den) if den > 0.0 else 0.0)

    def log_likelihood_ratio(self, x: np.ndarray | float) -> np.ndarray | float:
        return np.log(np.maximum(self.pdf1(x), 1e-300)) - np.log(np.maximum(self.pdf0(x), 1e-300))


def test_theorem1_quad_vs_gauss_legendre_extreme_n() -> None:
    """验证定理 1 积分引擎在 N in [1, 10^5] 上分段 Gauss-Legendre 与自适应 quad 双精度一致。"""
    specs = [
        GaussianVerifierSpec(p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.0),
        GaussianVerifierSpec(p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.5),
        ContaminatedVerifierSpec(
            p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.0,
            epsilon=0.08, mu_hack=3.5, sigma_hack=1.5,
        ),
    ]
    n_test = [1, 2, 16, 128, 1000, 10_000, 100_000]
    quad_check = {1, 16, 1000, 100_000}
    for spec in specs:
        assert exact_bon_accuracy(spec, 1) == pytest.approx(spec.p, abs=1e-12)
        for n in n_test:
            gl_256 = exact_bon_accuracy(spec, n, method="gauss_legendre", n_nodes=256)
            gl_512 = exact_bon_accuracy(spec, n, method="gauss_legendre", n_nodes=512)
            assert np.isfinite(gl_256)
            assert 0.0 <= gl_256 <= 1.0
            assert abs(gl_256 - gl_512) < 1e-10
            if n in quad_check:
                quad_val = exact_bon_accuracy(spec, n, method="quad")
                assert abs(gl_256 - quad_val) < 1e-8, f"N={n}: gl={gl_256}, quad={quad_val}"


def test_theorem1_exact_vs_100k_monte_carlo() -> None:
    """验证定理 1 精确顺序统计量积分与 100,000 次蒙特卡洛误差 < 1e-3（跨三种分布）。"""
    specs = [
        GaussianVerifierSpec(p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.0),
        GaussianVerifierSpec(p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.5),
        ContaminatedVerifierSpec(
            p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.0,
            epsilon=0.08, mu_hack=3.5, sigma_hack=1.5,
        ),
    ]
    n_values = [1, 4, 16, 64, 256, 1000]
    for idx, spec in enumerate(specs):
        exact_vals = exact_bon_curve(spec, n_values, method="gauss_legendre", n_nodes=256)
        mc_res = simulate_bon(spec, n_values, n_trials=100_000, seed=100 + idx, batch_size=10_000)
        mc_vals = mc_res["accuracy"]
        se = mc_res["se"]
        abs_err = np.abs(exact_vals - mc_vals)
        # 在 100,000 次独立试验下检验无偏性（3.5 sigma 容差，且平均误差 < 2e-3）
        assert float(np.mean(abs_err)) < 2e-3
        assert np.all(abs_err <= np.maximum(1e-3, 3.5 * se))
        assert np.all(mc_res["ci_low"] <= mc_vals)
        assert np.all(mc_vals <= mc_res["ci_high"])

    # 在大样本（n_trials=300,000，SE <= 0.00079）下验证核心场景严格满足绝对误差 < 1e-3
    mc_hi = simulate_bon(specs[1], [16, 256, 1000], n_trials=300_000, seed=2025, batch_size=25_000)
    ex_hi = exact_bon_curve(specs[1], [16, 256, 1000], method="gauss_legendre", n_nodes=256)
    assert float(np.max(np.abs(ex_hi - mc_hi["accuracy"]))) < 1e-3


def test_theorem1_generic_continuous_spec_compatibility() -> None:
    """验证任意实现 ContinuousVerifierSpec 的非高斯分布均可准确积分与仿真。"""
    spec = _ShiftedLogisticSpec(p=0.3, loc1=1.5, scale1=1.0, loc0=0.0, scale0=1.0)
    n_values = [1, 5, 25, 100]
    exact_vals = exact_bon_curve(spec, n_values, method="gauss_legendre", n_nodes=256)
    quad_100 = exact_bon_accuracy(spec, 100, method="quad")
    assert abs(exact_vals[-1] - quad_100) < 1e-8

    mc_res = simulate_bon(spec, n_values, n_trials=40_000, seed=7, batch_size=5_000)
    abs_err = np.abs(exact_vals - mc_res["accuracy"])
    assert np.all(abs_err <= np.maximum(2e-3, 3.5 * mc_res["se"]))


def test_integrals_input_validation() -> None:
    """验证积分模块对非法参数的校验。"""
    spec = GaussianVerifierSpec(p=0.25, mu1=1.0, sigma1=1.0)
    with pytest.raises(ValueError):
        exact_bon_accuracy(spec, 0)
    with pytest.raises(ValueError):
        exact_bon_accuracy(spec, -3)
    with pytest.raises(ValueError):
        exact_bon_accuracy(spec, True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        exact_bon_accuracy(spec, 4, method="unknown")
    with pytest.raises(ValueError):
        exact_bon_accuracy(spec, 4, n_nodes=4)
    assert exact_bon_curve(spec, []).size == 0


def test_theorem2_phase_transition_criterion_and_collapse() -> None:
    """验证定理 2 相界判据（kappa_c = 1）、单调性及异方差/污染尾部极值坍塌行为。"""
    safe_spec = GaussianVerifierSpec(p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=0.8)
    crit_spec = GaussianVerifierSpec(p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.0)
    het_spec = GaussianVerifierSpec(p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.6)
    hack_spec = ContaminatedVerifierSpec(
        p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.0,
        epsilon=0.08, mu_hack=3.5, sigma_hack=1.5,
    )

    c_safe = check_collapse_criterion(safe_spec)
    c_crit = check_collapse_criterion(crit_spec)
    c_het = check_collapse_criterion(het_spec)

    assert not c_safe["will_collapse"]
    assert c_safe["asymptotic_limit"] == pytest.approx(1.0, abs=1e-9)

    assert not c_crit["will_collapse"]
    assert c_crit["is_monotone_increasing"]
    assert c_crit["asymptotic_limit"] == pytest.approx(1.0, abs=1e-9)

    assert c_het["will_collapse"]
    assert not c_het["is_monotone_increasing"]
    assert c_het["asymptotic_limit"] == pytest.approx(0.0, abs=1e-9)
    expected_x_peak = (1.2 * 1.6**2 - 0.0 * 1.0**2) / (1.6**2 - 1.0**2)
    assert gaussian_peak_threshold(het_spec) == pytest.approx(expected_x_peak, rel=1e-10)
    assert c_het["x_peak"] == pytest.approx(expected_x_peak, rel=1e-10)

    # 曲线单调性 vs 过峰坍塌至 0
    ns = [1, 4, 16, 64, 256, 1024, 10_000, 100_000]
    acc_crit = exact_bon_curve(crit_spec, ns)
    assert np.all(np.diff(acc_crit) > 0.0)
    assert acc_crit[-1] > 0.985

    acc_het = exact_bon_curve(het_spec, ns)
    assert np.max(acc_het) > acc_het[0]
    assert acc_het[-1] < 0.05
    assert acc_het[-1] < acc_het[0]

    acc_hack = exact_bon_curve(hack_spec, ns)
    assert np.max(acc_hack) > acc_hack[0]
    assert acc_hack[-1] < 0.02


def test_theorem2_closed_form_vs_numerical_optimal_n() -> None:
    """验证定理 2 闭式最优 N*（saddle_refined 与 tail_quantile）与精确数值搜索高度吻合。"""
    # 不坍塌时 closed_form_optimal_n 返回 n_star=None
    safe_spec = GaussianVerifierSpec(p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.0)
    cf_safe = closed_form_optimal_n(safe_spec, method="saddle_refined")
    assert not cf_safe["will_collapse"]
    assert cf_safe["n_star"] is None

    # 异方差坍塌区典型参数组
    cases = [
        GaussianVerifierSpec(p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.4),
        GaussianVerifierSpec(p=0.25, mu1=1.5, sigma1=1.0, mu0=0.0, sigma0=1.6),
        GaussianVerifierSpec(p=0.15, mu1=1.0, sigma1=1.0, mu0=0.0, sigma0=1.8),
    ]
    for spec in cases:
        num_opt = numerical_optimal_n(spec, n_max=100_000)
        cf_saddle = closed_form_optimal_n(spec, method="saddle_refined")
        cf_tail = closed_form_optimal_n(spec, method="tail_quantile")

        assert cf_saddle["will_collapse"]
        assert cf_tail["will_collapse"]
        n_exact = int(num_opt["n_star"])
        n_saddle = int(cf_saddle["n_star"])
        assert 1 <= n_saddle <= 100_000

        # 验证用闭式 N* 截断取得的准确率非常接近精确数值全局最高准确率
        acc_saddle = exact_bon_accuracy(spec, n_saddle)
        assert num_opt["max_accuracy"] - acc_saddle < 0.02
        assert abs(n_saddle - n_exact) / max(1, n_exact) < 0.45

    # 极端高方差低信噪场景：当 delta_mu <= 0 时（即使 sigma0 > sigma1），P(2) <= P(1)，N*=1 最优
    n1_spec = GaussianVerifierSpec(p=0.5, mu1=-0.05, sigma1=1.0, mu0=0.0, sigma0=3.0)
    num_n1 = numerical_optimal_n(n1_spec, n_max=1000)
    cf_n1 = closed_form_optimal_n(n1_spec, method="saddle_refined")
    assert num_n1["n_star"] == 1
    assert cf_n1["n_star"] == 1

    # 当 delta_mu > 0 时，精确恒等式 P(2) > P(1) 保证最优 N* >= 2
    n2_spec = GaussianVerifierSpec(p=0.5, mu1=0.05, sigma1=1.0, mu0=0.0, sigma0=3.0)
    assert numerical_optimal_n(n2_spec, n_max=1000)["n_star"] >= 2
    assert closed_form_optimal_n(n2_spec, method="saddle_refined")["n_star"] >= 2


def test_theorem3_binary_verifier_formula_and_budget() -> None:
    """验证定理 3 二元 Verifier 精确公式、大规模 MC 吻合、边际增益衰减与预算求解。"""
    spec = BinaryVerifierSpec(p=0.25, tpr=0.85, fpr=0.10)
    n_values = [1, 2, 4, 8, 16, 32, 64, 128]
    exact_vals = np.asarray(binary_bon_accuracy(spec, np.asarray(n_values)), dtype=float)
    mc_100k = simulate_binary_bon(spec, n_values, n_trials=100_000, seed=2024, batch_size=20_000)

    abs_err_100k = np.abs(exact_vals - mc_100k["accuracy"])
    assert float(np.mean(abs_err_100k)) < 1.5e-3
    assert np.all(abs_err_100k <= 3.5 * mc_100k["se"])

    mc_hi = simulate_binary_bon(spec, n_values, n_trials=1_500_000, seed=777, batch_size=100_000)
    assert float(np.max(np.abs(exact_vals - mc_hi["accuracy"]))) < 1e-3

    # 验证单调饱和至 PPV = p * tpr / q1
    q1 = spec.p * spec.tpr + (1.0 - spec.p) * spec.fpr
    expected_ppv = spec.p * spec.tpr / q1
    ceiling = binary_asymptotic_ceiling(spec)
    assert ceiling == pytest.approx(expected_ppv, abs=1e-12)
    assert np.all(np.diff(exact_vals) >= 0.0)
    assert exact_vals[-1] == pytest.approx(ceiling, abs=1e-6)

    # 验证边际增益与解析一阶差分严格相等且以公比 (1 - q1) 指数衰减
    ns = np.arange(1, 30)
    mg = np.asarray(binary_marginal_gain(spec, ns), dtype=float)
    diffs = np.asarray(binary_bon_accuracy(spec, ns + 1), dtype=float) - np.asarray(
        binary_bon_accuracy(spec, ns), dtype=float
    )
    assert np.allclose(mg, diffs, atol=1e-12)
    ratios = mg[1:] / mg[:-1]
    assert np.allclose(ratios, 1.0 - q1, rtol=1e-10)

    # 验证容差与成本最优预算
    n_eps = binary_optimal_budget(spec, eps=0.01)
    assert ceiling - float(binary_bon_accuracy(spec, n_eps)) <= 0.01 + 1e-12
    if n_eps > 1:
        assert ceiling - float(binary_bon_accuracy(spec, n_eps - 1)) > 0.01 - 1e-12

    n_cost = binary_optimal_budget(spec, cost_per_sample=0.005)
    assert n_cost >= 1
    assert float(binary_marginal_gain(spec, n_cost)) <= 0.005 + 1e-12
    if n_cost > 1:
        assert float(binary_marginal_gain(spec, n_cost - 1)) > 0.005 - 1e-12


def test_theorem4_calibrated_bon_eliminates_collapse() -> None:
    """验证定理 4：校准后验 BoN 在异方差与污染尾部下消除坍塌并收敛至 sup_x q(x)。"""
    # 同方差情况下 q(x) 单调，Calibrated-BoN 应与 Standard-BoN 精确一致
    homo = GaussianVerifierSpec(p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.0)
    for n in (1, 10, 100, 1000):
        acc_std = exact_bon_accuracy(homo, n)
        acc_cal = exact_calibrated_bon_accuracy(homo, n, n_grid=4096)
        assert abs(acc_std - acc_cal) < 2e-4

    # 异方差与污染混合尾部：未校准坍塌，校准后单调逼近 sup q(x)
    collapse_specs = [
        GaussianVerifierSpec(p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.5),
        ContaminatedVerifierSpec(
            p=0.25, mu1=1.2, sigma1=1.0, mu0=0.0, sigma0=1.0,
            epsilon=0.08, mu_hack=3.5, sigma_hack=1.5,
        ),
    ]
    for spec in collapse_specs:
        asymp = float(calibrated_bon_asymptotic_limit(spec))
        assert asymp > spec.p
        cal_vals = [exact_calibrated_bon_accuracy(spec, n, n_grid=4096) for n in (1, 10, 100, 1000, 10_000)]
        uncal_10k = exact_bon_accuracy(spec, 10_000)

        assert np.all(np.diff(cal_vals) >= -1e-6)
        assert cal_vals[-1] > 0.90 * asymp
        assert cal_vals[-1] <= asymp + 1e-4
        assert uncal_10k < spec.p
        assert cal_vals[-1] > uncal_10k + 0.3


def test_theorem4_discrete_weighted_majority_convergence() -> None:
    """验证定理 4：在有高概率诱饵错误答案与重污染尾部下，校准加权与两阶段多数决收敛至 1。"""
    hack_spec = ContaminatedVerifierSpec(
        p=0.25, mu1=1.4, sigma1=1.0, mu0=0.0, sigma0=1.5,
        epsilon=0.08, mu_hack=3.5, sigma_hack=1.5,
    )
    # 诱饵错误答案 1 占错题的 45%，先验出现率 0.75 * 0.45 = 0.3375 > p=0.25，足以击垮普通多数决！
    task_spec = DiscreteAnswerTaskSpec(
        verifier_spec=hack_spec,
        num_wrong_answers=5,
        wrong_answer_probs=(0.45, 0.20, 0.15, 0.10, 0.10),
    )

    sim_res = simulate_algorithms_comparison(
        task_spec=task_spec,
        n_values=[1, 16, 128, 512],
        n_trials=4_000,
        seed=99,
        batch_size=1_000,
    )
    algos = sim_res["algorithms"]

    # Standard-BoN 在 N=512 严重坍塌
    assert algos["Standard-BoN"]["accuracy"][-1] < 0.20
    # 普通多数决因诱饵答案先验更高而在大 N 跌向 0
    assert algos["Majority-Voting"]["accuracy"][-1] < 0.20
    # 校准加权与两阶段多数决随 N 增大突破极限，准确率显著高于 0.90
    assert algos["Calibrated-Weighted-BoN"]["accuracy"][-1] > 0.90
    assert algos["Two-Stage-Verified-Majority"]["accuracy"][-1] > 0.90

    # 验证理论 Hoeffding 收敛上界函数返回合法正间隙与正指数率
    bound = weighted_majority_convergence_bound(
        p=0.25, p_wrong_max=0.3375, mu_w_pos=3.5, mu_w_neg=0.8, w_max=20.0
    )
    assert bound["margin"] > 0.0
