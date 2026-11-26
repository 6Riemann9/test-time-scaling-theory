"""V7 审稿反例回归与统计推断严谨性专项单元测试。

显式编码外部审稿报告中的四类数学反例及修订后真实池统计推断特性：
1. P0-01：窄峰反例 q_h(s) = 1/4 + (1/2)exp(-s^2/h^2) 证明缺统一常数约束时
   sup R_h(N) >= 1/2，同时验证 Fano 鼓包族严格落入统一正则类 F_beta^unif 与 all-beta>0 幂界；
2. P0-03：4x4 块对角含负相关正定矩阵给出 eta = 6/23 > 1/4（证非负相关仅充分不必要）；
3. P0-04：3x3 零乘子弱激活矩阵 w = (1, 1, 0)^T（I_+={0,1}, I_0={2}），经独立 2D 数值积分
   交叉检验精确尾率 Theta(t^{-2} e^{-t^2}) 与常因子 1/(4*pi) 及 Copula 对数指数判据；
4. P0-05：三层混合算例给出 kappa_Z ~ 0.3103 < 1 却依然发生 Z* 尾反转 q_Z(+inf) -> 0，
   并与单内点层充分条件 kappa_Z > 1 并列核验；
5. §5.1 & §5.3：验证真实异质性评估采用题目级 Cluster Bootstrap（SE 不随试验重复数虚缩）
   及不相交内部留出 + 配对差异检验。
"""

from __future__ import annotations

import math

import numpy as np
from scipy import integrate, special

from tts_theory import (
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


def test_v7_p0_01_narrow_peak_counterexample_and_uniform_class() -> None:
    """P0-01：复现窄峰反例 sup_{h>0} R_h(N) >= 1/2，并检验统一正则类包含 Fano 族。"""
    n_budget = 10
    regrets = []
    analytic_lbs = []
    for h in (1.0, 0.2, 0.05, 0.005):
        # S ~ N(0, 1)，R = min_{1<=i<=N} |S_i|，f_R(r) = 2 N phi(r) [2 Phi(-r)]^{N-1}
        def integrand(r: float) -> float:
            sf_abs = 2.0 * float(special.ndtr(-r))
            pdf_abs = 2.0 * math.exp(-0.5 * r * r) / math.sqrt(2.0 * math.pi)
            pdf_r = float(n_budget) * pdf_abs * (sf_abs ** (n_budget - 1))
            return math.exp(-(r * r) / (h * h)) * pdf_r

        upper_lim = min(10.0, 12.0 * h + 1.0)
        ex_exp, _ = integrate.quad(integrand, 0.0, upper_lim, epsabs=1e-12, epsrel=1e-10)
        reg = 0.5 * (1.0 - ex_exp)
        lb = 0.5 - (float(n_budget) * h) / (2.0 * math.sqrt(h * h + 2.0))
        regrets.append(reg)
        analytic_lbs.append(lb)
        assert reg >= lb - 1e-10

    # 当 h -> 0 时，固定 N=10 的遗憾单调逼近 0.5，不会以 O(N^{-2}) 一致衰减
    assert regrets[-1] > 0.48

    # 验证一致正则类 F_beta^unif(C_1, f_min=1, eta_0=0.2) 覆盖对任意 beta > 0 构造的 Fano 鼓包族
    c1_bound = 2.0
    eta0 = 0.2
    grid = np.linspace(-eta0, eta0, 401)
    for beta in (0.5, 1.0, 2.0):
        c_beta_amp = c1_bound * 0.5  # <= C_1
        for h in (0.1, 0.02, 0.005):
            # q_j(t_j + d) = 1/2 + c_beta * h^beta * max(0, 1 - |d/h|^beta)
            # 则 q_j^* - q_j(t_j + d) = c_beta * min(|d|^beta, h^beta) <= c_beta * |d|^beta <= C_1 |d|^beta
            psi_val = np.maximum(0.0, 1.0 - (np.abs(grid) / h) ** beta)
            drop = c_beta_amp * (h**beta) * (1.0 - psi_val)
            assert bool(np.all(drop <= c1_bound * (np.abs(grid) ** beta) + 1e-15))

    # 验证 tight_joint_minimax_scaling 对 0 < beta < 1 与 beta >= 1 正确应用 c_beta = max(1, 2^{beta-1})
    b_sub = tight_joint_minimax_scaling(m_total=10_000, n_budget=64, beta=0.5)
    b_sup = tight_joint_minimax_scaling(m_total=10_000, n_budget=64, beta=2.0)
    assert abs(float(b_sub["c_beta_factor"]) - 1.0) < 1e-15
    assert abs(float(b_sup["c_beta_factor"]) - 2.0) < 1e-15


def test_v7_p0_03_block_diagonal_negative_corr_eta_counterexample() -> None:
    """P0-03：4x4 块对角含负相关 SPD 矩阵给出 eta = 6/23 > 1/4（证非负相关仅充分不必要）。"""
    r_block = np.array(
        [
            [1.0, -0.2, 0.0, 0.0],
            [-0.2, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.5, 1.0],
        ],
        dtype=np.float64,
    )
    eigs = np.linalg.eigvalsh(r_block)
    assert float(np.min(eigs)) > 0.4

    qp = solve_positive_quadrant_qp(r_block)
    assert abs(qp["gamma"] - 23.0 / 6.0) < 1e-12
    assert qp["essential_set"] == [0, 1, 2, 3]
    assert qp["strict_complementarity"] is True

    spec = GaussianCopulaHeavyTailSpec(
        p=0.25, alpha1=2.0, alpha0=2.0, corr1=np.eye(4), corr0=r_block
    )
    crit = gaussian_copula_heavy_tail_criterion(spec)
    assert crit["is_nonneg_corr0"] is False
    assert abs(float(crit["eta0_copula"]) - 6.0 / 23.0) < 1e-12
    assert float(crit["eta0_copula"]) > 0.25
    assert crit["eta0_below_independent_floor"] is False


def test_v7_p0_04_degenerate_zero_multiplier_copula_and_gaussian_tail() -> None:
    """P0-04：3x3 零乘子弱激活矩阵精确检出 I_+={0,1}, I_0={2}, 并由独立 2D 数值积分核对常数。"""
    r_deg = np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.5],
            [0.5, 0.5, 1.0],
        ],
        dtype=np.float64,
    )
    qp = solve_positive_quadrant_qp(r_deg)
    assert abs(qp["gamma"] - 2.0) < 1e-12
    assert np.allclose(qp["u_star"], [1.0, 1.0, 1.0], atol=1e-10)
    assert np.allclose(qp["multipliers"], [1.0, 1.0, 0.0], atol=1e-10)
    assert qp["active_set"] == [0, 1, 2]
    assert qp["essential_set"] == [0, 1]
    assert qp["weakly_active_set"] == [2]
    assert qp["strict_complementarity"] is False

    consts = general_min_tail_asymptotic_constants(np.zeros(3), r_deg)
    assert consts["essential_cardinality"] == 2
    assert abs(float(consts["orthant_prob"]) - 0.5) < 1e-12
    assert abs(float(consts["c_sf"]) - 1.0 / (4.0 * math.pi)) < 1e-12
    assert abs(float(consts["c_pdf"]) - 1.0 / (2.0 * math.pi)) < 1e-12

    # 独立精确 2D 积分校验 P(Z_1 > t, Z_2 > t, Z_3 > t)
    # 做标准化变量替换 z1 = t + x1/t, z2 = t + x2/t 并提取 exp(-t^2) 避免大 t 下浮点下溢：
    # exp(t^2) * t^2 * P(Z_1>t, Z_2>t, Z_3>t) -> C_sf = 1/(4*pi)
    inv_2pi = 1.0 / (2.0 * math.pi)
    ratios = []
    for t_val in (20.0, 100.0, 500.0):
        def _scaled_tail_integrand(x2: float, x1: float, t: float = t_val) -> float:
            # dblquad(f, a, b, gfun, hfun) 参数顺序为 (y, x)
            cond_p = float(special.ndtr((x1 + x2) / (t * math.sqrt(2.0))))
            return inv_2pi * math.exp(-(x1 + x2) - 0.5 * ((x1 * x1 + x2 * x2) / (t * t))) * cond_p

        scaled_sf, _ = integrate.dblquad(
            _scaled_tail_integrand, 0.0, 18.0, 0.0, 18.0, epsabs=0.0, epsrel=1e-9
        )
        ratios.append(scaled_sf / float(consts["c_sf"]))

    assert 0.99 <= ratios[-1] <= 1.005
    assert abs(ratios[-1] - 1.0) < abs(ratios[1] - 1.0) < abs(ratios[0] - 1.0)

    # 检验 Copula 对数修正指数与次阶决胜判据：
    # corr1 为正三元等相关 rho=0.25 -> gamma1=2.0, |I_{+,1}|=3 -> log_exp1 = -0.5
    # corr0 为 3x3 退化矩阵 -> gamma0=2.0, |I_{+,0}|=2 -> log_exp0 = 0.0 (> -0.5)
    r_equi = np.full((3, 3), 0.25, dtype=np.float64)
    np.fill_diagonal(r_equi, 1.0)
    spec_cop = GaussianCopulaHeavyTailSpec(
        p=0.25, alpha1=2.0, alpha0=2.0, corr1=r_equi, corr0=r_deg, epsilon_shared=0.0
    )
    cop_crit = gaussian_copula_heavy_tail_criterion(spec_cop)
    assert abs(float(cop_crit["log_factor_exp1"]) - (-0.5)) < 1e-12
    assert abs(float(cop_crit["log_factor_exp0"]) - 0.0) < 1e-12
    assert cop_crit["strict_complementarity0"] is False
    assert cop_crit["regime"] == "min_collapse_copula_log_tiebreak"
    assert cop_crit["min_will_collapse"] is True

    # 检验 GeneralGaussian Min QP tie-break 亦使用本质基数 |I_+| 而非边界基数 |E|
    spec_g = GeneralGaussianMultiVerifierSpec(
        p=0.25, mu1=np.zeros(3), cov1=r_equi, mu0=np.zeros(3), cov0=r_deg
    )
    g_crit = general_min_ensemble_qp_criterion(spec_g)
    assert g_crit["essential_set0"] == [0, 1]
    assert g_crit["regime"] == "min_collapse_log_tiebreak"
    assert g_crit["min_will_collapse"] is True


def test_v7_p0_05_zscore_kappa_lt_1_counterexample_with_tail_inversion() -> None:
    """P0-05：复现审稿人 kappa_Z ~ 0.3103 < 1 反例，并与单内点充分条件对比验证。"""
    z_g = np.linspace(-3.5, 9.0, 1251)
    diag = diagnose_zscore_induced_collapse(
        prompt_weights=[0.01, 0.495, 0.495],
        prompt_p=[0.0, 0.1, 0.2],
        mu1_base=100.0,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
        z_grid=z_g,
    )
    # 精确匹配全方差分解数值与反例结果
    assert abs(float(diag["var_z_pos"]) - 0.22255) < 5e-4
    assert abs(float(diag["var_z_neg"]) - 0.02142) < 5e-4
    assert abs(float(diag["kappa_z"]) - 0.3103) < 2e-3
    assert float(diag["kappa_z"]) < 1.0
    assert float(diag["between_var_z_pos"]) > 0.20

    # 尽管整体 kappa_Z < 1，右尾仍绝对反转趋近 0 且存在内部峰后倒挂（注意 Delta=100 极其分离时
    # 有限点 z_peak=3.0 上可解题条件高斯密度已高度分离，故渐近上限损失极微；尾部反转与整体方差比解耦）
    assert abs(float(diag["z_tail_posterior_limit"]) - 0.0) < 1e-12
    assert float(diag["z_tail_value"]) < 1e-4
    assert diag["tail_turn_down"] is True
    assert float(diag["q_z_peak"]) > 0.999
    assert float(diag["asymptotic_ceiling_loss"]) >= 0.0

    # 非循环充分条件验证：单内点可解层 {0, p_1} 下 Var(m_1 | Y=1) = 0，必严格有 kappa_Z > 1
    for p1 in (0.1, 0.5, 0.9):
        for pi0 in (0.05, 0.5):
            for delta_val in (0.5, 5.0, 50.0):
                d_suff = diagnose_zscore_induced_collapse(
                    prompt_weights=[pi0, 1.0 - pi0],
                    prompt_p=[0.0, p1],
                    mu1_base=delta_val,
                    sigma1_base=1.0,
                    mu0_base=0.0,
                    sigma0_base=1.0,
                    z_grid=z_g,
                )
                assert float(d_suff["between_var_z_pos"]) < 1e-15
                assert float(d_suff["kappa_z"]) > 1.0 + 1e-4


def test_v7_real_eval_disjoint_hybrid_and_cluster_bootstrap() -> None:
    """§5.1 & §5.3：检验题目级 Cluster Bootstrap SE 不随重复采样次数虚缩，并核对配对差异检验。"""
    rng = np.random.default_rng(7701)
    pools = []
    strata_cfg = [("hard", 0.20, -4.5), ("medium", 0.35, 0.0), ("easy", 0.75, 4.5)]
    for i in range(150):
        name, p_g, shift = strata_cfg[i % 3]
        y = rng.random(32) < p_g
        s = shift + np.where(y, rng.normal(1.0, 1.0, 32), rng.normal(0.0, 1.65, 32))
        pools.append((name, s, y))

    res_t5 = evaluate_real_prompt_heterogeneity(
        pools,
        n_values=[1, 2, 8, 32],
        n_folds=5,
        n_trials_per_problem=5,
        n_bins=12,
        seed=301,
        n_bootstrap=300,
    )
    res_t40 = evaluate_real_prompt_heterogeneity(
        pools,
        n_values=[1, 2, 8, 32],
        n_folds=5,
        n_trials_per_problem=40,
        n_bins=12,
        seed=301,
        n_bootstrap=300,
    )
    assert res_t5["se_method"] == "prompt_cluster_bootstrap"
    assert res_t40["se_method"] == "prompt_cluster_bootstrap"
    assert int(res_t40["n_evaluated_problems_by_n"][-1]) == 150

    se_5 = float(res_t5["algorithms"]["Pooled-KL-LCB"]["se"][-1])
    se_40 = float(res_t40["algorithms"]["Pooled-KL-LCB"]["se"][-1])
    # 若错误把 trials 作为独立分母，8 倍 trials 会使 SE 缩小 sqrt(8)=2.83 倍；
    # 题目级 cluster bootstrap 的 SE 应始终保持同一题目数量级（~ 1 / sqrt(150)）
    ratio = se_40 / max(se_5, 1e-12)
    assert 0.60 <= ratio <= 1.50
    assert se_40 > 0.015

    paired = res_t40["paired_bootstrap_vs_pooled"]["Within-Prompt-Z-KL-LCB"]
    assert float(paired["mean_diff"][-1]) > 0.05
    assert float(paired["ci_low"][-1]) > 0.0
    assert float(paired["p_value"][-1]) < 0.05
