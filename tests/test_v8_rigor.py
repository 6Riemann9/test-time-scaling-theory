"""V8 二审数学与统计审稿问题专项回归测试集。

覆盖二审指出的全部数学边界、反例与统计推断修复：
1. V8-P0-01（定理 7B）：同方差单调高斯下 N=2 纯秩界取等号（0.630125），非单调异方差下取严格不等号；
2. V8-P0-02（定理 6C）：beta = 0.25 <= 0.5 时 DKW 相对误差发散失效，而样本分割 Uniform 顺序统计量
   Gamma 间距界保证 1/(2K) <= pi_hat_k < 2/K 且经验分桶 KL-LCB 保持理论收敛阶；
3. V8-P1-01（定理 7A）：显式有限参数二层高斯模型在闭区间 [3.5, 4.2] 上同时满足两层题内导数严格为负且池化导数严格为正；
4. V8-P1-02 & P1-03（定理 2B/2C）：kappa < 1, delta_mu = 0 时 P_BoN(1) = P_BoN(2) = p 随后对 N >= 2 严格递增，
   以及定理 2C 远左尾不等式在 x = a 处成立；
5. V8-P1-04（定理 5B-II）：K=1 与 K>=2 正定相关矩阵下剩余尾相依系数 eta_c 的精确区间分类；
6. V8-P1-05（定理 7C(4)）：在线样本题内标准化 Z_{i,N} 对总体 Z_i^*(X) 的一致极值收敛 O_p(sqrt((ln N)/N)) 及渐近上限损失；
7. V8-P1-06 & P1-07（实证推断）：三分支内部留出门控（含 Standard-BoN）避免在单调域退化，且 Phipson-Smyth
   配对符号翻转置换检验 p 值严格为正（不出现 p = 0）。
"""

from __future__ import annotations

import math

import numpy as np
from scipy import special

from tts_theory import (
    GaussianCopulaHeavyTailSpec,
    GaussianVerifierSpec,
    PromptHeterogeneitySpec,
    calibration_minimax_scaling,
    check_collapse_criterion,
    compute_bin_calibrator_scores,
    diagnose_simpson_confounding,
    diagnose_within_prompt_residual,
    empirical_quantile_spacings_bounds,
    evaluate_real_prompt_heterogeneity,
    exact_bon_curve,
    exact_conditional_bin_bon_accuracy,
    gaussian_copula_heavy_tail_criterion,
    verify_sample_zscore_extremal_convergence,
)


def test_v8_p0_01_theorem7b_monotone_equality_vs_order_reversal_strict() -> None:
    """复现二审 P0-01 反例：同方差单调高斯在 N=2 处三者精确相等（~0.630125），非单调时严格小于 Oracle。"""
    # 反例：S|Y=1 ~ N(1,1), S|Y=0 ~ N(0,1), p = 0.5 => q(s) = sigma(s - 0.5) 严格递增
    spec_mono = PromptHeterogeneitySpec(
        prompt_weights=[1.0],
        prompt_p=[0.5],
        prompt_shift=[0.0],
        mu1_base=1.0,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
    )
    diag_mono = diagnose_within_prompt_residual(spec_mono)
    analytic_n2 = 0.5 + 0.25 * (2.0 * float(special.ndtr(1.0 / math.sqrt(2.0))) - 1.0)
    assert math.isclose(analytic_n2, 0.6301249694532616, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(diag_mono["raw_bon_n2"]), analytic_n2, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(diag_mono["rank_only_n2_ceiling"]), analytic_n2, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(diag_mono["oracle_cal_n2"]), analytic_n2, rel_tol=0.0, abs_tol=1e-6)
    assert float(diag_mono["rank_vs_oracle_n2_gap"]) <= 1e-6
    assert diag_mono["strict_gap_over_rank_n2"] is False

    # 充分必要条件正面检验：异方差 kappa = 2.0 > 1 时存在双向后验序反转，严格不等式成立
    spec_nonmono = PromptHeterogeneitySpec(
        prompt_weights=[1.0],
        prompt_p=[0.5],
        prompt_shift=[0.0],
        mu1_base=0.5,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=2.0,
    )
    diag_nonmono = diagnose_within_prompt_residual(spec_nonmono)
    assert float(diag_nonmono["oracle_cal_n2"]) > float(diag_nonmono["rank_only_n2_ceiling"]) + 0.01
    assert diag_nonmono["strict_gap_over_rank_n2"] is True


def test_v8_p0_02_empirical_quantile_spacings_beta_le_half() -> None:
    """验证二审 P0-02：beta = 0.25 <= 0.5 时 Uniform 顺序统计量间距界有效而 DKW 相对误差界发散。"""
    beta = 0.25
    m_total = 2_500_000
    k_opt = int(round((float(m_total) / math.log(float(m_total))) ** (1.0 / (2.0 * beta + 1.0))))

    sp_bounds = empirical_quantile_spacings_bounds(m_total=m_total, n_bins=k_opt, delta=0.01)
    # beta = 0.25 时 K = (M / ln M)^{2/3}，DKW 相对误差上界 ~ 8.97 > 1（完全空洞），而 Gamma 间距界严格成立
    assert float(sp_bounds["dkw_relative_error_bound"]) > 5.0
    assert sp_bounds["dkw_non_vacuous"] is False
    assert sp_bounds["spacings_condition_holds"] is True
    assert float(sp_bounds["spacings_failure_prob_upper"]) < 0.01

    # 蒙特卡洛直接检验连续分布下顺序统计量边界真实质量 pi_hat_k in [1/(2K), 2/K]
    rng = np.random.default_rng(202608)
    m_mc = 24_000
    k_mc = int(round((float(m_mc) / math.log(float(m_mc))) ** (1.0 / (2.0 * beta + 1.0))))
    m1 = m_mc // 2
    r_idx = np.floor(np.arange(k_mc + 1, dtype=np.float64) * float(m1 + 1) / float(k_mc)).astype(np.int64)
    for _ in range(80):
        u_sorted = np.sort(rng.random(m1))
        edges_u = np.empty(k_mc + 1, dtype=np.float64)
        edges_u[0] = 0.0
        edges_u[-1] = 1.0
        edges_u[1:-1] = u_sorted[r_idx[1:-1] - 1]
        pi_hat = np.diff(edges_u)
        assert float(np.min(pi_hat)) >= 0.5 / float(k_mc)
        assert float(np.max(pi_hat)) < 2.0 / float(k_mc)

    # 验证未知分位数样本分割经验分桶 KL-LCB 在 beta = 0.25 尖峰分布上的遗憾随 M 单调衰减
    # 分布：S ~ Uniform(-1, 1)，q(s) = 0.85 - 0.50 * |s|^{0.25}（唯一峰 s*=0, q*=0.85）
    regrets = []
    n_online = 256
    fine_grid = np.linspace(-1.0, 1.0, 4001)
    for m_val in (600, 4_000, 24_000):
        m_e = m_val // 2
        m_est = m_val - m_e
        k_m = max(4, min(m_e // 30, int(round(1.5 * ((m_val / math.log(m_val)) ** (1.0 / 1.5))))))
        r_m = np.floor(np.arange(k_m + 1, dtype=np.float64) * float(m_e + 1) / float(k_m)).astype(np.int64)
        trial_acc = []
        for _ in range(80):
            s_edge = np.sort(rng.uniform(-1.0, 1.0, size=m_e))
            inner_e = s_edge[r_m[1:-1] - 1]
            full_e = np.concatenate([[-1.0], inner_e, [1.0]])
            pi_true = 0.5 * np.diff(full_e)
            # 精确计算各经验桶上的真实条件正确率 q_k
            q_true = np.empty(k_m, dtype=np.float64)
            for b_i in range(k_m):
                a_l, a_r = float(full_e[b_i]), float(full_e[b_i + 1])
                pts = np.linspace(a_l, a_r, 33)
                qv = 0.85 - 0.50 * (np.abs(pts) ** 0.25)
                q_true[b_i] = float(np.trapezoid(qv, pts) / max(1e-15, a_r - a_l))
            s_est = rng.uniform(-1.0, 1.0, size=m_est)
            y_est = rng.random(m_est) < (0.85 - 0.50 * (np.abs(s_est) ** 0.25))
            b_est = np.searchsorted(inner_e, s_est, side="right")
            cnts = np.bincount(b_est, minlength=k_m).astype(np.float64)
            poss = np.bincount(b_est, weights=y_est.astype(np.float64), minlength=k_m)
            lcb_est = compute_bin_calibrator_scores(
                cnts, poss, method="kl_lcb", prior_p=0.5, delta=1.0 / float(m_val)
            )
            acc_n = float(exact_conditional_bin_bon_accuracy(q_true, pi_true, lcb_est, [n_online])[0])
            trial_acc.append(acc_n)
        regrets.append(0.85 - float(np.mean(trial_acc)))
    _ = fine_grid
    assert regrets[0] > regrets[1] > regrets[2]


def test_v8_p1_01_theorem7a_explicit_two_stratum_simpson_masking() -> None:
    """验证二审 P1-01：显式有限参数二层高斯模型在闭区间 [3.5, 4.2] 上处处发生辛普森假单调遮蔽。"""
    spec = PromptHeterogeneitySpec(
        prompt_weights=[0.85, 0.15],
        prompt_p=[0.05, 0.95],
        prompt_shift=[0.0, 3.0],
        prompt_scale=[1.0, 1.0],
        mu1_base=0.3,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=2.0,
    )
    s_grid = np.linspace(3.5, 4.2, 351)
    st = spec.stratum_stats(s_grid)
    diag = diagnose_simpson_confounding(spec, s_grid=s_grid)

    # 两层各自峰值分别为 0.4 与 3.4，故 [3.5, 4.2] 上两层题内后验导数均严格为负
    assert float(np.max(st["dq_g"][0])) < -1e-3
    assert float(np.max(st["dq_g"][1])) < -1e-3
    assert float(np.max(diag["within_slope_term"])) < -3e-3
    # 后验层权重乘积 w_1(s) * w_2(s) 严格远离 0，跨题正协方差主导使池化导数处处严格为正
    w_prod = st["w_post"][0] * st["w_post"][1]
    assert float(np.min(w_prod)) > 0.20
    assert float(np.min(diag["between_cov_term"])) > 0.03
    assert float(np.min(diag["q_pool_deriv"])) > 0.02
    assert float(diag["decomposition_max_abs_err"]) < 1e-8


def test_v8_p1_02_and_p1_03_gaussian_kappa_lt_1_zero_dmu_and_left_tail() -> None:
    """验证二审 P1-02（kappa < 1, delta_mu = 0 边界）与 P1-03（定理 2C 远左尾不等式）。"""
    spec_tie = GaussianVerifierSpec(p=0.35, mu1=0.0, sigma1=2.0, mu0=0.0, sigma0=1.0)
    crit_tie = check_collapse_criterion(spec_tie)
    assert crit_tie["will_collapse"] is False
    assert crit_tie["is_monotone_increasing"] is False
    assert crit_tie["initial_step_sign"] == 0

    curve_tie = exact_bon_curve(spec_tie, [1, 2, 3, 4, 8, 16])
    # N=1 与 N=2 精确平局等于 p=0.35，随后对一切 N >= 2 严格递增
    assert math.isclose(float(curve_tie[0]), 0.35, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(curve_tie[1]), 0.35, rel_tol=0.0, abs_tol=1e-12)
    assert float(curve_tie[2]) > float(curve_tie[1]) + 0.02
    assert np.all(np.diff(curve_tie[1:]) > 0.0)

    # 对比 delta_mu < 0：真正出现 P_BoN(2) < P_BoN(1) 的 U 型谷底
    spec_u = GaussianVerifierSpec(p=0.35, mu1=-0.6, sigma1=2.0, mu0=0.0, sigma0=1.0)
    crit_u = check_collapse_criterion(spec_u)
    assert crit_u["initial_step_sign"] == -1
    curve_u = exact_bon_curve(spec_u, [1, 2, 16])
    assert float(curve_u[1]) < float(curve_u[0]) - 0.01
    assert float(curve_u[2]) > float(curve_u[0]) + 0.10

    # P1-03：检验 [F_mix(x)]^N <= [F_mix(a)]^{N-1} F_mix(x) 对一切 x <= a（含端点 x = a）恒成立
    spec_col = GaussianVerifierSpec(p=0.25, mu1=1.0, sigma1=1.0, mu0=0.0, sigma0=1.8)
    a_val = 0.5 * float(check_collapse_criterion(spec_col)["x_peak"])
    x_pts = np.linspace(a_val - 4.0, a_val, 200)
    f_x = np.asarray(spec_col.mix_cdf(x_pts), dtype=np.float64)
    f_a = float(spec_col.mix_cdf(a_val))
    for n_val in (2, 5, 20):
        lhs = f_x**n_val
        rhs = (f_a ** (n_val - 1)) * f_x
        assert np.all(lhs <= rhs + 1e-15)
        assert math.isclose(float(lhs[-1]), float(rhs[-1]), rel_tol=1e-13, abs_tol=1e-15)


def test_v8_p1_04_copula_eta_exact_boundary_classification() -> None:
    """验证二审 P1-04：K=1 与 K>=2 正定相关矩阵下 eta_c 的精确四类边界划分。"""
    # 1. K = 1 => eta = 1
    c_k1 = gaussian_copula_heavy_tail_criterion(
        GaussianCopulaHeavyTailSpec(p=0.3, alpha1=4.0, alpha0=2.0, corr1=[[1.0]], corr0=[[1.0]])
    )
    assert math.isclose(float(c_k1["eta0_copula"]), 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert c_k1["eta_strictly_below_one"] is False
    assert c_k1["eta0_boundary_class"] == "single_verifier_eta_eq_1"

    # 2. K = 3 且高相关 rho = 0.99（仍严格正定）=> eta = (1 + 2*0.99)/3 = 0.99333... < 1
    r_high = np.full((3, 3), 0.99, dtype=np.float64)
    np.fill_diagonal(r_high, 1.0)
    c_high = gaussian_copula_heavy_tail_criterion(
        GaussianCopulaHeavyTailSpec(p=0.3, alpha1=4.0, alpha0=2.0, corr1=r_high, corr0=r_high)
    )
    assert math.isclose(float(c_high["eta0_copula"]), (1.0 + 1.98) / 3.0, rel_tol=0.0, abs_tol=1e-10)
    assert float(c_high["eta0_copula"]) < 1.0 - 1e-3
    assert c_high["eta_strictly_below_one"] is True
    assert c_high["eta0_boundary_class"] == "spd_nonneg_interval_[1/K,1)"

    # 3. 含负相关 SPD => 开区间 (0, 1) 且可突破 1/K
    r_neg = np.array([[1.0, -0.4], [-0.4, 1.0]], dtype=np.float64)
    c_neg = gaussian_copula_heavy_tail_criterion(
        GaussianCopulaHeavyTailSpec(p=0.3, alpha1=4.0, alpha0=2.0, corr1=np.eye(2), corr0=r_neg)
    )
    assert math.isclose(float(c_neg["eta0_copula"]), 0.3, rel_tol=0.0, abs_tol=1e-12)
    assert c_neg["eta_strictly_below_one"] is True
    assert c_neg["eta0_boundary_class"] == "spd_signed_open_(0,1)"

    # 4. 同源上尾漏洞 epsilon_shared > 0 => eta_{0,eff} = 1
    c_sh = gaussian_copula_heavy_tail_criterion(
        GaussianCopulaHeavyTailSpec(
            p=0.3, alpha1=4.0, alpha0=2.0, corr1=np.eye(3), corr0=np.eye(3), epsilon_shared=0.05
        )
    )
    assert math.isclose(float(c_sh["eta0_eff"]), 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert c_sh["eta0_boundary_class"] == "shared_tail_eta_eff_eq_1"


def test_v8_p1_05_sample_zscore_uniform_extremal_convergence() -> None:
    """验证二审第六节核心加强（定理 7C(4)）：在线样本 Z_{i,N} 一致极值收敛与渐近上限损失。"""
    res = verify_sample_zscore_extremal_convergence(
        prompt_weights=[0.55, 0.45],
        prompt_p=[0.0, 0.40],
        n_values=[16, 64, 256, 1024],
        mu1_base=1.1,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
        n_eval_prompts=2_500,
        seed=808,
    )
    sup_err = res["mean_sup_abs_z_error"]
    scaled_ratio = res["scaled_sup_error_over_sqrt_lnN_N"]
    # 极值误差随 N 严格单调衰减趋向 0，且除以 sqrt((ln N)/N) 后保持有界 O(1)
    assert np.all(np.diff(sup_err) < 0.0)
    assert float(sup_err[-1]) < 0.20
    assert np.all((scaled_ratio > 0.5) & (scaled_ratio < 4.0))

    # 在线样本 Z_{i,N} Oracle 准确率收敛至总体 Z* Oracle 上限，并严格低于 Raw-BoN 渐近上限 1 - pi_0 = 0.45
    sample_z_final = float(res["sample_z_oracle_acc"][-1])
    theo_z_ceil = float(res["theoretical_z_oracle_ceiling"])
    theo_raw_ceil = float(res["theoretical_raw_ceiling"])
    theo_loss = float(res["theoretical_ceiling_loss"])
    assert theo_loss > 0.02
    assert abs(sample_z_final - theo_z_ceil) < 0.010
    assert sample_z_final < theo_raw_ceil - 0.015


def test_v8_p1_06_and_p1_07_three_branch_hybrid_and_phipson_smyth_pvalue() -> None:
    """验证三分支门控包含 Standard-BoN 防止单调域退化，且置换检验 p 值严格为正。"""
    rng = np.random.default_rng(909)
    # 构造同方差严格单调打分 + 大量 p=0 不可解题：
    # - Pooled-KL-LCB 因有限 6 桶离散化同桶随机平局略逊于连续 Standard-BoN；
    # - Within-Prompt-Z-KL-LCB 因定理 7C 尾部反转落后于 Standard-BoN；
    # 三分支 Hybrid-Adaptive 应在内部验证集上选中 Standard-BoN，不发生退化。
    mono_pools = []
    for i in range(250):
        st = "hard_zero" if i < 125 else "solvable"
        p_g = 0.0 if i < 125 else 0.55
        y = rng.random(32) < p_g
        s = np.where(y, rng.normal(1.6, 1.0, 32), rng.normal(0.0, 1.0, 32))
        mono_pools.append((st, s, y))

    res = evaluate_real_prompt_heterogeneity(
        mono_pools,
        n_values=[1, 2, 8, 32],
        n_folds=5,
        n_trials_per_problem=15,
        n_bins=6,
        seed=910,
        n_bootstrap=400,
        n_permutations=2_000,
    )
    algs = res["algorithms"]
    std_32 = float(algs["Standard-BoN"]["accuracy"][-1])
    pool_32 = float(algs["Pooled-KL-LCB"]["accuracy"][-1])
    z_32 = float(algs["Within-Prompt-Z-KL-LCB"]["accuracy"][-1])
    hyb_32 = float(algs["Hybrid-Adaptive-KL-LCB"]["accuracy"][-1])

    assert std_32 > pool_32 + 0.005
    assert std_32 > z_32 + 0.01
    assert res["hybrid_branch_selection_counts"]["32"]["Standard-BoN"] >= 4
    assert hyb_32 >= pool_32 + 0.005

    # 验证 Phipson-Smyth 符号翻转置换 p 值严格位于 [1 / (B_perm + 1), 1.0]，绝不等于 0.0
    min_possible_p = 1.0 / float(res["n_permutations"] + 1)
    for paired_key in ("paired_bootstrap_vs_pooled", "paired_bootstrap_vs_standard_bon"):
        for comp_stats in res[paired_key].values():
            p_arr = np.asarray(comp_stats["p_value"], dtype=np.float64)
            assert np.all(p_arr >= min_possible_p - 1e-15)
            assert np.all(p_arr <= 1.0 + 1e-15)
            assert np.all(p_arr > 0.0)
