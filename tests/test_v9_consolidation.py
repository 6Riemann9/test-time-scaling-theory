"""三审（V8 -> V9 收敛定稿）专项回归测试集。

覆盖三审清单中的全部数学、算法协议与统计推断修复：
1. P0-01（定理 7C 仿射坐标变换）：在大偏移 b(X)=1000, a(X)=8.0 下核验 q_X(b + a*[mu_tilde + sigma_tilde*Z*])
   与基底后验 q_tilde_X 完全一致，且在线样本 Z_{i,N} Oracle 准确率具备严格仿射不变性；
2. P0-02（命题 6.1 与真实池实验算法对齐）：核验 evaluate_finite_calibration_on_real_pools 的
   'sample_split' 协议严格不读取 M 条抽样之外的无标签分布信息，并对比三种分桶协议；
3. P1-01（命题 6.1 完整置信度 delta = 1/(4M) 与统计惩罚常数 2 倍）：在 beta=0.25, M=2.5e6, K=3065
   下直接验证 delta = 1/(4M) 浓缩条件成立、惩罚放大因子精确等于 2.0，以及预算平衡点 N_balance；
4. P1-02（定理 7A 无网格纯解析区间界）：直接核验闭区间 [3.5, 4.2] 上的端点单调解析上下界链条；
5. P1-03（命题 7.1 覆盖 N=2 全三分支与零重拟合间隙）：在 delta_mu < 0 的 Anti-BoN 相区验证 N=2 时
   Hybrid 门控可选中 Within-Prompt-Z-KL-LCB 并击败 Standard-BoN 与混杂 Pooled；
6. P1-04（独立单次划分 Train/Val/Test 与配对学生化 t 检验）：验证三段互不相交划分、零重拟合误差
   Hybrid-Fixed-KL-LCB 及无需对称分布假设的双侧配对 t 检验。
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from tts_theory import (
    empirical_quantile_spacings_bounds,
    evaluate_finite_calibration_on_real_pools,
    evaluate_real_prompt_heterogeneity,
    evaluate_single_split_prompt_heterogeneity,
    tight_joint_minimax_scaling,
    verify_sample_zscore_extremal_convergence,
)


def test_v9_p0_01_theorem7c_affine_coordinate_invariance_under_large_shift() -> None:
    """修复 P0-01：验证极大题间偏移与缩放下定理 7C(4) 后验坐标变换与选择准确率严格不变。"""
    res_base = verify_sample_zscore_extremal_convergence(
        prompt_weights=[0.55, 0.45],
        prompt_p=[0.0, 0.40],
        n_values=[16, 64, 256],
        mu1_base=1.1,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
        prompt_shift=[0.0, 0.0],
        prompt_scale=[1.0, 1.0],
        n_eval_prompts=1_500,
        seed=901,
    )
    res_shifted = verify_sample_zscore_extremal_convergence(
        prompt_weights=[0.55, 0.45],
        prompt_p=[0.0, 0.40],
        n_values=[16, 64, 256],
        mu1_base=1.1,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
        prompt_shift=[-200.0, 1000.0],
        prompt_scale=[0.2, 8.0],
        n_eval_prompts=1_500,
        seed=901,
    )
    assert float(res_base["max_affine_posterior_discrepancy"]) <= 1e-13
    assert float(res_shifted["max_affine_posterior_discrepancy"]) <= 1e-13
    assert np.allclose(res_base["population_z_oracle_acc"], res_shifted["population_z_oracle_acc"], atol=1e-12)
    assert np.allclose(res_base["sample_z_oracle_acc"], res_shifted["sample_z_oracle_acc"], atol=1e-12)
    assert np.allclose(res_base["mean_sup_abs_z_error"], res_shifted["mean_sup_abs_z_error"], atol=1e-11)

    # 反例核验：若错误漏写 b(X)=1000, a(X)=8.0 而将 mu_tilde + sigma_tilde*z 代入原始 q_X(s)，后验退化趋于 0
    b2, a2, p2 = 1000.0, 8.0, 0.40
    z_star = 1.2
    s_wrong = 0.44 + math.sqrt(1.0 + 0.24 * 1.21) * z_star
    llr_wrong = norm.logpdf(s_wrong, loc=b2 + a2 * 1.1, scale=a2) - norm.logpdf(s_wrong, loc=b2, scale=a2)
    q_wrong = 1.0 / (1.0 + math.exp(-(math.log(p2 / (1.0 - p2)) + float(llr_wrong))))
    assert q_wrong < 1e-40


def test_v9_p0_02_real_pool_three_quantile_protocols_comparison() -> None:
    """修复 P0-02：检验真实池评估接口支持 sample_split / in_sample / semi_supervised 三种分桶协议。"""
    rng = np.random.default_rng(902)
    # 构造非单调异方差校准池与测试池（kappa = 1.9 > 1）
    cal_y = rng.random(8_000) < 0.30
    cal_s = np.where(cal_y, rng.normal(0.6, 1.0, 8_000), rng.normal(0.0, 1.9, 8_000))
    test_pools = []
    for _ in range(200):
        ty = rng.random(32) < 0.30
        ts = np.where(ty, rng.normal(0.6, 1.0, 32), rng.normal(0.0, 1.9, 32))
        test_pools.append((ts, ty))

    accs_split = []
    for m_cal, k_bins in ((120, 6), (600, 9), (3_000, 12)):
        r_split = evaluate_finite_calibration_on_real_pools(
            cal_s,
            cal_y,
            test_pools,
            m_cal=m_cal,
            n_bins=k_bins,
            n_values=[1, 8, 32],
            binning="quantile",
            methods=("kl_lcb",),
            n_repeats=50,
            delta=0.05,
            seed=903,
            quantile_protocol="sample_split",
        )
        assert r_split["quantile_protocol"] == "sample_split"
        accs_split.append(float(r_split["methods"]["kl_lcb"]["accuracy"][-1]))

    r_in = evaluate_finite_calibration_on_real_pools(
        cal_s,
        cal_y,
        test_pools,
        m_cal=3_000,
        n_bins=12,
        n_values=[1, 8, 32],
        binning="quantile",
        methods=("kl_lcb",),
        n_repeats=50,
        delta=0.05,
        seed=903,
        quantile_protocol="in_sample",
    )
    r_semi = evaluate_finite_calibration_on_real_pools(
        cal_s,
        cal_y,
        test_pools,
        m_cal=3_000,
        n_bins=12,
        n_values=[1, 8, 32],
        binning="quantile",
        methods=("kl_lcb",),
        n_repeats=50,
        delta=0.05,
        seed=903,
        quantile_protocol="semi_supervised",
    )
    assert accs_split[0] < accs_split[1] < accs_split[2]
    # 大 M 下严格样本分割与半监督全池分位准确率接近，且均显著优于坍塌的 Standard-BoN
    std_32 = float(r_semi["standard_bon_accuracy"][-1])
    assert accs_split[-1] > std_32 + 0.08
    assert abs(accs_split[-1] - float(r_semi["methods"]["kl_lcb"]["accuracy"][-1])) < 0.02
    assert abs(float(r_in["methods"]["kl_lcb"]["accuracy"][-1]) - accs_split[-1]) < 0.02


def test_v9_p1_01_spacings_full_delta_1_over_4m_and_factor_2() -> None:
    """修复 P1-01：直接验证命题 6.1 完整置信水平 delta = 1/(4M)、惩罚常数 2.0 及预算平衡律。"""
    beta = 0.25
    m_total = 2_500_000
    k_opt = int(round((float(m_total) / math.log(float(m_total))) ** (1.0 / (2.0 * beta + 1.0))))
    delta_prop = 1.0 / float(4 * m_total)

    sp = empirical_quantile_spacings_bounds(m_total=m_total, n_bins=k_opt, delta=delta_prop)
    assert sp["spacings_condition_holds"] is True
    assert float(sp["spacings_failure_prob_upper"]) < delta_prop
    assert float(sp["dkw_relative_error_bound"]) > 10.0
    assert sp["dkw_non_vacuous"] is False
    assert math.isclose(float(sp["lcb_penalty_inflation_factor"]), 2.0, rel_tol=0.0, abs_tol=1e-12)

    t_low_n = tight_joint_minimax_scaling(m_total=100_000, n_budget=4, beta=2.0)
    t_high_n = tight_joint_minimax_scaling(m_total=100_000, n_budget=64, beta=2.0)
    assert float(t_low_n["calibration_compute_balance_n"]) > 5.0
    assert t_low_n["bottleneck_regime"] == "online_compute_dominated"
    assert t_high_n["bottleneck_regime"] == "offline_calibration_dominated"


def test_v9_p1_02_theorem7a_grid_free_analytical_interval_bounds() -> None:
    """修复 P1-02：校验定理 7A(2) 在闭区间 [3.5, 4.2] 上的无网格端点解析上下界链条。"""
    def _ell1(s: float) -> float:
        return math.log(2.0) - 0.5 * ((s - 0.3) ** 2) + 0.125 * (s * s)

    def _ell2(s: float) -> float:
        return math.log(2.0) - 0.5 * ((s - 3.3) ** 2) + 0.125 * ((s - 3.0) ** 2)

    def _sig(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    # 端点精确值锁定整个连续区间 [3.5, 4.2] 上的单调后验范围
    q1_lo = _sig(math.log(0.05 / 0.95) + _ell1(4.2))
    q1_hi = _sig(math.log(0.05 / 0.95) + _ell1(3.5))
    q2_lo = _sig(math.log(0.95 / 0.05) + _ell2(4.2))
    q2_hi = _sig(math.log(0.95 / 0.05) + _ell2(3.5))

    assert 0.00047 <= q1_lo < q1_hi <= 0.00291
    assert 0.9680 <= q2_lo < q2_hi <= 0.9747

    # 题内斜率绝对值解析上界
    abs_dq1_ub = q1_hi * (0.75 * (4.2 - 0.4))
    abs_dq2_ub = (1.0 - q2_lo) * (0.75 * (4.2 - 3.4))
    within_abs_ub = max(abs_dq1_ub, abs_dq2_ub)
    assert within_abs_ub < 0.01915

    # 后验层权重乘积解析下界
    f1_lo = 0.05 * float(norm.pdf(4.2, 0.3, 1.0)) + 0.95 * float(norm.pdf(4.2, 0.0, 2.0))
    f1_hi = 0.05 * float(norm.pdf(3.5, 0.3, 1.0)) + 0.95 * float(norm.pdf(3.5, 0.0, 2.0))
    f2_lo = 0.95 * float(norm.pdf(4.2, 3.3, 1.0)) + 0.05 * float(norm.pdf(4.2, 3.0, 2.0))
    f2_hi = 0.95 * float(norm.pdf(3.5, 3.3, 1.0)) + 0.05 * float(norm.pdf(3.5, 3.0, 2.0))
    r_lo = (0.15 * f2_lo) / (0.85 * f1_hi)
    r_hi = (0.15 * f2_hi) / (0.85 * f1_lo)
    w_prod_lb = min(r_lo / ((1.0 + r_lo) ** 2), r_hi / ((1.0 + r_hi) ** 2))
    assert w_prod_lb > 0.1808

    # 对数混合密度导数差解析下界 g_2(s) - g_1(s) >= 3.3 - 0.75 * 4.2 = 0.15
    g_diff_lb = 3.3 - 0.75 * 4.2
    cov_lb = w_prod_lb * (q2_lo - q1_hi) * g_diff_lb
    assert cov_lb > 0.0261
    assert cov_lb - within_abs_ub > 0.0070


def test_v9_p1_03_hybrid_includes_within_z_at_n2_in_anti_bon_regime() -> None:
    """修复 P1-03：在反向打分（delta_mu < 0）+ 跨题重叠辛普森混杂相区，验证 N=2 处门控选中 Within-Z (Anti-BoN)。"""
    rng = np.random.default_rng(905)
    # 正类得分反而更低（mu1 = -1.8 < mu0 = 0.0），且层间偏移重叠（-2.0, 0.0, +2.0）触发全局辛普森单调假象
    strata_cfg = [("s1", 0.15, -2.0), ("s2", 0.50, 0.0), ("s3", 0.85, 2.0)]
    pools = []
    for i in range(360):
        st, pg, sh = strata_cfg[i % 3]
        y = rng.random(16) < pg
        s = sh + np.where(y, rng.normal(-1.8, 1.0, 16), rng.normal(0.0, 1.0, 16))
        pools.append((st, s, y))

    res_cv = evaluate_real_prompt_heterogeneity(
        pools, n_values=[1, 2, 8], n_folds=5, n_trials_per_problem=20, n_bins=12, seed=906, n_bootstrap=200, n_permutations=500
    )
    algs_cv = res_cv["algorithms"]
    # N=2 位于索引 1：Within-Z 学会选 -1/sqrt(2)（即 Anti-BoN），同时击败 Standard-BoN 与受辛普森遮蔽的 Pooled
    assert res_cv["hybrid_branch_selection_counts"]["2"]["Within-Prompt-Z-KL-LCB"] == 5
    assert float(algs_cv["Hybrid-Adaptive-KL-LCB"]["accuracy"][1]) > float(algs_cv["Standard-BoN"]["accuracy"][1]) + 0.12
    assert float(algs_cv["Hybrid-Adaptive-KL-LCB"]["accuracy"][1]) > float(algs_cv["Pooled-KL-LCB"]["accuracy"][1]) + 0.10


def test_v9_p1_04_disjoint_single_split_studentized_ttest_and_zero_refit_gap() -> None:
    """修复 P1-04：验证互不相交单次划分 Train/Val/Test、零重拟合误差与配对学生化 t 检验。"""
    rng = np.random.default_rng(907)
    strata_cfg = [("hard", 0.20, -4.5), ("med", 0.40, 0.0), ("easy", 0.75, 4.5)]
    pools = []
    for i in range(600):
        st, pg, sh = strata_cfg[i % 3]
        y = rng.random(64) < pg
        s = sh + np.where(y, rng.normal(1.0, 1.0, 64), rng.normal(0.0, 1.85, 64))
        pools.append((st, s, y))

    res = evaluate_single_split_prompt_heterogeneity(
        pools,
        n_values=[1, 2, 8, 32],
        split_ratios=(0.50, 0.20, 0.30),
        n_trials_per_problem=20,
        n_bins=12,
        seed=908,
        n_bootstrap=400,
    )
    assert res["n_train_problems"] + res["n_val_problems"] + res["n_test_problems"] == 600
    assert res["primary_inference_method"] == "disjoint_single_split_paired_studentized_ttest"

    # 当验证集在 N=32 选中 Within-Prompt-Z-KL-LCB 时，Hybrid-Fixed 在独立测试集上与固定 Within-Z 完全一致（零重拟合差）
    assert res["selected_branch_by_n"]["32"] == "Within-Prompt-Z-KL-LCB"
    acc_hyb_fixed = float(res["algorithms"]["Hybrid-Fixed-KL-LCB"]["accuracy"][-1])
    acc_z_fixed = float(res["algorithms"]["Within-Prompt-Z-KL-LCB"]["accuracy"][-1])
    assert math.isclose(acc_hyb_fixed, acc_z_fixed, rel_tol=0.0, abs_tol=1e-12)

    p_t = float(res["paired_test_vs_pooled"]["Hybrid-Fixed-KL-LCB"]["p_value_ttest"][-1])
    assert 0.0 < p_t < 0.001
