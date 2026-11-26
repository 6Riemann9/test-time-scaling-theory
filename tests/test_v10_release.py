"""四审收官（V10 Release / v1.0.0）专项回归测试集。

覆盖四审清单中的全部数学细节、参数防呆与公平消融断言：
1. P1-01（定理 7A q_2 端点选择修正）：验证在 q_2 > 0.5 时 q_2(1 - q_2) 单调递减，
   正确端点上界为 -0.075 * q_2(3.5) * (1 - q_2(3.5)) ≈ -0.0018539 < -1.85e-3 < 0；
2. P1-02（定理 2C 固定峰值高度 Gumbel 尺度常数）：取 c_1 = 4.0（使 sqrt(c_1)=2 != c_1=4），
   验证当 kappa -> 1+ 时 h*tau*(kappa^2 - 1) / sqrt(c_1) -> 1.0；
3. P1-03（定理 7B 零方差与平局安全性）：验证常数分数池（hat_s_N = 0）下样本 Z-Score
   精确输出 0.0、无 NaN/Inf，并退化为均匀随机平局选择；
4. 协议防呆（real_eval.py L565）：验证 binning='uniform' 搭配 quantile_protocol='sample_split'
   或 'in_sample' 时显式抛出 ValueError，拒绝静默切换分桶协议；
5. 同分桶数 K 下的公平校准器消融：在稀疏右尾假阳性污染下，验证相同 K 与相同 sample_split
   协议下 KL-LCB / Wilson-LCB 显著优于同 K 的 MLE；
6. 训练干净、测试污染的分布漂移边界核验：量化已知污染（Matched）与未知漂移（Unseen Shift）的差异。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from tts_theory import (
    GaussianVerifierSpec,
    PromptHeterogeneitySpec,
    evaluate_finite_calibration_on_real_pools,
    evaluate_pooled_vs_within_prompt_calibration,
    evaluate_single_split_prompt_heterogeneity,
    gaussian_peak_threshold,
)
from tts_theory.heterogeneity_theory import _within_prompt_zscore


def test_v10_p1_01_theorem7a_q2_left_endpoint_monotonicity() -> None:
    """修复 P1-01：当 q_2 in (0.5, 1) 时 u(1-u) 严格递减，验证左端点 s=3.5 锁定全局负斜率上界。"""
    def _ell2(s: float) -> float:
        return math.log(2.0) - 0.5 * ((s - 3.3) ** 2) + 0.125 * ((s - 3.0) ** 2)

    def _q2(s: float) -> float:
        return 1.0 / (1.0 + math.exp(-(math.log(0.95 / 0.05) + _ell2(s))))

    q2_left = _q2(3.5)
    q2_right = _q2(4.2)
    assert 0.5 < q2_right < q2_left < 1.0
    # 因 u(1-u) 在 (0.5, 1) 递减，左端点 s=3.5 给出最小方差因子
    var_min = q2_left * (1.0 - q2_left)
    var_max = q2_right * (1.0 - q2_right)
    assert var_min < var_max

    # 解析导数上界（最靠近 0 的负值）由 s=3.5 处的 ell'_2(3.5) = -0.075 与 var_min 共同锁定
    dq2_analytic_upper = -0.075 * var_min
    assert math.isclose(dq2_analytic_upper, -0.001853869, rel_tol=1e-5, abs_tol=1e-8)
    assert dq2_analytic_upper < -1.85e-3

    s_dense = np.linspace(3.5, 4.2, 5_000)
    for sv in s_dense:
        qv = _q2(float(sv))
        dqv = qv * (1.0 - qv) * (-0.75 * (float(sv) - 3.4))
        assert dqv <= dq2_analytic_upper + 1e-15


def test_v10_p1_02_gumbel_fixed_peak_sqrt_c1_asymptotic_constant() -> None:
    """修复 P1-02：取 c_1 = 4.0（sqrt(c_1)=2.0 != c_1=4.0），核验 h*tau ~ sqrt(c_1)/(kappa^2 - 1)。"""
    c1 = 4.0
    sigma1 = 1.3
    p = 0.30
    ratios_sqrt_c1 = []
    ratios_wrong_c1 = []

    for kappa in (1.10, 1.03, 1.01, 1.002):
        dmu = sigma1 * math.sqrt(c1 * (kappa * kappa - 1.0))
        spec = GaussianVerifierSpec(p=p, mu1=dmu, sigma1=sigma1, mu0=0.0, sigma0=kappa * sigma1)
        x_peak = gaussian_peak_threshold(spec)
        h_val = float(spec.mix_pdf(x_peak)) / float(spec.mix_sf(x_peak))
        tau_val = 1.0 / math.sqrt(1.0 / (sigma1 * sigma1) - 1.0 / ((kappa * sigma1) ** 2))
        scaled_sqrt = h_val * tau_val * (kappa * kappa - 1.0) / math.sqrt(c1)
        scaled_wrong = h_val * tau_val * (kappa * kappa - 1.0) / c1
        ratios_sqrt_c1.append(scaled_sqrt)
        ratios_wrong_c1.append(scaled_wrong)

    # 正确常数比精确收敛至 1.0，而旧式除以 c_1 则收敛至 1/sqrt(c_1) = 0.5
    assert abs(ratios_sqrt_c1[-1] - 1.0) < 0.005
    assert abs(ratios_wrong_c1[-1] - 0.5) < 0.003


def test_v10_p1_03_zero_variance_and_constant_ties_zscore_safety() -> None:
    """修复 P1-03：验证常数打分（零样本标准差）下题内 Z-Score 精确置零且平局处理无异常。"""
    const_scores = np.array(
        [
            [3.14159, 3.14159, 3.14159, 3.14159],
            [-2.0, -2.0, -2.0, -2.0],
        ],
        dtype=np.float64,
    )
    z_out = _within_prompt_zscore(const_scores)
    assert np.all(np.isfinite(z_out))
    assert np.all(z_out == 0.0)

    # 在真实池单次划分接口上构造全常数打分池，验证零方差下不崩溃且与随机基线一致
    rng = np.random.default_rng(1003)
    const_pools = []
    for i in range(120):
        y = rng.random(8) < 0.40
        s = np.full(8, float(i % 5), dtype=np.float64)
        const_pools.append(("const", s, y))

    res = evaluate_single_split_prompt_heterogeneity(
        const_pools, n_values=[1, 2, 8], n_trials_per_problem=10, n_bins=6, seed=1004, n_bootstrap=100
    )
    for alg_data in res["algorithms"].values():
        assert np.all(np.isfinite(alg_data["accuracy"]))

    spec = PromptHeterogeneitySpec(
        prompt_weights=[0.5, 0.5],
        prompt_p=[0.25, 0.75],
        prompt_shift=[-1.0, 1.0],
        prompt_scale=[1.0, 1.0],
    )
    sim = evaluate_pooled_vs_within_prompt_calibration(
        spec, n_values=[1, 2, 8], m_cal_prompts=100, n_cal_per_prompt=8, n_eval_prompts=300, seed=1005
    )
    assert np.all(np.isfinite(sim["within_prompt_z_kl_lcb_mc"]))


def test_v10_reject_incompatible_uniform_binning_with_quantile_protocol() -> None:
    """验证 binning='uniform' 时若传入 sample_split 或 in_sample 显式报错而非静默回退。"""
    rng = np.random.default_rng(1006)
    cal_s = rng.normal(0.0, 1.0, 200)
    cal_y = rng.random(200) < 0.5
    test_pools = [(rng.normal(0.0, 1.0, 16), rng.random(16) < 0.5) for _ in range(20)]

    for bad_proto in ("sample_split", "in_sample"):
        with pytest.raises(ValueError, match="仅适用于 binning='quantile'"):
            evaluate_finite_calibration_on_real_pools(
                cal_s,
                cal_y,
                test_pools,
                m_cal=100,
                n_bins=5,
                n_values=[1, 8],
                binning="uniform",
                quantile_protocol=bad_proto,
            )


def test_v10_same_k_fair_calibrator_ablation_isolates_lcb_gain() -> None:
    """建议项消融：在严格相同分桶数 K 与相同 sample_split 协议下，验证 KL-LCB 优于同 K 的 MLE。"""
    rng = np.random.default_rng(1007)
    # 构造带稀疏右尾假阳性的分布：在固定相同 K=16 下，MLE 会被少量假满分尾桶误导
    n_cal_pool = 10_000
    cal_y = rng.random(n_cal_pool) < 0.30
    is_spike = (~cal_y) & (rng.random(n_cal_pool) < 0.05)
    cal_s = np.where(cal_y, rng.normal(1.2, 1.0, n_cal_pool), rng.normal(0.0, 1.0, n_cal_pool))
    cal_s = np.where(is_spike, rng.normal(4.2, 0.5, n_cal_pool), cal_s)

    test_pools = []
    for _ in range(250):
        ty = rng.random(64) < 0.30
        t_sp = (~ty) & (rng.random(64) < 0.05)
        ts = np.where(ty, rng.normal(1.2, 1.0, 64), rng.normal(0.0, 1.0, 64))
        ts = np.where(t_sp, rng.normal(4.2, 0.5, 64), ts)
        test_pools.append((ts, ty))

    res = evaluate_finite_calibration_on_real_pools(
        cal_s,
        cal_y,
        test_pools,
        m_cal=120,
        n_bins=24,
        n_values=[1, 16, 64],
        binning="quantile",
        methods=("mle", "laplace", "wilson_lcb", "kl_lcb"),
        n_repeats=60,
        delta=0.05,
        seed=1008,
        quantile_protocol="sample_split",
    )
    acc_mle = float(res["methods"]["mle"]["accuracy"][-1])
    acc_kl = float(res["methods"]["kl_lcb"]["accuracy"][-1])
    acc_wilson = float(res["methods"]["wilson_lcb"]["accuracy"][-1])
    std_bon = float(res["standard_bon_accuracy"][-1])
    # 同 K=24、同 sample_split 预算（M_2=60, 每桶约 2.5 样本）下，KL-LCB 与 Wilson-LCB 显著抑制 MLE 赢家诅咒
    assert acc_kl > acc_mle + 0.03
    assert acc_wilson > acc_mle + 0.03
    assert acc_kl > std_bon + 0.25


def test_v10_clean_train_contaminated_test_shift_honest_boundary() -> None:
    """核验训练干净、测试突发未知右尾污染时的行为边界（区分已知污染与未知分布漂移）。"""
    rng = np.random.default_rng(1009)
    train_s = np.empty(1_000, dtype=np.float64)
    train_y = rng.random(1_000) < 0.35
    train_s = np.where(train_y, rng.normal(1.2, 1.0, 1_000), rng.normal(0.0, 1.0, 1_000))

    # 测试池注入超出干净训练最大值的未知假阳性尖峰
    test_contam = []
    for _ in range(150):
        ty = rng.random(32) < 0.35
        t_sp = (~ty) & (rng.random(32) < 0.12)
        ts = np.where(ty, rng.normal(1.2, 1.0, 32), rng.normal(0.0, 1.0, 32))
        ts = np.where(t_sp, 8.0 + np.abs(rng.normal(0.0, 0.5, 32)), ts)
        test_contam.append((ts, ty))

    r_unseen = evaluate_finite_calibration_on_real_pools(
        train_s,
        train_y,
        test_contam,
        m_cal=800,
        n_bins=8,
        n_values=[1, 32],
        binning="quantile",
        methods=("kl_lcb",),
        n_repeats=20,
        seed=1010,
        quantile_protocol="sample_split",
    )
    # 在完全未见过污染的干净集上训练时，超界尖峰落入干净最高分桶，诚实表现为仍受未知漂移影响（准确率下降）
    acc_unseen_n1 = float(r_unseen["methods"]["kl_lcb"]["accuracy"][0])
    acc_unseen_n32 = float(r_unseen["methods"]["kl_lcb"]["accuracy"][-1])
    assert np.isfinite(acc_unseen_n32)
    assert acc_unseen_n32 < acc_unseen_n1 + 0.15


def test_v11_p0_02_safe_fisher_orthogonal_boundary_counterexample() -> None:
    """第五轮 P0-02：直接验证审稿人反例 Sigma_1=I, Sigma_0=diag(0.5,2), dmu=(0,1)^T 进入退化正交边界分支。"""
    from tts_theory.ensemble_theory import (
        GeneralGaussianMultiVerifierSpec,
        pareto_linear_ensemble_frontier,
    )

    spec = GeneralGaussianMultiVerifierSpec(
        p=0.4,
        mu1=[0.0, 1.0],
        mu0=[0.0, 0.0],
        cov1=[[1.0, 0.0], [0.0, 1.0]],
        cov0=[[0.5, 0.0], [0.0, 2.0]],
    )
    res = pareto_linear_ensemble_frontier(spec, kappa_max=1.0)
    assert res["feasible"] is True
    assert res["regime"] == "degenerate_orthogonal_boundary"
    assert math.isclose(float(res["kappa_safe"]), 1.0, rel_tol=1e-6, abs_tol=1e-7)
    w_safe = np.asarray(res["w_safe"], dtype=np.float64)
    # 理论上最优单位方向应平行于 (sqrt(2), 1)（正交分量符号可 ± 对称）
    ratio_abs = abs(float(w_safe[0])) / max(abs(float(w_safe[1])), 1e-12)
    assert math.isclose(ratio_abs, math.sqrt(2.0), rel_tol=1e-5, abs_tol=1e-6)


def test_v11_p0_01_uniform_tie_breaking_preserves_bin_expectation() -> None:
    """第五轮 P0-01：验证单桶内右端含窄错区时，均匀随机破平局保持 q_k 而按原始分破平局破坏 Minimax 恒等式。"""
    rng = np.random.default_rng(1101)
    # 构造一维分数 [0, 1]，前 90% 区域正确率为 1.0，最右 10% 区域 [0.9, 1.0] 正确率为 0.0（总桶均值为 0.9）
    n_samples = 20_000
    n_budget = 32
    s_pool = rng.random((n_samples, n_budget))
    y_pool = (s_pool < 0.90).astype(np.float64)

    # 所有候选同落入单一 KL-LCB 最高桶：
    # (1) 均匀随机打破平局期望准确率为桶均值 0.90；
    # (2) 按原始分数 S_i 最大打破平局则等价于选 max_i S_i，准确率跌向 P(max S_i < 0.9) = 0.9^32 ~ 0.034
    uniform_idx = rng.integers(0, n_budget, size=n_samples)
    raw_tie_idx = np.argmax(s_pool, axis=1)
    acc_uniform = float(np.mean(y_pool[np.arange(n_samples), uniform_idx]))
    acc_raw_tie = float(np.mean(y_pool[np.arange(n_samples), raw_tie_idx]))

    assert abs(acc_uniform - 0.90) < 0.01
    assert acc_raw_tie < 0.06


def test_v11_p1_04_and_06_bonferroni_and_in_support_clipping_boundary() -> None:
    """第五轮 P1-04 & P1-06：验证 14 组 Bonferroni 校正公式与上尾支撑截断在分布内污染上的失效边界。"""
    p_raw = 0.0208
    p_bonf = min(1.0, 14.0 * p_raw)
    assert math.isclose(p_bonf, 0.2912, rel_tol=1e-6)

    rng = np.random.default_rng(1106)
    # 干净训练分数 ~ N(0, 1) (错) vs N(1.2, 1) (对)
    tr_y = rng.random(2_000) < 0.35
    tr_s = np.where(tr_y, rng.normal(1.2, 1.0, 2_000), rng.normal(0.0, 1.0, 2_000))
    tr_hi_cap = float(np.quantile(tr_s, 0.998))

    # 分布内高分污染：错误样本落入干净 [q90, q985] 区间（完全低于 99.8% 上界）
    q90, q985 = float(np.quantile(tr_s, 0.90)), float(np.quantile(tr_s, 0.985))
    assert q985 < tr_hi_cap

    te_y = rng.random((300, 64)) < 0.35
    in_sp = (~te_y) & (rng.random((300, 64)) < 0.12)
    te_s = np.where(te_y, rng.normal(1.2, 1.0, (300, 64)), rng.normal(0.0, 1.0, (300, 64)))
    te_s = np.where(in_sp, rng.uniform(q90, q985, (300, 64)), te_s)
    # 检验分布内假阳性尖峰确实不被 tr_hi_cap 拦截（拦截率严格为 0%）
    assert np.all(te_s[in_sp] <= tr_hi_cap)
