"""Stage 2 真实大池与异构多裁判评估接口（real_eval 扩展）全离线单元测试。"""

from __future__ import annotations

import numpy as np

from tts_theory.real_eval import (
    evaluate_heterogeneous_multi_rm_on_real_pools,
    evaluate_joint_m_n_calibration_on_real_pools,
    evaluate_real_prompt_heterogeneity,
    exact_pool_bon_accuracy,
    fit_and_predict_sota_calibrator,
    is_math_answer_correct_any,
)


def test_is_math_answer_correct_any_and_sota_calibrators() -> None:
    """验证多参考答案等价判分及 8 类 SOTA 校准器（含非单调尾部下 KL-LCB 抗坍塌）。"""
    assert is_math_answer_correct_any("2", ["2", "4"])
    assert is_math_answer_correct_any(r"\frac{1}{2n+2}", [r"$\frac{1}{2 n+2}$"])
    assert not is_math_answer_correct_any("5", ["2", "4"])

    rng = np.random.default_rng(202611)
    # 构造强异方差 + Reward Hacking 重右尾污染（中分优、极右尾坍塌）
    def _sample_pool(n_items: int) -> tuple[np.ndarray, np.ndarray]:
        y = rng.random(n_items) < 0.25
        s = np.where(y, rng.normal(1.2, 1.0, n_items), rng.normal(0.0, 1.5, n_items))
        hack = (~y) & (rng.random(n_items) < 0.12)
        s[hack] = rng.normal(4.0, 0.8, int(np.count_nonzero(hack)))
        return s, y

    cal_s, cal_y = _sample_pool(400)
    test_pools = [_sample_pool(64) for _ in range(180)]

    # 自动 k_joint_star 与指定 n_bins 均应正常运行
    auto_pred = fit_and_predict_sota_calibrator(cal_s, cal_y, test_pools[0][0], method="kl_lcb", n_bins=None, n_budget=64)
    assert auto_pred.shape == (64,)
    assert np.all((auto_pred >= 0.0) & (auto_pred <= 1.0))

    # 在 400 条校准样本、K=30 下对比 8 类方法在 N=64 的表现
    methods = (
        "standard_bon",
        "platt_scaling",
        "isotonic_pava",
        "histogram_mle",
        "laplace_smoothed",
        "empirical_bernstein_lcb",
        "wilson_lcb",
        "kl_lcb",
    )
    acc_n64 = {}
    for m in methods:
        curves = [
            exact_pool_bon_accuracy(
                fit_and_predict_sota_calibrator(cal_s, cal_y, ts, method=m, n_bins=30, delta=0.05),
                ty,
                [64],
            )[0]
            for ts, ty in test_pools
        ]
        acc_n64[m] = float(np.mean(curves))

    # 单调方法（Standard-BoN, Platt, Isotonic）被右尾假高分拉崩；KL-LCB / Wilson-LCB 显著胜出
    assert acc_n64["kl_lcb"] > acc_n64["standard_bon"] + 0.10
    assert acc_n64["kl_lcb"] > acc_n64["platt_scaling"] + 0.10
    assert acc_n64["kl_lcb"] > acc_n64["isotonic_pava"] + 0.08
    assert acc_n64["kl_lcb"] > acc_n64["empirical_bernstein_lcb"] + 0.05


def test_evaluate_heterogeneous_multi_rm_5fold_cv() -> None:
    """验证多裁判 5-Fold 无泄露评估：高错相关下 GenEig-OptLinear 负权逆转优于 Mean。"""
    rng = np.random.default_rng(1234)
    n_probs, pool_len = 120, 32
    mats = []
    ys = []
    for _ in range(n_probs):
        y = rng.random(pool_len) < 0.28
        mat = np.empty((pool_len, 2), dtype=np.float64)
        n_pos = int(np.count_nonzero(y))
        n_neg = pool_len - n_pos
        if n_pos > 0:
            mat[y] = rng.normal(1.1, 1.0, size=(n_pos, 2))
        if n_neg > 0:
            # 强同源错相关 rho0 = 0.88, sigma0 = 1.25 -> Mean 必坍塌，w*=(1,-1) 逆转
            z_shared = rng.normal(0.0, 1.0, size=(n_neg, 1))
            e_ind = rng.normal(0.0, 1.0, size=(n_neg, 2))
            mat[~y] = 1.25 * (np.sqrt(0.88) * z_shared + np.sqrt(0.12) * e_ind)
        mats.append(mat)
        ys.append(y)

    res = evaluate_heterogeneous_multi_rm_on_real_pools(
        ["RM-1", "RM-2"], mats, ys, n_values=[1, 4, 16, 32], n_folds=5, seed=17
    )
    assert res["general_linear_diag"]["negative_weight_rescues_mean"] is True
    assert 0.0 < res["copula_eta0"] <= 1.0
    cv = res["cv_results"]
    acc_mean = float(cv["Mean-Ensemble"]["accuracy"][-1])
    acc_nonneg = float(cv["Nonneg-OptLinear"]["accuracy"][-1])
    acc_geneig = float(cv["GenEig-OptLinear"]["accuracy"][-1])
    assert acc_geneig > acc_mean + 0.15
    assert acc_geneig > acc_nonneg + 0.15


def test_evaluate_joint_m_n_calibration_on_real_pools() -> None:
    """验证 2D (M, N) 联合网格评估接口输出维度、n_match 与 1/5 标度收益。"""
    rng = np.random.default_rng(88)

    def _gen(n_items: int) -> tuple[np.ndarray, np.ndarray]:
        y = rng.random(n_items) < 0.25
        s = np.where(y, rng.normal(1.2, 1.0, n_items), rng.normal(0.0, 1.5, n_items))
        return s, y

    cal_s, cal_y = _gen(3_000)
    test_pools = [_gen(32) for _ in range(120)]
    res = evaluate_joint_m_n_calibration_on_real_pools(
        cal_s,
        cal_y,
        test_pools,
        m_grid=[120, 400, 1500],
        n_values=[1, 4, 16, 32],
        sota_m_cal=400,
        sota_n_bins=16,
        sota_repeats=15,
        grid_repeats=20,
        seed=9,
    )
    assert res["grid_opt_1_5_kl"].shape == (3, 4)
    assert res["grid_joint_kl"].shape == (3, 4)
    assert res["grid_legacy_2_5_kl"].shape == (3, 4)
    assert res["grid_fixed60_mle"].shape == (3, 4)
    assert len(res["n_match_theory_by_m"]) == 3
    # 在 M=120, N=32 下最优 1/5 + KL-LCB 明显优于过细 Fixed-60 + MLE
    assert float(res["grid_opt_1_5_kl"][0, -1]) > float(res["grid_fixed60_mle"][0, -1]) + 0.03


def test_evaluate_real_prompt_heterogeneity_simpson_and_zscore() -> None:
    """验证定理 7A 离散辛普森遮蔽检验与定理 7B 题内 Z-Score 在大 N vs N=2 交叉。"""
    rng = np.random.default_rng(515)
    # 场景 1A：辛普森假单调遮蔽（全部区间全局递增，但高分区题内期望导数为负）
    masked_pools = []
    masked_cfg = [("hard", 0.15, -2.0), ("med", 0.48, 0.2), ("easy", 0.92, 3.5)]
    masked_w = [0.40, 0.45, 0.15]
    for _ in range(400):
        name, p_g, shift = masked_cfg[int(rng.choice(3, p=masked_w))]
        y = rng.random(32) < p_g
        s = shift + np.where(y, rng.normal(0.7, 1.0, 32), rng.normal(0.0, 1.35, 32))
        masked_pools.append((name, s, y))

    res_mask = evaluate_real_prompt_heterogeneity(
        masked_pools, n_values=[1, 4, 16], n_folds=5, n_trials_per_problem=10, n_bins=12, n_simpson_bins=12, seed=20
    )
    assert res_mask["simpson_decomposition"]["pooled_appears_monotone"] is True
    assert res_mask["simpson_decomposition"]["masked_transitions"] >= 1

    # 场景 1B：强跨题偏移 + 域内坍塌 (kappa=1.65) -> 大 N 下 Within-Z 显著胜出
    strata_pools = []
    strata_cfg = [("hard", 0.20, -4.5), ("medium", 0.35, 0.0), ("easy", 0.75, 4.5)]
    for _ in range(240):
        name, p_g, shift = strata_cfg[int(rng.integers(0, 3))]
        y = rng.random(32) < p_g
        s = shift + np.where(y, rng.normal(1.0, 1.0, 32), rng.normal(0.0, 1.65, 32))
        strata_pools.append((name, s, y))

    res_shift = evaluate_real_prompt_heterogeneity(
        strata_pools, n_values=[1, 2, 8, 32], n_folds=5, n_trials_per_problem=15, n_bins=12, seed=21
    )
    assert res_shift["affine_shift_var"] > 0.5
    alg_shift = res_shift["algorithms"]
    assert float(alg_shift["Within-Prompt-Z-KL-LCB"]["accuracy"][-1]) > float(
        alg_shift["Pooled-KL-LCB"]["accuracy"][-1]
    ) + 0.05

    # 场景 2：无跨题偏移但非单调坍塌 (kappa=2.2) -> N=2 下 Pooled & Hybrid 胜过退化的 Within-Z
    unshifted_pools = []
    for _ in range(240):
        y = rng.random(16) < 0.30
        s = np.where(y, rng.normal(0.4, 1.0, 16), rng.normal(0.0, 2.2, 16))
        unshifted_pools.append(("homog", s, y))

    res_noshift = evaluate_real_prompt_heterogeneity(
        unshifted_pools, n_values=[1, 2, 16], n_folds=5, n_trials_per_problem=20, n_bins=14, seed=22
    )
    alg_no = res_noshift["algorithms"]
    # N=2 位于索引 1：Pooled 与 Hybrid 应击败 Within-Z
    assert float(alg_no["Pooled-KL-LCB"]["accuracy"][1]) > float(
        alg_no["Within-Prompt-Z-KL-LCB"]["accuracy"][1]
    ) + 0.02
    assert float(alg_no["Hybrid-Adaptive-KL-LCB"]["accuracy"][1]) >= float(
        alg_no["Pooled-KL-LCB"]["accuracy"][1]
    ) - 1e-12
