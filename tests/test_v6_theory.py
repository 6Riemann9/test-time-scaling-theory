"""V6 核心理论深度优化升级单元测试：覆盖定理 5A-III (Safe-Fisher & QDA)、定理 6C 零间隙与定理 7C。"""

from __future__ import annotations

import math

import numpy as np

from tts_theory import (
    GaussianVerifierSpec,
    GeneralGaussianMultiVerifierSpec,
    bayes_qda_multi_verifier_analysis,
    calibration_minimax_scaling,
    diagnose_zscore_induced_collapse,
    gaussian_peak_threshold,
    joint_calibration_compute_scaling,
    pareto_linear_ensemble_frontier,
    simulate_finite_calibration_scaling,
    simulate_multi_verifier_bon,
    simulate_zscore_induced_collapse,
    tight_joint_minimax_scaling,
)


def _orthonormal_matrix(k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(k, k)))
    return q


def test_v6_thm5a3_pareto_safe_fisher_and_degenerate_case() -> None:
    """定理 5A-III Part 1：验证 S-Lemma 凸化帕累托路径单调性、Safe-Fisher 全局最优及退化正交边界解。"""
    # 1. 非退化 3D 耦合实例：Fisher 无约束时 kappa > 1，但 lam_min < 1 可由 Safe-Fisher 挽回
    q_rot = _orthonormal_matrix(3, seed=101)
    cov1 = q_rot @ np.diag([1.0, 1.0, 1.0]) @ q_rot.T
    cov0 = q_rot @ np.diag([0.45, 2.2, 3.0]) @ q_rot.T
    mu1 = q_rot @ np.array([0.22, 1.4, 1.1])
    mu0 = np.zeros(3)
    spec = GeneralGaussianMultiVerifierSpec(p=0.25, mu1=mu1, cov1=cov1, mu0=mu0, cov0=cov0)

    res = pareto_linear_ensemble_frontier(spec, kappa_max=1.0, n_path_points=30)
    assert res["feasible"] is True
    assert res["regime"] == "pareto_interior"
    assert res["kappa_fisher"] > 1.05
    assert abs(res["kappa_safe"] - 1.0) < 1e-6
    assert res["snr2_fisher"] > res["snr2_safe"] > 0.0
    assert res["snr2_deriv_nonpositive"] is True
    assert res["ratio_deriv_nonpositive"] is True

    # 随机采样 1,500 个满足 kappa(w) <= 1.0 的可行方向，验证 w_safe 达到全局极大 SNR
    rng = np.random.default_rng(202)
    dmu = mu1 - mu0
    max_rand_snr2 = -1.0
    for _ in range(1_500):
        w_try = rng.normal(size=3)
        v1 = float(w_try @ cov1 @ w_try)
        v0 = float(w_try @ cov0 @ w_try)
        if v0 / v1 <= 1.0 + 1e-9:
            snr2 = float((w_try @ dmu) ** 2) / (v1 + v0)
            if snr2 > max_rand_snr2:
                max_rand_snr2 = snr2
    assert res["snr2_safe"] >= max_rand_snr2 - 1e-9

    # 2. 退化正交情形：最小特征空间内 d_min == 0，验证精确边界混入解 tau* = lam_min
    q2 = _orthonormal_matrix(2, seed=303)
    cov1_d = q2 @ np.eye(2) @ q2.T
    cov0_d = q2 @ np.diag([0.5, 2.4]) @ q2.T
    mu1_d = q2 @ np.array([0.0, 1.3])
    spec_deg = GeneralGaussianMultiVerifierSpec(p=0.3, mu1=mu1_d, cov1=cov1_d, mu0=np.zeros(2), cov0=cov0_d)

    res_deg = pareto_linear_ensemble_frontier(spec_deg, kappa_max=1.0)
    assert res_deg["regime"] == "degenerate_orthogonal_boundary"
    assert abs(res_deg["tau_star"] - 0.5) < 1e-10
    assert abs(res_deg["kappa_safe"] - 1.0) < 1e-6
    assert res_deg["snr2_safe"] > 0.0


def test_v6_thm5a3_qda_ceiling_monotonicity_and_n_minus_2_over_k() -> None:
    """定理 5A-III Part 2：验证不可逆相 Bayes QDA 天花板 K 单调性、水平集常数 c_K 及 N^{-2/K} 收敛。"""
    p_val = 0.25
    q_ceilings: list[float] = []

    for k_dim in (1, 2, 3):
        mu1 = np.array([0.8, 0.5, 0.35][:k_dim], dtype=np.float64)
        mu0 = np.zeros(k_dim, dtype=np.float64)
        cov1 = np.eye(k_dim, dtype=np.float64)
        cov0 = (1.6**2) * np.eye(k_dim, dtype=np.float64)
        spec_k = GeneralGaussianMultiVerifierSpec(p=p_val, mu1=mu1, cov1=cov1, mu0=mu0, cov0=cov0)
        qda = bayes_qda_multi_verifier_analysis(spec_k)
        assert qda["is_irreversible_linear_collapse"] is True
        assert qda["has_finite_bayes_ceiling"] is True
        assert abs(qda["curse_rate_exponent"] - 2.0 / float(k_dim)) < 1e-12
        q_ceilings.append(float(qda["bayes_ceiling_q"]))

        if k_dim == 1:
            g1 = GaussianVerifierSpec(p=p_val, mu1=0.8, sigma1=1.0, mu0=0.0, sigma0=1.6)
            xp = gaussian_peak_threshold(g1)
            q_peak = float(g1.posterior(xp))
            f_peak = float(g1.mix_pdf(xp))
            tau_sq = 1.0 / (1.0 - 1.0 / (1.6**2))
            curv_c = q_peak * (1.0 - q_peak) / (2.0 * tau_sq)
            pred_a1 = curv_c / (2.0 * (f_peak**2))
            assert abs(q_peak - qda["bayes_ceiling_q"]) < 1e-12
            assert abs(pred_a1 - qda["asymptotic_coef_a_k"]) < 1e-12 * pred_a1

        # 用 400k 抽样精确检验 K=1 和 K=2 的水平集尾常数 c_K u^{K/2}
        if k_dim in (1, 2):
            rng = np.random.default_rng(7 + k_dim)
            n_s = 400_000
            y = rng.random(n_s) < p_val
            s_draw = np.where(
                y[:, None],
                mu1[None, :] + rng.normal(size=(n_s, k_dim)),
                1.6 * rng.normal(size=(n_s, k_dim)),
            )
            a_m, b_v, c0_s = qda["a_qda"], qda["b_qda"], qda["c0_qda"]
            ell_s = np.einsum("ij,jk,ik->i", s_draw, a_m, s_draw) + (s_draw @ b_v) + c0_s
            q_draw = 1.0 / (1.0 + np.exp(-(math.log(p_val / (1.0 - p_val)) + ell_s)))
            u_probe = 0.005
            emp_sf = float(np.mean(q_draw > float(qda["bayes_ceiling_q"]) - u_probe))
            th_sf = float(qda["c_k"]) * (u_probe ** (float(k_dim) / 2.0))
            assert 0.96 <= emp_sf / th_sf <= 1.04

    # 天花板随裁判数 K 严格递增（正新信息严格提升上界）
    assert q_ceilings[0] < q_ceilings[1] < q_ceilings[2] < 1.0

    # 用 A_2 显著的多裁判规格检验 60k MC 的 N^{-1} 遗憾速率（regret >> binomial SE）
    spec2_large = GeneralGaussianMultiVerifierSpec(
        p=0.20,
        mu1=np.array([1.2, 1.0]),
        cov1=np.eye(2),
        mu0=np.zeros(2),
        cov0=(2.2**2) * np.eye(2),
    )
    qda2 = bayes_qda_multi_verifier_analysis(spec2_large)
    mc2 = simulate_multi_verifier_bon(
        spec2_large, n_values=[32, 64], aggregation="qda", n_trials=60_000, seed=11, batch_size=3_000
    )
    reg_32 = float(qda2["bayes_ceiling_q"]) - float(mc2["accuracy"][0])
    reg_64 = float(qda2["bayes_ceiling_q"]) - float(mc2["accuracy"][1])
    th_32 = float(qda2["asymptotic_coef_a_k"]) / 32.0
    th_64 = float(qda2["asymptotic_coef_a_k"]) / 64.0
    assert 0.80 <= reg_32 / th_32 <= 1.25
    assert 0.70 <= reg_64 / th_64 <= 1.35


def test_v6_thm6c_zero_gap_n_agnostic_beats_truncated() -> None:
    """定理 6C（V6）：验证零对数间隙全平面上界，及大 M 小 N 时 N-无关分桶优于截断分桶。"""
    z_gap = tight_joint_minimax_scaling(
        m_total=100_000, n_budget=8, beta=2.0, curvature_c=0.1, f_min=0.25, eta_radius=1.5
    )
    assert z_gap["is_zero_gap_minimax_tight"] is True
    assert z_gap["uniform_bin_count_condition_holds"] is True
    assert z_gap["total_upper_bound"] > z_gap["minimax_rate_lower"] > 0.0

    # 在大 M=50,000, 小 N=8 下对比 N-无关最优分桶 K*(M) 与 V5 截断分桶 K_comp=2
    spec = GaussianVerifierSpec(p=0.30, mu1=0.6, sigma1=1.0, mu0=0.0, sigma0=1.8)
    k_n_agnostic = int(calibration_minimax_scaling(50_000, beta=2.0)["k_opt_theory"])
    assert k_n_agnostic >= 6

    sim_agnostic = simulate_finite_calibration_scaling(
        spec, m_cal=50_000, n_bins=k_n_agnostic, n_values=[4, 8, 16], methods=("kl_lcb",), n_repeats=120, delta=0.05, seed=91
    )
    sim_trunc = simulate_finite_calibration_scaling(
        spec, m_cal=50_000, n_bins=2, n_values=[4, 8, 16], methods=("kl_lcb",), n_repeats=120, delta=0.05, seed=91
    )
    acc_ag = sim_agnostic["methods"]["kl_lcb"]["accuracy"]
    acc_tr = sim_trunc["methods"]["kl_lcb"]["accuracy"]
    assert float(acc_ag[1]) > float(acc_tr[1]) + 0.015
    assert float(acc_ag[2]) > float(acc_tr[2]) + 0.025


def test_v6_thm7c_zscore_self_induced_tail_inversion_and_kappa_gt_1() -> None:
    """定理 7C：验证同方差单调裁判 (kappa=1) 在含边界原子时 Z-空间自诱导 kappa_Z > 1 与尾反转。"""
    z_g = np.linspace(-3.5, 9.0, 1251)

    # 1. 无全对题 (pi_1 = 0 < pi_0 = 0.55)：Z-空间右尾精确极限为 0，且 Oracle Z 严劣于 Raw
    diag_no1 = diagnose_zscore_induced_collapse(
        prompt_weights=[0.55, 0.45],
        prompt_p=[0.0, 0.50],
        mu1_base=1.2,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
        z_grid=z_g,
    )
    assert abs(diag_no1["pi_0_unsolvable"] - 0.55) < 1e-12
    assert abs(diag_no1["pi_1_trivial"] - 0.0) < 1e-12
    assert abs(diag_no1["z_tail_posterior_limit"] - 0.0) < 1e-12
    assert diag_no1["z_tail_value"] < 5e-4
    assert diag_no1["kappa_z"] > 1.10
    assert diag_no1["tail_turn_down"] is True
    assert diag_no1["z_induces_collapse"] is True
    assert diag_no1["oracle_z_cal_asymptotic_acc"] < diag_no1["raw_bon_asymptotic_acc"] - 0.015

    # 2. 含全对题与不可解题 (pi_1 = 0.05, pi_0 = 0.15)：精确右尾极限 = 0.05 / 0.20 = 0.25
    diag_with1 = diagnose_zscore_induced_collapse(
        prompt_weights=[0.15, 0.05, 0.40, 0.40],
        prompt_p=[0.0, 1.0, 0.35, 0.80],
        mu1_base=2.0,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
        z_grid=z_g,
    )
    assert abs(diag_with1["z_tail_posterior_limit"] - 0.25) < 1e-12
    assert abs(diag_with1["z_tail_value"] - 0.25) < 1e-3
    assert diag_with1["kappa_z"] > 1.05
    assert diag_with1["z_induces_collapse"] is True


def test_v6_thm7c_simulation_induced_collapse_on_monotone_verifier() -> None:
    """定理 7C 仿真：基底 kappa=1 单调模型上，Within-Prompt-Z-KL-LCB 呈现倒 U 坍塌而 Raw/Pooled 递增。"""
    sim = simulate_zscore_induced_collapse(
        prompt_weights=[0.55, 0.45],
        prompt_p=[0.0, 0.50],
        n_values=[1, 8, 32, 64, 128, 256],
        mu1_base=1.2,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
        m_cal_prompts=2_000,
        n_cal_per_prompt=64,
        n_eval_prompts=2_000,
        n_bins=30,
        seed=42,
    )
    acc_raw = sim["standard_bon_mc"]
    acc_pool = sim["pooled_kl_lcb_mc"]
    acc_z = sim["within_prompt_z_kl_lcb_mc"]
    # Within-Z 在 N=64 后倒 U 下跌，到 N=256 显著落后 Raw 与 Pooled
    assert float(acc_z[3]) > float(acc_z[-1]) + 0.01
    assert float(acc_raw[-1]) > float(acc_z[-1]) + 0.025
    assert float(acc_pool[-1]) > float(acc_z[-1]) + 0.025


def test_v6_backward_compatibility_joint_scaling_keys() -> None:
    """确认 V6 对 joint_calibration_compute_scaling 的扩展保持 V5 字段 100% 向后兼容。"""
    res = joint_calibration_compute_scaling(m_total=400, n_budget=10_000, beta=2.0)
    for k in (
        "k_cal_star",
        "k_comp_limit",
        "k_joint_star",
        "n_match",
        "oracle_n_rate",
        "cal_m_rate",
        "failure_prob_delta_opt",
        "minimax_lower_rate",
        "kl_lcb_upper_rate",
        "is_minimax_tight",
        "compute_log_gap_factor",
        "regime",
        "n_agnostic_k_opt",
        "n_agnostic_upper_rate",
        "n_agnostic_is_minimax_tight",
    ):
        assert k in res
    assert res["n_agnostic_is_minimax_tight"] is True
