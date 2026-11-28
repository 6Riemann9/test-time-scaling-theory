"""TMLR 深度诊断与一般 pi_1 >= 0 理论扩展回归测试集。

覆盖 TMLR 改版全部新增理论与反事实审计逻辑（对应 TMLR Theorem 3 与 §7.3）：
1. 一般 pi_1 > 0 且 pi_0 > 0 下紧致峰值相变判据：验证存在有限峰值超越 L_tail = pi_1 / (pi_0 + pi_1)、
   compact_maximizer_exists=True、渐近上限严格小于 1 - pi_0，且极端仿射变换保持 1e-12 级不变；
2. Case 2 自适应右扩界峰值检出（z_peak ≈ 18.81）与 [-4, 8] 实用窗口单调性，及 pi_0 = 0 单内点无损失边界；
3. Case 4 多内点“左尾反超陷阱”与右半轴约束 (Z >= 0) 恢复零上限损失验证；
4. 反事实过滤全错题（hat_pi_0 = 0）与等样本量分层保留对照逻辑正确性；
5. TMLR 实证诊断输出产物 results/tmlr_diagnostics.json 的完整结构、真 i.i.d. Alg1 无虚假暴跌、
   Table 4/5 一致性与 8-Seed 稳健性硬校验。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from tts_theory import (
    diagnose_zscore_induced_collapse,
    verify_sample_zscore_extremal_convergence,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_tmlr_general_pi1_compact_peak_ceiling_loss_and_affine_invariance() -> None:
    """一般 pi_1 > 0 且 pi_0 > 0 下，紧致峰值超越 L_tail 触发在线样本 Z_{i,N} 上限损失与仿射不变性。"""
    diag = diagnose_zscore_induced_collapse(
        prompt_weights=[0.45, 0.45, 0.10],
        prompt_p=[0.0, 0.50, 1.0],
        mu1_base=1.2,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
    )
    expected_tail = 0.10 / (0.45 + 0.10)
    assert math.isclose(float(diag["z_tail_posterior_limit"]), expected_tail, rel_tol=1e-12)
    assert diag["peak_exceeds_tail_limit"] is True
    assert diag["compact_maximizer_exists"] is True
    assert diag["boundary_dominated_regime"] is False
    assert float(diag["q_z_peak"]) > expected_tail + 0.10
    assert float(diag["oracle_z_cal_asymptotic_acc"]) < float(diag["raw_bon_asymptotic_acc"]) - 0.01

    conv_base = verify_sample_zscore_extremal_convergence(
        prompt_weights=[0.45, 0.45, 0.10],
        prompt_p=[0.0, 0.50, 1.0],
        n_values=[16, 64, 256, 1024],
        mu1_base=1.2,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
        prompt_shift=[0.0, 0.0, 0.0],
        prompt_scale=[1.0, 1.0, 1.0],
        n_eval_prompts=2_000,
        seed=123,
    )
    conv_aff = verify_sample_zscore_extremal_convergence(
        prompt_weights=[0.45, 0.45, 0.10],
        prompt_p=[0.0, 0.50, 1.0],
        n_values=[16, 64, 256, 1024],
        mu1_base=1.2,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
        prompt_shift=[-200.0, 500.0, 1000.0],
        prompt_scale=[0.5, 2.0, 5.0],
        n_eval_prompts=2_000,
        seed=123,
    )
    assert float(conv_aff["max_affine_posterior_discrepancy"]) < 1e-12
    diff_acc = np.max(np.abs(conv_base["sample_z_oracle_acc"] - conv_aff["sample_z_oracle_acc"]))
    assert float(diff_acc) < 1e-12
    assert bool(conv_aff["compact_maximizer_exists"]) is True
    assert float(conv_aff["sample_z_oracle_acc"][-1]) < float(conv_aff["raw_bon_acc"][-1]) - 0.01


def test_tmlr_adaptive_peak_detection_and_left_tail_overtake_trap() -> None:
    """验证高 pi_1 自适应扩界真峰检出、pi_0=0 无损失界及 Case 4 左尾反超陷阱。"""
    # 1. 过滤全部不可解题 (pi_0 = 0, pi_1 = 0.50, 单内点层)：右尾极限为 1.0，无紧致损失
    conv_nopi0 = verify_sample_zscore_extremal_convergence(
        prompt_weights=[0.50, 0.50],
        prompt_p=[0.40, 1.0],
        n_values=[16, 64, 256, 1024],
        mu1_base=1.2,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
        n_eval_prompts=2_000,
        seed=456,
    )
    assert math.isclose(float(conv_nopi0["z_tail_posterior_limit"]), 1.0, rel_tol=1e-12)
    assert bool(conv_nopi0["peak_exceeds_tail_limit"]) is False
    assert bool(conv_nopi0["compact_maximizer_exists"]) is False
    assert float(conv_nopi0["theoretical_ceiling_loss"]) == 0.0
    assert abs(float(conv_nopi0["sample_z_oracle_acc"][-1]) - float(conv_nopi0["raw_bon_acc"][-1])) < 0.01

    # 2. 高 pi_1 Case 2：默认 z_grid=None 自动右扩界检出真实理论峰 z_peak ≈ 18.81；
    #    [-4, 8] 上因 [-4, -3.68] 微降至 0.5233 故全窗差分单调性为 False，而 [-3.6, 8] 右侧严格单调递增
    diag_case2 = diagnose_zscore_induced_collapse(
        prompt_weights=[0.05, 0.45, 0.50],
        prompt_p=[0.0, 0.15, 1.0],
        mu1_base=0.4,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
    )
    assert len(diag_case2["z_grid"]) == 1201
    assert math.isclose(float(diag_case2["z_tail_posterior_limit"]), 0.50 / 0.55, rel_tol=1e-12)
    assert diag_case2["practical_window_monotone"] is False
    assert diag_case2["practical_right_window_monotone"] is True
    assert math.isclose(float(diag_case2["practical_window_min_z"]), -3.68, abs_tol=0.05)
    assert math.isclose(float(diag_case2["practical_window_min_q"]), 0.52332, abs_tol=1e-4)
    diag_case2_right = diagnose_zscore_induced_collapse(
        prompt_weights=[0.05, 0.45, 0.50],
        prompt_p=[0.0, 0.15, 1.0],
        mu1_base=0.4,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
        z_grid=np.linspace(-3.6, 8.0, 1161),
    )
    assert diag_case2_right["practical_window_monotone"] is True
    assert diag_case2["peak_exceeds_tail_limit"] is True
    assert diag_case2["compact_maximizer_exists"] is True
    assert 18.0 < float(diag_case2["z_peak"]) < 19.5
    assert float(diag_case2["q_z_peak"]) > float(diag_case2["z_tail_posterior_limit"]) + 0.05
    assert float(diag_case2["asymptotic_ceiling_loss"]) > 5e-4

    # 3. Case 4：多内点“左尾反超陷阱” vs 右半轴约束 (Z >= 0) 恢复零损失
    c4_uncon_diag = diagnose_zscore_induced_collapse(
        prompt_weights=[0.10, 0.45, 0.45],
        prompt_p=[1.0, 0.01, 0.95],
        mu1_base=15.0,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
        constrain_nonnegative_z=False,
    )
    c4_right_diag = diagnose_zscore_induced_collapse(
        prompt_weights=[0.10, 0.45, 0.45],
        prompt_p=[1.0, 0.01, 0.95],
        mu1_base=15.0,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
        constrain_nonnegative_z=True,
    )
    assert c4_uncon_diag["left_tail_overtake_trap"] is True
    assert math.isclose(float(c4_uncon_diag["oracle_z_cal_asymptotic_acc"]), 0.55, abs_tol=1e-12)
    assert math.isclose(float(c4_uncon_diag["asymptotic_ceiling_loss"]), 0.45, abs_tol=1e-12)
    assert c4_right_diag["left_tail_overtake_trap"] is False
    assert math.isclose(float(c4_right_diag["asymptotic_ceiling_loss"]), 0.0, abs_tol=1e-12)

    c4_uncon = verify_sample_zscore_extremal_convergence(
        prompt_weights=[0.10, 0.45, 0.45],
        prompt_p=[1.0, 0.01, 0.95],
        n_values=[64, 1024, 4096],
        mu1_base=15.0,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
        n_eval_prompts=2_000,
        seed=42,
        constrain_nonnegative_z=False,
    )
    c4_right = verify_sample_zscore_extremal_convergence(
        prompt_weights=[0.10, 0.45, 0.45],
        prompt_p=[1.0, 0.01, 0.95],
        n_values=[64, 1024, 4096],
        mu1_base=15.0,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
        n_eval_prompts=2_000,
        seed=42,
        constrain_nonnegative_z=True,
    )
    assert float(c4_uncon["theoretical_ceiling_loss"]) > 0.40
    assert float(c4_right["theoretical_ceiling_loss"]) == 0.0
    assert float(c4_uncon["sample_z_oracle_acc"][-1]) < float(c4_uncon["raw_bon_acc"][-1]) - 0.03
    assert abs(float(c4_right["sample_z_oracle_acc"][-1]) - float(c4_right["raw_bon_acc"][-1])) < 1e-4


def test_tmlr_counterfactual_filtering_logic_invariants() -> None:
    """验证反事实 pi_0 = 0 过滤与等样本量对照组构造不变量。"""
    rng = np.random.default_rng(789)
    synthetic_pools: list[tuple[str, np.ndarray, np.ndarray]] = []
    for i in range(120):
        n_c = 16
        if i % 5 == 0:
            y = np.zeros(n_c, dtype=bool)
        elif i % 5 == 1:
            y = np.ones(n_c, dtype=bool)
        else:
            y = rng.random(n_c) < 0.40
            if np.sum(y) == 0:
                y[0] = True
            if np.sum(y) == n_c:
                y[0] = False
        s = np.where(y, rng.normal(1.2, 1.0, n_c), rng.normal(0.0, 1.0, n_c))
        synthetic_pools.append((f"s{i % 3}", s, y))

    idx_all = list(range(len(synthetic_pools)))
    idx_pi0_zero = [i for i in idx_all if int(np.sum(synthetic_pools[i][2])) > 0]
    idx_interior = [
        i for i in idx_all if 0 < int(np.sum(synthetic_pools[i][2])) < int(synthetic_pools[i][2].size)
    ]
    idx_matched = np.sort(rng.choice(idx_all, size=len(idx_pi0_zero), replace=False)).tolist()

    assert len(idx_pi0_zero) == 96
    assert len(idx_interior) == 72
    assert len(idx_matched) == len(idx_pi0_zero)
    assert all(int(np.sum(synthetic_pools[i][2])) > 0 for i in idx_pi0_zero)
    assert all(
        0 < int(np.sum(synthetic_pools[i][2])) < synthetic_pools[i][2].size for i in idx_interior
    )
    assert any(int(np.sum(synthetic_pools[i][2])) == 0 for i in idx_matched)


def test_tmlr_diagnostics_json_artifact_schema() -> None:
    """验证 results/tmlr_diagnostics.json 完整有效、真 i.i.d. Alg1 无虚假暴跌且 Table 4/5 完全一致。"""
    json_path = PROJECT_ROOT / "results" / "tmlr_diagnostics.json"
    assert json_path.exists(), "请先运行 python scripts/verify_tmlr_diagnostics.py 生成 results/tmlr_diagnostics.json"
    data = json.loads(json_path.read_text(encoding="utf-8"))

    for top_key in (
        "module_a_general_pi1_theory",
        "module_b_real_benchmarks_boundary_audit",
        "module_c_counterfactual_pi0_ablation",
    ):
        assert top_key in data

    # Module A 4-Case 检查
    mod_a = data["module_a_general_pi1_theory"]
    assert "case4_left_tail_overtake_trap" in mod_a
    assert mod_a["case2_high_pi1_practical_boundary"]["adaptive_global_diag"]["compact_maximizer_exists"] is True

    # Module B 非高斯/平局偏离指标检查
    bench = data["module_b_real_benchmarks_boundary_audit"]["benchmarks"]
    assert set(bench.keys()) == {
        "s3_3domain",
        "rb2_skywork",
        "acemath_rm_b",
        "acemath_rm_a",
        "olympiadbench_uprm70b",
    }
    for item in bench.values():
        assert 0.0 <= float(item["pi_0_hat_all_wrong"]) <= 1.0
        assert 0.0 <= float(item["pi_1_hat_all_right"]) <= 1.0
        assert 0.0 <= float(item["prompt_tie_rate"]) <= 1.0
        assert math.isfinite(float(item["z_pos_skewness"]))
        assert math.isfinite(float(item["z_pos_excess_kurtosis"]))
        assert math.isfinite(float(item["v9_within_z_minus_raw"]))

    # Module C：Table 4 Full Train 与 Table 5 行 1/3a/3b 严格一致，无虚假 -7.60 pp 暴跌，8-seed 稳健
    mod_c = data["module_c_counterfactual_pi0_ablation"]
    regimes = mod_c["regimes"]
    assert set(regimes.keys()) == {
        "full_train",
        "filtered_pi0_zero",
        "size_matched_full",
        "filtered_interior_only",
    }
    assert regimes["filtered_pi0_zero"]["pi_0_train"] == 0.0
    assert regimes["filtered_interior_only"]["pi_0_train"] == 0.0
    assert regimes["filtered_interior_only"]["pi_1_train"] == 0.0
    assert regimes["size_matched_full"]["n_train_problems"] == regimes["filtered_pi0_zero"]["n_train_problems"]
    assert regimes["size_matched_full"]["pi_0_train"] > 0.0

    # bin 16 -> bin 20 低先验占比上升且平均先验下滑
    b16_20 = regimes["filtered_pi0_zero"]["alg1_within_z_diag"]["bin16_to_bin20"]
    assert b16_20["bin20_mean_p"] < b16_20["bin16_mean_p"] - 0.15
    assert b16_20["bin20_frac_p_le_010"] > b16_20["bin16_frac_p_le_010"]
    assert b16_20["ols_slope_lcb_16_20"] < 0.0

    # 确认真 i.i.d. Alg1 在 3a_ood 上无虚假暴跌，且与 8-seed 汇总一致
    ms = mod_c["multi_seed_8_summary"]["by_env"]
    assert len(mod_c["multi_seed_8_summary"]["seeds"]) == 8
    diff_3a_alg1 = ms["3a_ood_unseen_test"]["filtered_minus_size_matched"]["alg1_within_z"]
    assert abs(float(diff_3a_alg1["mean"])) < 0.015

    # Table 4 Full Train 与 Table 5 行 1 / 3a / 3b 完全对齐
    shift = mod_c["distribution_shift_matrix"]
    for env_c, env_s in (
        ("1_clean_test", "1_clean"),
        ("3a_ood_unseen_test", "3a_ood_unseen_shift"),
        ("3b_in_support_unseen_test", "3b_in_support_unseen_shift"),
    ):
        assert math.isclose(
            regimes["full_train"]["test_envs"][env_c]["within_z_kl_lcb_acc"],
            shift[env_s]["within_prompt_z_n64"],
            abs_tol=1e-12,
        )
        assert math.isclose(
            regimes["full_train"]["test_envs"][env_c]["alg1_within_z_acc"],
            shift[env_s]["alg1_within_prompt_z_n64"],
            abs_tol=1e-12,
        )
