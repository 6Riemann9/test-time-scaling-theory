"""TMLR 深度诊断与一般 pi_1 >= 0 理论扩展验证脚本（Module A / B / C + 多划分稳健性 + 分布漂移矩阵）。

输出 results/tmlr_diagnostics.json（绝不修改或破坏既有归档 JSON 文件）：
1. 模块 A（一般 pi_1 >= 0 理论数值核验，对应 TMLR Theorem 3）：
   - Case 1 (pi_0=0.45, pi_1=0.10, mu1=1.2)：有限点峰值超越 L_tail=pi_1/(pi_0+pi_1)，存在紧致最大值集并产生真实上限损失（含极端仿射变换下 1e-12 级不变性）；
   - Case 2 (pi_0=0.05, pi_1=0.50, mu1=0.4)：自适应右扩界精确检出理论峰 (z≈18.81 > L_tail=0.9091)，同时记录 [-4, 8] 实用窗口单调性；
   - Case 3 (pi_0=0.0, pi_1=0.50, mu1=1.2)：去掉不可解题 (pi_0=0) 后，L_tail = 1.0，右半轴/单内点下理论上限损失严格为 0；
   - Case 4 (pi_0=0.0, pi_1=0.10, p_int=(0.01, 0.95), Delta=15.0)：双内点“左尾反超陷阱”实证，对比无约束双尾极大化在高 N 下的退化与右半轴约束 (Z >= 0) 恢复零上限损失。
2. 模块 B（真实基准经验 pi_0_hat / pi_1_hat 与 TMLR Theorem 3 判据实测审计）：
   - 在 Strategic-TTC (S3 3-Domain)、RB2 Skywork、AceMath RM-B/A、OlympiadBench uPRM-70B 上统计经验边界比例、Z*-分桶后验走势、题目平局率与 Z|Y=1 偏度/峰度非高斯偏离指标，并与 50/20/30 留出测试集上的 Within-Z 掉点幅度对照。
3. 模块 C（AceMath RM-B 反事实 pi_0=0 分布漂移消融 + 完整 7-Regime 分布漂移矩阵 + 8-Seed 稳健性）：
   - 修复真 i.i.d. 样本分割随机打乱（禁止按子集顺序切片）及支撑截断置 -1.0（禁止置 0.0 引发全零桶平局复活）；
   - 同口径输出 seed=835 主划分下连续 EmpiricalScoreCalibrator 与离散 Alg 1 的完整 Table 4 & Table 5 对齐数据；
   - 增补 8 个随机划分种子 ([101, 202, 303, 404, 505, 606, 707, 835]) 的 mean ± SD 稳健性复核与 bin 16 -> bin 20 局部斜率/先验组成定量诊断。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import run_stage2_experiments as rs2  # noqa: E402
from tts_theory import (  # noqa: E402
    EmpiricalScoreCalibrator,
    diagnose_zscore_induced_collapse,
    verify_sample_zscore_extremal_convergence,
)
from tts_theory.calibration_theory import compute_bin_calibrator_scores  # noqa: E402
from tts_theory.heterogeneity_theory import _within_prompt_zscore  # noqa: E402


def _paired_ttest(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """对同批独立测试题上的配对得分向量执行双侧学生化 t 检验与 95% 正态渐近区间。"""
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = int(diff.size)
    mean_d = float(np.mean(diff))
    sd_d = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    se_d = sd_d / math.sqrt(n) if n > 1 else 0.0
    if se_d <= 1e-15:
        p_val = 1.0
    else:
        t_stat = mean_d / se_d
        p_val = float(max(np.finfo(np.float64).tiny, 2.0 * stats.t.sf(abs(t_stat), df=n - 1)))
    return {
        "mean_diff": mean_d,
        "se_diff": se_d,
        "ci_low": mean_d - 1.96 * se_d,
        "ci_high": mean_d + 1.96 * se_d,
        "p_value_ttest": p_val,
    }


def _load_acemath_problems() -> list[dict[str, Any]]:
    ace_dst = rs2.DATA_DIR / "acemath_rewardbench" / "default_test_0.parquet"
    tbl = pq.read_table(ace_dst)
    rm_all = tbl["rm_score"].to_pylist()
    corr_all = tbl["score"].to_pylist()
    offset_b = 8987
    # 与 run_stage2_experiments.py 保持完全一致的 7 个分段边界
    if hasattr(rs2, "ACEMATH_SEGMENTS"):
        segments = rs2.ACEMATH_SEGMENTS
    else:
        segments = (
            ("college_math", 0, 2818),
            ("gaokao2023en", 2818, 3203),
            ("gsm8k", 3203, 4522),
            ("math500", 4522, 5022),
            ("minerva_math", 5022, 5294),
            ("mmlu_stem", 5294, 8312),
            ("olympiadbench", 8312, 8987),
        )
    all_probs = []
    for sub_name, s0, s1 in segments:
        for idx in range(s0, s1):
            all_probs.append(
                {
                    "subset": sub_name,
                    "rm_b": np.asarray(rm_all[offset_b + idx], dtype=np.float64),
                    "rm_a": np.asarray(rm_all[idx], dtype=np.float64),
                    "is_correct": np.asarray(corr_all[idx], dtype=bool),
                }
            )
    return all_probs


def run_module_a_general_pi1_theory() -> dict[str, Any]:
    """模块 A：一般 pi_1 >= 0 紧致峰值损失、自适应峰值、左尾反超陷阱及右半轴约束恢复核验。"""
    print("[TMLR Module A] Verifying general pi_1 >= 0 regimes, adaptive peak, and left-tail trap...")
    n_vals = [16, 64, 256, 1024, 4096]

    # Case 1: pi_0 = 0.45, pi_1 = 0.10, p_int = 0.50 (Delta=1.2) -> L_tail = 0.1818 < q_peak ≈ 0.5291
    c1_base = verify_sample_zscore_extremal_convergence(
        prompt_weights=[0.45, 0.45, 0.10],
        prompt_p=[0.0, 0.50, 1.0],
        n_values=n_vals,
        mu1_base=1.2,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
        prompt_shift=[0.0, 0.0, 0.0],
        prompt_scale=[1.0, 1.0, 1.0],
        n_eval_prompts=3_000,
        seed=42,
    )
    c1_aff = verify_sample_zscore_extremal_convergence(
        prompt_weights=[0.45, 0.45, 0.10],
        prompt_p=[0.0, 0.50, 1.0],
        n_values=n_vals,
        mu1_base=1.2,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
        prompt_shift=[-200.0, 500.0, 1000.0],
        prompt_scale=[0.5, 2.0, 5.0],
        n_eval_prompts=3_000,
        seed=42,
    )
    c1_diag = diagnose_zscore_induced_collapse(
        prompt_weights=[0.45, 0.45, 0.10],
        prompt_p=[0.0, 0.50, 1.0],
        mu1_base=1.2,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
    )

    # Case 2: pi_0 = 0.05, pi_1 = 0.50, p_int = 0.15 (Delta=0.4) -> 默认 z_grid=None 下自适应扩展找到真实理论峰 z≈18.81
    c2_default = verify_sample_zscore_extremal_convergence(
        prompt_weights=[0.05, 0.45, 0.50],
        prompt_p=[0.0, 0.15, 1.0],
        n_values=n_vals,
        mu1_base=0.4,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
        prompt_shift=[-200.0, 500.0, 1000.0],
        prompt_scale=[0.5, 2.0, 5.0],
        n_eval_prompts=3_000,
        seed=42,
    )
    c2_adaptive_diag = diagnose_zscore_induced_collapse(
        prompt_weights=[0.05, 0.45, 0.50],
        prompt_p=[0.0, 0.15, 1.0],
        mu1_base=0.4,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
    )

    # Case 3: 过滤全错题 pi_0 = 0, pi_1 = 0.50, p_int = 0.40 (Delta=1.2) -> L_tail = 1.0，单内点下严格无上限损失
    c3_nopi0 = verify_sample_zscore_extremal_convergence(
        prompt_weights=[0.50, 0.50],
        prompt_p=[0.40, 1.0],
        n_values=n_vals,
        mu1_base=1.2,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
        prompt_shift=[-200.0, 500.0],
        prompt_scale=[0.5, 2.0],
        n_eval_prompts=3_000,
        seed=42,
    )
    c3_diag = diagnose_zscore_induced_collapse(
        prompt_weights=[0.50, 0.50],
        prompt_p=[0.40, 1.0],
        mu1_base=1.2,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
    )

    # Case 4: 双内点“左尾反超陷阱” w=(0.10, 0.45, 0.45), p=(1.0, 0.01, 0.95), Delta=15.0, sigma=1.0
    c4_unconstrained = verify_sample_zscore_extremal_convergence(
        prompt_weights=[0.10, 0.45, 0.45],
        prompt_p=[1.0, 0.01, 0.95],
        n_values=n_vals,
        mu1_base=15.0,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
        n_eval_prompts=3_000,
        seed=42,
        constrain_nonnegative_z=False,
    )
    c4_right_half = verify_sample_zscore_extremal_convergence(
        prompt_weights=[0.10, 0.45, 0.45],
        prompt_p=[1.0, 0.01, 0.95],
        n_values=n_vals,
        mu1_base=15.0,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
        n_eval_prompts=3_000,
        seed=42,
        constrain_nonnegative_z=True,
    )
    c4_diag = diagnose_zscore_induced_collapse(
        prompt_weights=[0.10, 0.45, 0.45],
        prompt_p=[1.0, 0.01, 0.95],
        mu1_base=15.0,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
        z_grid=np.linspace(-6.0, 6.0, 1201),
        constrain_nonnegative_z=False,
    )
    c4_diag_right = diagnose_zscore_induced_collapse(
        prompt_weights=[0.10, 0.45, 0.45],
        prompt_p=[1.0, 0.01, 0.95],
        mu1_base=15.0,
        sigma1_base=1.0,
        mu0_base=0.0,
        sigma0_base=1.0,
        z_grid=np.linspace(-6.0, 6.0, 1201),
        constrain_nonnegative_z=True,
    )

    return {
        "case1_compact_peak_loss": {
            "spec": {"weights": [0.45, 0.45, 0.10], "p": [0.0, 0.50, 1.0], "mu1": 1.2},
            "base": c1_base,
            "affine_shifted": c1_aff,
            "curve": {
                "z_grid": c1_diag["z_grid"],
                "q_pool_z": c1_diag["q_pool_z"],
                "z_peak": float(c1_diag["z_peak"]),
                "q_z_peak": float(c1_diag["q_z_peak"]),
                "z_tail_posterior_limit": float(c1_diag["z_tail_posterior_limit"]),
            },
            "max_sample_z_acc_diff_under_affine": float(
                np.max(np.abs(c1_base["sample_z_oracle_acc"] - c1_aff["sample_z_oracle_acc"]))
            ),
        },
        "case2_high_pi1_practical_boundary": {
            "spec": {"weights": [0.05, 0.45, 0.50], "p": [0.0, 0.15, 1.0], "mu1": 0.4},
            "practical_grid_eval": c2_default,
            "adaptive_global_diag": {
                "z_peak": float(c2_adaptive_diag["z_peak"]),
                "q_z_peak": float(c2_adaptive_diag["q_z_peak"]),
                "z_tail_posterior_limit": float(c2_adaptive_diag["z_tail_posterior_limit"]),
                "peak_exceeds_tail_limit": bool(c2_adaptive_diag["peak_exceeds_tail_limit"]),
                "compact_maximizer_exists": bool(c2_adaptive_diag["compact_maximizer_exists"]),
                "practical_window_monotone": bool(c2_adaptive_diag["practical_window_monotone"]),
                "practical_right_window_monotone": bool(c2_adaptive_diag["practical_right_window_monotone"]),
                "practical_window_min_z": float(c2_adaptive_diag["practical_window_min_z"]),
                "practical_window_min_q": float(c2_adaptive_diag["practical_window_min_q"]),
                "oracle_z_cal_asymptotic_acc": float(c2_adaptive_diag["oracle_z_cal_asymptotic_acc"]),
                "raw_bon_asymptotic_acc": float(c2_adaptive_diag["raw_bon_asymptotic_acc"]),
                "asymptotic_ceiling_loss": float(c2_adaptive_diag["asymptotic_ceiling_loss"]),
                "z_grid_practical": c2_adaptive_diag["z_grid"],
                "q_pool_z_practical": c2_adaptive_diag["q_pool_z"],
            },
        },
        "case3_filtered_pi0_zero_no_loss": {
            "spec": {"weights": [0.50, 0.50], "p": [0.40, 1.0], "mu1": 1.2},
            "eval": c3_nopi0,
            "curve": {
                "z_grid": c3_diag["z_grid"],
                "q_pool_z": c3_diag["q_pool_z"],
            },
        },
        "case4_left_tail_overtake_trap": {
            "spec": {"weights": [0.10, 0.45, 0.45], "p": [1.0, 0.01, 0.95], "mu1": 15.0},
            "unconstrained_two_sided": c4_unconstrained,
            "right_half_constrained_z_ge_0": c4_right_half,
            "diag_unconstrained": {
                "left_tail_overtake_trap": bool(c4_diag["left_tail_overtake_trap"]),
                "oracle_z_cal_asymptotic_acc": float(c4_diag["oracle_z_cal_asymptotic_acc"]),
                "asymptotic_ceiling_loss": float(c4_diag["asymptotic_ceiling_loss"]),
            },
            "diag_right_half": {
                "left_tail_overtake_trap": bool(c4_diag_right["left_tail_overtake_trap"]),
                "oracle_z_cal_asymptotic_acc": float(c4_diag_right["oracle_z_cal_asymptotic_acc"]),
                "asymptotic_ceiling_loss": float(c4_diag_right["asymptotic_ceiling_loss"]),
            },
            "curve": {
                "z_grid": c4_diag["z_grid"],
                "q_pool_z": c4_diag["q_pool_z"],
            },
        },
    }


def _audit_single_benchmark_z_boundary(
    name: str,
    pools: list[tuple[str, np.ndarray, np.ndarray]],
    v9_key: str,
    v9_data: dict[str, Any],
    n_bins: int = 20,
) -> dict[str, Any]:
    """统计单一真实基准的经验边界原子比例、Z*-分桶峰值与非高斯/平局偏离统计。"""
    p_cnt = len(pools)
    cand_lens = np.array([int(y.size) for _, _, y in pools], dtype=np.int64)
    corrs = np.array([int(np.sum(y)) for _, _, y in pools], dtype=np.int64)
    is_zero = corrs == 0
    is_all1 = corrs == cand_lens
    is_int = (~is_zero) & (~is_all1)

    pi_0 = float(np.mean(is_zero))
    pi_1 = float(np.mean(is_all1))
    pi_int = float(np.mean(is_int))
    tail_limit = float(pi_1 / (pi_0 + pi_1)) if (pi_0 + pi_1) > 0.0 else None

    # 统计题目内候选打分平局率（存在任意一对打分近似相等的题目比例）与最大平局簇占比
    tie_prompts = 0
    const_prompts = 0
    z_chunks = []
    y_chunks = []
    z_flag_zero = []
    z_flag_all1 = []
    for idx, (_, s_arr, y_arr) in enumerate(pools):
        s_flat = np.asarray(s_arr, dtype=np.float64)
        s_sorted = np.sort(s_flat)
        if s_sorted.size > 1 and float(np.min(np.diff(s_sorted))) <= 1e-12:
            tie_prompts += 1
        if float(np.ptp(s_flat)) <= 1e-12:
            const_prompts += 1

        s_vec = s_flat[None, :]
        z_vec = _within_prompt_zscore(s_vec)[0]
        y_vec = np.asarray(y_arr, dtype=bool)
        z_chunks.append(z_vec)
        y_chunks.append(y_vec)
        z_flag_zero.append(np.full(z_vec.size, bool(is_zero[idx]), dtype=bool))
        z_flag_all1.append(np.full(z_vec.size, bool(is_all1[idx]), dtype=bool))

    z_all = np.concatenate(z_chunks)
    y_all = np.concatenate(y_chunks)
    zero_all = np.concatenate(z_flag_zero)
    all1_all = np.concatenate(z_flag_all1)

    # 经验样本 Z-score 条件方差比 kappa_Z_hat 及偏度/峰度非高斯偏离诊断
    z_pos = z_all[y_all]
    z_neg = z_all[~y_all]
    var_z_pos = float(np.var(z_pos, ddof=1)) if z_pos.size > 1 else 0.0
    var_z_neg = float(np.var(z_neg, ddof=1)) if z_neg.size > 1 else 0.0
    kappa_z_hat = float(math.sqrt(var_z_neg / max(var_z_pos, 1e-15)))
    z_pos_skew = float(stats.skew(z_pos, bias=False)) if z_pos.size > 2 else 0.0
    z_pos_kurt = float(stats.kurtosis(z_pos, fisher=True, bias=False)) if z_pos.size > 3 else 0.0
    z_neg_skew = float(stats.skew(z_neg, bias=False)) if z_neg.size > 2 else 0.0
    z_neg_kurt = float(stats.kurtosis(z_neg, fisher=True, bias=False)) if z_neg.size > 3 else 0.0

    # 离散等频分位分桶，统计各桶实测正确率及右尾边界占比
    def _bin_stats(k_target: int) -> dict[str, Any]:
        qs = np.linspace(0.0, 1.0, k_target + 1)
        raw_edges = np.quantile(z_all, qs)
        inner_edges = np.unique(raw_edges[1:-1])
        bin_ids = np.searchsorted(inner_edges, z_all, side="right")
        k_eff = int(inner_edges.size + 1)
        counts = np.bincount(bin_ids, minlength=k_eff).astype(np.float64)
        pos_cnts = np.bincount(bin_ids, weights=y_all.astype(np.float64), minlength=k_eff)
        zero_cnts = np.bincount(bin_ids, weights=zero_all.astype(np.float64), minlength=k_eff)
        all1_cnts = np.bincount(bin_ids, weights=all1_all.astype(np.float64), minlength=k_eff)
        sum_z = np.bincount(bin_ids, weights=z_all, minlength=k_eff)

        safe_cnt = np.maximum(1.0, counts)
        q_bins = pos_cnts / safe_cnt
        z_centers = sum_z / safe_cnt
        peak_idx = int(np.argmax(q_bins))
        q_peak = float(q_bins[peak_idx])
        q_right = float(q_bins[-1])
        peak_exceeds = bool(q_peak > tail_limit + 1e-9) if tail_limit is not None else None
        return {
            "k_requested": int(k_target),
            "k_effective": k_eff,
            "z_bin_centers": [float(v) for v in z_centers],
            "q_z_bins": [float(v) for v in q_bins],
            "bin_counts": [int(v) for v in counts],
            "q_z_peak_hat": q_peak,
            "peak_bin_1idx": peak_idx + 1,
            "q_z_right_tail_hat": q_right,
            "right_tail_drop": float(q_peak - q_right),
            "right_tail_zero_prompt_frac": float(zero_cnts[-1] / safe_cnt[-1]),
            "right_tail_all1_prompt_frac": float(all1_cnts[-1] / safe_cnt[-1]),
            "peak_zero_prompt_frac": float(zero_cnts[peak_idx] / safe_cnt[peak_idx]),
            "empirical_peak_exceeds_tail_limit": peak_exceeds,
        }

    b_main = _bin_stats(n_bins)
    b_alt10 = _bin_stats(10) if n_bins != 10 else b_main

    v9_entry = v9_data["disjoint_single_split_primary"][v9_key]
    algs = v9_entry["algorithms"]
    raw_nmax = float(algs["Standard-BoN"]["accuracy"][-1])
    pool_nmax = float(algs["Pooled-KL-LCB"]["accuracy"][-1])
    z_nmax = float(algs["Within-Prompt-Z-KL-LCB"]["accuracy"][-1])

    return {
        "benchmark": name,
        "n_problems": p_cnt,
        "n_cand_min": int(np.min(cand_lens)),
        "n_cand_max": int(np.max(cand_lens)),
        "prompt_tie_rate": float(tie_prompts / max(1, p_cnt)),
        "prompt_constant_score_rate": float(const_prompts / max(1, p_cnt)),
        "pi_0_hat_all_wrong": pi_0,
        "pi_1_hat_all_right": pi_1,
        "pi_int_hat_interior": pi_int,
        "tail_limit_pred_pi1_over_pi0_plus_pi1": tail_limit,
        "var_z_pos_hat": var_z_pos,
        "var_z_neg_hat": var_z_neg,
        "kappa_z_hat": kappa_z_hat,
        "z_pos_skewness": z_pos_skew,
        "z_pos_excess_kurtosis": z_pos_kurt,
        "z_neg_skewness": z_neg_skew,
        "z_neg_excess_kurtosis": z_neg_kurt,
        "binned_k20": b_main,
        "binned_k10": b_alt10,
        "v9_50_20_30_n_max": int(v9_entry["n_values"][-1]),
        "v9_raw_bon": raw_nmax,
        "v9_pooled_kl_lcb": pool_nmax,
        "v9_within_z": z_nmax,
        "v9_within_z_minus_raw": float(z_nmax - raw_nmax),
        "v9_within_z_minus_pooled": float(z_nmax - pool_nmax),
    }


def run_module_b_real_benchmarks_boundary_audit(all_probs_ace: list[dict[str, Any]]) -> dict[str, Any]:
    """模块 B：加载 5 个基准设置（来自 4 个评测数据集）并计算经验边界比例与非高斯偏离审计。"""
    print("[TMLR Module B] Loading 5 benchmark-RM configurations and auditing (pi_0, pi_1) & non-Gaussianity...")
    v9_path = PROJECT_ROOT / "results" / "v9_consolidation_verification.json"
    v9_data = json.loads(v9_path.read_text(encoding="utf-8"))

    ace_b_pools = [(p["subset"], p["rm_b"], p["is_correct"]) for p in all_probs_ace]
    ace_a_pools = [(p["subset"], p["rm_a"], p["is_correct"]) for p in all_probs_ace]

    rb2_root = rs2.STAGE2_DIR / "reward_bench_2"
    rb2_loaded = {}
    for m_name, rel in rs2.RB2_HETERO6.items():
        fpath = rs2.download_hf_file_retry("allenai/reward-bench-2-results", rel, rb2_root)
        rb2_loaded[m_name] = rs2._load_rb2_model(fpath)
    _, h6_mats, h6_ys, h6_strata = rs2._align_rb2_group(rb2_loaded)
    rb2_pools = [(st, mat[:, 0], y) for st, mat, y in zip(h6_strata, h6_mats, h6_ys)]

    sky_raw = json.loads((rb2_root / rs2.RB2_HETERO6["Skywork-V2-Llama3.1-8B"]).read_text(encoding="utf-8"))
    sky_nc = np.asarray([int(x) for x in sky_raw["num_correct"]], dtype=np.int64)
    sky_len = np.asarray([len(s) for s in sky_raw["scores"]], dtype=np.int64)
    rb2_raw_file_stats = {
        "total_items_in_raw_json": int(sky_nc.size),
        "all_wrong_count": int(np.sum(sky_nc <= 0)),
        "all_right_count": int(np.sum(sky_nc >= sky_len)),
        "aligned_interior_problems": len(rb2_pools),
        "note": "RB2 benchmark items construct 1+ correct vs wrong choices; alignment keeps mixed-label items.",
    }

    oly_root = rs2.STAGE2_DIR / "olympiadbench"
    repo_70b = "sibasmarakp/Qwen2.5-7B-Instruct-uPRM-70B-T80-olympiadbench-best_of_n-completions"
    repo_32b = "sibasmarakp/Qwen2.5-7B-Instruct-uPRM-32B-T80-olympiadbench-best_of_n-completions"
    p70, _ = rs2._parse_olympiad_repo(repo_70b, oly_root / "uprm_70b")
    p32, _ = rs2._parse_olympiad_repo(repo_32b, oly_root / "uprm_32b")
    map70 = {p["id"]: p for p in p70}
    map32 = {p["id"]: p for p in p32}
    common_ids = sorted(set(map70.keys()) & set(map32.keys()))
    oly_pools = [(map70[pid]["subfield"], map70[pid]["prm_last"], map70[pid]["is_correct"]) for pid in common_ids]

    ttc_parsed = {
        tag: rs2._stream_parse_strategic_ttc(tag, rs2.TTC_FILES[tag])
        for tag in ("gsm8k_qwen7b", "gpqa_qwen7b", "aime_qwen7b")
    }
    s3_pools: list[tuple[str, np.ndarray, np.ndarray]] = []
    s3_domain_breakdown: dict[str, Any] = {}
    for dom_name, tag in (("GSM8K", "gsm8k_qwen7b"), ("GPQA", "gpqa_qwen7b"), ("AIME", "aime_qwen7b")):
        dom_pools = [
            (dom_name, np.asarray(x["rewards"], dtype=np.float64), np.asarray(x["correct"], dtype=bool))
            for x in ttc_parsed[tag]
        ]
        s3_pools.extend(dom_pools)
        corrs = np.array([int(np.sum(y)) for _, _, y in dom_pools], dtype=np.int64)
        lens = np.array([int(y.size) for _, _, y in dom_pools], dtype=np.int64)
        s3_domain_breakdown[dom_name] = {
            "n_problems": len(dom_pools),
            "pi_0_hat": float(np.mean(corrs == 0)),
            "pi_1_hat": float(np.mean(corrs == lens)),
            "pi_int_hat": float(np.mean((corrs > 0) & (corrs < lens))),
        }

    audits = {
        "s3_3domain": _audit_single_benchmark_z_boundary("S3 3-Domain", s3_pools, "s3_3domain", v9_data, n_bins=20),
        "rb2_skywork": _audit_single_benchmark_z_boundary("RB2 Skywork", rb2_pools, "rb2_skywork", v9_data, n_bins=20),
        "acemath_rm_b": _audit_single_benchmark_z_boundary("AceMath RM-B", ace_b_pools, "acemath_rm_b", v9_data, n_bins=20),
        "acemath_rm_a": _audit_single_benchmark_z_boundary("AceMath RM-A", ace_a_pools, "acemath_rm_a", v9_data, n_bins=20),
        "olympiadbench_uprm70b": _audit_single_benchmark_z_boundary(
            "OlympiadBench uPRM-70B", oly_pools, "olympiadbench_uprm70b", v9_data, n_bins=20
        ),
    }
    return {
        "benchmarks": audits,
        "rb2_raw_file_metadata": rb2_raw_file_stats,
        "s3_domain_boundary_breakdown": s3_domain_breakdown,
    }


def _describe_calibrator_curve(cal: EmpiricalScoreCalibrator) -> dict[str, Any]:
    """抽取 EmpiricalScoreCalibrator 学得的 knots_x_ / knots_q_ 曲线，诊断是否右尾倒挂。"""
    assert cal.knots_x_ is not None and cal.knots_q_ is not None
    centers = np.asarray(cal.knots_x_, dtype=np.float64)
    scores = np.asarray(cal.knots_q_, dtype=np.float64)
    peak_idx = int(np.argmax(scores))
    peak_val = float(scores[peak_idx])
    right_val = float(scores[-1])
    return {
        "k_knots": int(scores.size),
        "knots_x": [float(v) for v in centers],
        "knots_lcb_scores": [float(v) for v in scores],
        "peak_knot_1idx": peak_idx + 1,
        "peak_x": float(centers[peak_idx]),
        "peak_lcb": peak_val,
        "right_tail_x": float(centers[-1]),
        "right_tail_lcb": right_val,
        "right_tail_lcb_drop": float(peak_val - right_val),
        "has_right_tail_downturn": bool(right_val < peak_val - 1e-4),
    }


def _describe_alg1_z_diagnostics(
    vec_z: np.ndarray,
    vec_y: np.ndarray,
    vec_p_emp: np.ndarray,
    n_bins: int = 20,
    split_seed: int = 202612,
) -> dict[str, Any]:
    """在真 i.i.d. 对半分割的 Alg 1 离散分桶上提取 LCB 曲线、bin 16 -> bin 20 组成及局部斜率。"""
    m_tot = int(vec_z.size)
    perm = np.random.default_rng(split_seed).permutation(m_tot)
    z_shuf = vec_z[perm]
    y_shuf = vec_y[perm]
    p_shuf = vec_p_emp[perm]

    m_half = m_tot // 2
    s_sorted = np.sort(z_shuf[:m_half])
    r_ord = np.floor(np.arange(1, n_bins, dtype=np.float64) * float(m_half + 1) / float(n_bins)).astype(np.int64)
    r_ord = np.clip(r_ord - 1, 0, m_half - 1)
    edges = np.unique(s_sorted[r_ord])

    z_est, y_est = z_shuf[m_half:], y_shuf[m_half:]
    k_cur = int(edges.size + 1)
    b_est = np.searchsorted(edges, z_est, side="right")
    cnts = np.bincount(b_est, minlength=k_cur).astype(np.float64)
    poss = np.bincount(b_est, weights=y_est.astype(np.float64), minlength=k_cur)
    scores = compute_bin_calibrator_scores(
        cnts, poss, method="kl_lcb", prior_p=float(np.mean(y_est)), delta=0.05
    )

    # 在全部训练样本上统计每个分桶的中心 Z、经验准确率和所属题目经验先验分布（bin 16 -> bin 20 定量分解）
    b_all = np.searchsorted(edges, z_shuf, side="right")
    cnt_all = np.maximum(1.0, np.bincount(b_all, minlength=k_cur).astype(np.float64))
    z_centers = np.bincount(b_all, weights=z_shuf, minlength=k_cur) / cnt_all
    q_emp_all = np.bincount(b_all, weights=y_shuf.astype(np.float64), minlength=k_cur) / cnt_all
    mean_p_by_bin = np.bincount(b_all, weights=p_shuf, minlength=k_cur) / cnt_all
    low_p10_by_bin = np.bincount(b_all, weights=(p_shuf <= 0.10).astype(np.float64), minlength=k_cur) / cnt_all
    low_p20_by_bin = np.bincount(b_all, weights=(p_shuf <= 0.20).astype(np.float64), minlength=k_cur) / cnt_all

    peak_idx = int(np.argmax(scores))
    peak_val = float(scores[peak_idx])
    right_val = float(scores[-1])

    # bin 16 -> bin 20（1-indexed: 16..20, 0-indexed: 15..19）局部斜率与组成变化
    idx_16 = max(0, min(k_cur - 2, 15))
    idx_20 = k_cur - 1
    dz_16_20 = max(1e-9, float(z_centers[idx_20] - z_centers[idx_16]))
    slope_q_16_20 = float((q_emp_all[idx_20] - q_emp_all[idx_16]) / dz_16_20)
    slope_lcb_16_20 = float((scores[idx_20] - scores[idx_16]) / dz_16_20)
    # 亦做 bin 16..20 五点的线性回归斜率
    if idx_20 - idx_16 >= 2:
        x_seg = z_centers[idx_16 : idx_20 + 1]
        ols_slope_lcb_16_20 = float(np.polyfit(x_seg, scores[idx_16 : idx_20 + 1], 1)[0])
        ols_slope_q_16_20 = float(np.polyfit(x_seg, q_emp_all[idx_16 : idx_20 + 1], 1)[0])
    else:
        ols_slope_lcb_16_20 = slope_lcb_16_20
        ols_slope_q_16_20 = slope_q_16_20

    return {
        "k_bins": k_cur,
        "edges": [float(v) for v in edges],
        "bin_scores_lcb": [float(v) for v in scores],
        "bin_z_centers": [float(v) for v in z_centers],
        "bin_q_emp": [float(v) for v in q_emp_all],
        "bin_mean_prompt_p": [float(v) for v in mean_p_by_bin],
        "bin_frac_p_le_010": [float(v) for v in low_p10_by_bin],
        "bin_frac_p_le_020": [float(v) for v in low_p20_by_bin],
        "peak_bin_1idx": peak_idx + 1,
        "peak_lcb": peak_val,
        "right_tail_lcb": right_val,
        "peak_minus_right_lcb": float(peak_val - right_val),
        "bin16_to_bin20": {
            "bin16_mean_p": float(mean_p_by_bin[idx_16]),
            "bin20_mean_p": float(mean_p_by_bin[idx_20]),
            "bin16_frac_p_le_010": float(low_p10_by_bin[idx_16]),
            "bin20_frac_p_le_010": float(low_p10_by_bin[idx_20]),
            "bin16_frac_p_le_020": float(low_p20_by_bin[idx_16]),
            "bin20_frac_p_le_020": float(low_p20_by_bin[idx_20]),
            "bin16_lcb": float(scores[idx_16]),
            "bin20_lcb": float(scores[idx_20]),
            "endpoint_slope_lcb_16_20": slope_lcb_16_20,
            "endpoint_slope_q_16_20": slope_q_16_20,
            "ols_slope_lcb_16_20": ols_slope_lcb_16_20,
            "ols_slope_q_16_20": ols_slope_q_16_20,
        },
    }


def _fit_alg1_discrete_iid(
    vec_s: np.ndarray,
    vec_y: np.ndarray,
    n_bins: int = 20,
    split_seed: int = 202612,
    clip_q: float = 0.998,
) -> tuple[np.ndarray, np.ndarray, float]:
    """严格遵循 Algorithm 1：先对 (vec_s, vec_y) 做真 i.i.d. 随机置换，再对半切分边界与估计集。"""
    m_tot = int(vec_s.size)
    perm = np.random.default_rng(split_seed).permutation(m_tot)
    s_shuf = vec_s[perm]
    y_shuf = vec_y[perm]

    m_half = m_tot // 2
    s_sorted = np.sort(s_shuf[:m_half])
    r_ord = np.floor(np.arange(1, n_bins, dtype=np.float64) * float(m_half + 1) / float(n_bins)).astype(np.int64)
    r_ord = np.clip(r_ord - 1, 0, m_half - 1)
    edges = np.unique(s_sorted[r_ord])
    s_est, y_est = s_shuf[m_half:], y_shuf[m_half:]
    k_cur = int(edges.size + 1)
    b_est = np.searchsorted(edges, s_est, side="right")
    cnts = np.bincount(b_est, minlength=k_cur).astype(np.float64)
    poss = np.bincount(b_est, weights=y_est.astype(np.float64), minlength=k_cur)
    scores = compute_bin_calibrator_scores(
        cnts, poss, method="kl_lcb", prior_p=float(np.mean(y_est)), delta=0.05
    )
    s_max = float(s_sorted[min(m_half - 1, max(0, math.ceil(clip_q * m_half) - 1))])
    return edges, scores, s_max


def _eval_per_problem_bon(
    score_list: list[np.ndarray], y_list: list[np.ndarray]
) -> np.ndarray:
    """在每道题全部 N 个候选上，按均匀随机破平局计算精确单题命中期望。"""
    out = np.empty(len(score_list), dtype=np.float64)
    for idx, (sc, yy) in enumerate(zip(score_list, y_list)):
        mx = float(np.max(sc))
        ties = np.abs(sc - mx) <= 1e-12
        out[idx] = float(np.mean(yy[ties]))
    return out


def _stratified_50_20_30_split(
    pools: list[tuple[str, np.ndarray, np.ndarray]], seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """分层随机划分 50% / 20% / 30%（返回题级索引）。"""
    split_rng = np.random.default_rng(seed)
    by_stratum: dict[str, list[int]] = {}
    for idx, (st, _, _) in enumerate(pools):
        by_stratum.setdefault(str(st), []).append(idx)
    train_idx_l: list[int] = []
    val_idx_l: list[int] = []
    test_idx_l: list[int] = []
    for st_key in sorted(by_stratum.keys()):
        arr = np.asarray(by_stratum[st_key], dtype=np.int64)
        perm = split_rng.permutation(arr)
        n_s = int(perm.size)
        n_tr = max(1, int(round(0.50 * n_s)))
        n_va = max(1, int(round(0.20 * n_s)))
        if n_tr + n_va >= n_s:
            n_tr = max(1, n_s - 2)
            n_va = 1
        train_idx_l.extend(perm[:n_tr].tolist())
        val_idx_l.extend(perm[n_tr : n_tr + n_va].tolist())
        test_idx_l.extend(perm[n_tr + n_va :].tolist())
    return (
        np.asarray(sorted(train_idx_l), dtype=np.int64),
        np.asarray(sorted(val_idx_l), dtype=np.int64),
        np.asarray(sorted(test_idx_l), dtype=np.int64),
    )


def _build_acemath_contaminated_pools(
    all_probs_ace: list[dict[str, Any]],
) -> tuple[
    list[tuple[str, np.ndarray, np.ndarray]],
    list[tuple[str, np.ndarray, np.ndarray]],
    list[tuple[str, np.ndarray, np.ndarray]],
]:
    """构造干净、6% OOD 尖峰污染与 6% In-Support [q90, q985] 污染三套 AceMath RM-B 候选池。"""
    clean_pools = [(p["subset"], p["rm_b"], p["is_correct"]) for p in all_probs_ace]
    all_rm_b = np.concatenate([p["rm_b"] for p in all_probs_ace])
    sd_b = float(np.std(all_rm_b))
    q995 = float(np.quantile(all_rm_b, 0.995))
    q90 = float(np.quantile(all_rm_b, 0.90))
    q985 = float(np.quantile(all_rm_b, 0.985))

    rng_ood = np.random.default_rng(202609)
    ood_pools: list[tuple[str, np.ndarray, np.ndarray]] = []
    for p in all_probs_ace:
        s_arr = p["rm_b"].copy()
        y_arr = p["is_correct"].copy()
        w_mask = ~y_arr
        if np.any(w_mask):
            sp = (rng_ood.random(s_arr.size) < 0.06) & w_mask
            if np.any(sp):
                s_arr[sp] = q995 + 0.8 * sd_b + np.abs(rng_ood.normal(0.0, 0.5 * sd_b, size=int(np.sum(sp))))
        ood_pools.append((p["subset"], s_arr, y_arr))

    rng_insup = np.random.default_rng(202610)
    insup_pools: list[tuple[str, np.ndarray, np.ndarray]] = []
    for p in all_probs_ace:
        s_arr = p["rm_b"].copy()
        y_arr = p["is_correct"].copy()
        w_mask = ~y_arr
        if np.any(w_mask):
            sp = (rng_insup.random(s_arr.size) < 0.06) & w_mask
            n_sp = int(np.sum(sp))
            if n_sp > 0:
                s_arr[sp] = rng_insup.uniform(q90, q985, size=n_sp)
        insup_pools.append((p["subset"], s_arr, y_arr))

    return clean_pools, ood_pools, insup_pools


def _run_shift_matrix_one_seed(
    clean_pools: list[tuple[str, np.ndarray, np.ndarray]],
    ood_pools: list[tuple[str, np.ndarray, np.ndarray]],
    insup_pools: list[tuple[str, np.ndarray, np.ndarray]],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    alg1_seed: int,
    n_bins: int = 20,
) -> dict[str, Any]:
    """用修复后的真 i.i.d. Alg 1 与连续 EmpiricalScoreCalibrator 运行完整 7-Regime 分布漂移矩阵。"""

    def _run_one(
        tr_src: list[tuple[str, np.ndarray, np.ndarray]],
        va_src: list[tuple[str, np.ndarray, np.ndarray]],
        te_src: list[tuple[str, np.ndarray, np.ndarray]],
        seed_offset_p: int,
        seed_offset_z: int,
    ) -> dict[str, Any]:
        tr_s = np.concatenate([tr_src[i][1] for i in train_idx])
        tr_y = np.concatenate([tr_src[i][2].astype(np.int64) for i in train_idx])
        tr_z = np.concatenate([_within_prompt_zscore(tr_src[i][1][None, :])[0] for i in train_idx])

        cal_pool = EmpiricalScoreCalibrator(
            n_bins=n_bins, method="quantile_lcb", lcb_delta=0.05, lcb_type="kl"
        ).fit(tr_s, tr_y)
        cal_z = EmpiricalScoreCalibrator(
            n_bins=n_bins, method="quantile_lcb", lcb_delta=0.05, lcb_type="kl"
        ).fit(tr_z, tr_y)
        tr_hi_cap = float(np.quantile(tr_s, 0.998))

        p_edges, p_alg1_scores, p_alg1_smax = _fit_alg1_discrete_iid(
            tr_s, tr_y, n_bins=n_bins, split_seed=alg1_seed + seed_offset_p
        )
        z_edges, z_alg1_scores, _ = _fit_alg1_discrete_iid(
            tr_z, tr_y, n_bins=n_bins, split_seed=alg1_seed + seed_offset_z
        )

        def _eval_idx(indices: np.ndarray, src: list[tuple[str, np.ndarray, np.ndarray]]) -> dict[str, np.ndarray]:
            sc_raw, sc_pool, sc_z, sc_clip = [], [], [], []
            sc_a1_pool, sc_a1_z, sc_a1_clip, ys = [], [], [], []
            for p_i in indices:
                _, s_p, y_p = src[int(p_i)]
                q_pool = np.asarray(cal_pool.predict_proba(s_p), dtype=np.float64)
                z_p = _within_prompt_zscore(s_p[None, :])[0]
                q_z = np.asarray(cal_z.predict_proba(z_p), dtype=np.float64)
                # 严格按修复后的 Algorithm 1：超支撑截断置 -1.0（绝对低于合法后验区间 [0, 1]，杜绝全零桶复活）
                q_clip = np.where(s_p > tr_hi_cap, -1.0, q_pool)

                bp_p = np.searchsorted(p_edges, s_p, side="right")
                qa1_pool = p_alg1_scores[bp_p]
                qa1_clip = np.where(s_p > p_alg1_smax, -1.0, qa1_pool)

                bp_z = np.searchsorted(z_edges, z_p, side="right")
                qa1_z = z_alg1_scores[bp_z]

                sc_raw.append(s_p)
                sc_pool.append(q_pool)
                sc_z.append(q_z)
                sc_clip.append(q_clip)
                sc_a1_pool.append(qa1_pool)
                sc_a1_z.append(qa1_z)
                sc_a1_clip.append(qa1_clip)
                ys.append(y_p)

            return {
                "Raw-BoN": _eval_per_problem_bon(sc_raw, ys),
                "Pooled-KL-LCB": _eval_per_problem_bon(sc_pool, ys),
                "Within-Prompt-Z": _eval_per_problem_bon(sc_z, ys),
                "Support-Clipped": _eval_per_problem_bon(sc_clip, ys),
                "Alg1-Pooled": _eval_per_problem_bon(sc_a1_pool, ys),
                "Alg1-Within-Z": _eval_per_problem_bon(sc_a1_z, ys),
                "Alg1-Support-Clipped": _eval_per_problem_bon(sc_a1_clip, ys),
            }

        val_res = _eval_idx(val_idx, va_src)
        te_res = _eval_idx(test_idx, te_src)
        branches = ("Raw-BoN", "Pooled-KL-LCB", "Within-Prompt-Z")
        best_cont = max(branches, key=lambda b: float(np.mean(val_res[b])))
        hyb_cont = te_res[best_cont]

        alg1_branches = ("Raw-BoN", "Alg1-Pooled", "Alg1-Within-Z")
        best_alg1 = max(alg1_branches, key=lambda b: float(np.mean(val_res[b])))
        hyb_alg1 = te_res[best_alg1]

        t_vs_raw = _paired_ttest(hyb_cont, te_res["Raw-BoN"])
        t_z_vs_raw = _paired_ttest(te_res["Within-Prompt-Z"], te_res["Raw-BoN"])
        return {
            "n_train": int(train_idx.size),
            "n_val": int(val_idx.size),
            "n_test": int(test_idx.size),
            "selected_branch_cont": best_cont,
            "selected_branch_alg1": best_alg1,
            "raw_bon_n64": float(np.mean(te_res["Raw-BoN"])),
            "pooled_kl_lcb_n64": float(np.mean(te_res["Pooled-KL-LCB"])),
            "within_prompt_z_n64": float(np.mean(te_res["Within-Prompt-Z"])),
            "hybrid_fixed_n64": float(np.mean(hyb_cont)),
            "support_clipped_n64": float(np.mean(te_res["Support-Clipped"])),
            "alg1_pooled_kl_lcb_n64": float(np.mean(te_res["Alg1-Pooled"])),
            "alg1_within_prompt_z_n64": float(np.mean(te_res["Alg1-Within-Z"])),
            "alg1_hybrid_fixed_n64": float(np.mean(hyb_alg1)),
            "alg1_support_clipped_n64": float(np.mean(te_res["Alg1-Support-Clipped"])),
            "hybrid_vs_raw_ttest": t_vs_raw,
            "within_z_vs_raw_ttest": t_z_vs_raw,
        }

    # 按训练集本身固定 alg1 随机置换种子：clean_pools 与 Module C full_train 严格对齐 (300000+835*100+1/2)
    return {
        "1_clean": _run_one(clean_pools, clean_pools, clean_pools, 1, 2),
        "2a_ood_matched": _run_one(ood_pools, ood_pools, ood_pools, 11, 12),
        "3a_ood_unseen_shift": _run_one(clean_pools, clean_pools, ood_pools, 1, 2),
        "4a_ood_target_val": _run_one(clean_pools, ood_pools, ood_pools, 1, 2),
        "2b_in_support_matched": _run_one(insup_pools, insup_pools, insup_pools, 21, 22),
        "3b_in_support_unseen_shift": _run_one(clean_pools, clean_pools, insup_pools, 1, 2),
        "4b_in_support_target_val": _run_one(clean_pools, insup_pools, insup_pools, 1, 2),
    }


def _run_counterfactual_one_seed(
    clean_pools: list[tuple[str, np.ndarray, np.ndarray]],
    ood_pools: list[tuple[str, np.ndarray, np.ndarray]],
    insup_pools: list[tuple[str, np.ndarray, np.ndarray]],
    split_seed: int,
    n_bins: int = 20,
    include_full_diagnostics: bool = True,
) -> dict[str, Any]:
    """在指定划分种子上执行 4 组干净训练集变体的反事实 pi_0=0 消融。"""
    train_idx, _, test_idx = _stratified_50_20_30_split(clean_pools, split_seed)
    tr_full = train_idx.tolist()
    tr_pi0_zero = [i for i in tr_full if int(np.sum(clean_pools[i][2])) > 0]
    tr_interior_only = [
        i for i in tr_full if 0 < int(np.sum(clean_pools[i][2])) < int(clean_pools[i][2].size)
    ]

    rng_sz = np.random.default_rng(202611 + split_seed)
    tr_size_matched = np.sort(rng_sz.choice(train_idx, size=len(tr_pi0_zero), replace=False)).tolist()

    regimes = [
        ("full_train", tr_full),
        ("filtered_pi0_zero", tr_pi0_zero),
        ("size_matched_full", tr_size_matched),
        ("filtered_interior_only", tr_interior_only),
    ]

    test_environments = [
        ("1_clean_test", [clean_pools[i] for i in test_idx]),
        ("3a_ood_unseen_test", [ood_pools[i] for i in test_idx]),
        ("3b_in_support_unseen_test", [insup_pools[i] for i in test_idx]),
    ]

    raw_per_env: dict[str, np.ndarray] = {}
    for env_name, te_list in test_environments:
        raw_per_env[env_name] = _eval_per_problem_bon(
            [x[1] for x in te_list], [x[2] for x in te_list]
        )

    regime_results: dict[str, Any] = {}
    z_test_vectors: dict[str, dict[str, np.ndarray]] = {}
    alg1_z_test_vectors: dict[str, dict[str, np.ndarray]] = {}

    for reg_idx, (reg_name, tr_ids) in enumerate(regimes):
        tr_items = [clean_pools[i] for i in tr_ids]
        tr_corrs = np.array([int(np.sum(x[2])) for x in tr_items], dtype=np.int64)
        tr_lens = np.array([int(x[2].size) for x in tr_items], dtype=np.int64)
        tr_emp_p = tr_corrs.astype(np.float64) / np.maximum(1.0, tr_lens.astype(np.float64))

        tr_s = np.concatenate([x[1] for x in tr_items])
        tr_y = np.concatenate([x[2].astype(np.int64) for x in tr_items])
        tr_z = np.concatenate([_within_prompt_zscore(x[1][None, :])[0] for x in tr_items])
        tr_p_cand = np.concatenate([np.full(x[1].size, p_val, dtype=np.float64) for x, p_val in zip(tr_items, tr_emp_p)])

        cal_pool = EmpiricalScoreCalibrator(
            n_bins=n_bins, method="quantile_lcb", lcb_delta=0.05, lcb_type="kl"
        ).fit(tr_s, tr_y)
        cal_z = EmpiricalScoreCalibrator(
            n_bins=n_bins, method="quantile_lcb", lcb_delta=0.05, lcb_type="kl"
        ).fit(tr_z, tr_y)
        tr_hi_cap = float(np.quantile(tr_s, 0.998))

        alg1_seed_p = 300000 + split_seed * 100 + reg_idx * 10 + 1
        alg1_seed_z = 300000 + split_seed * 100 + reg_idx * 10 + 2
        p_edges, p_alg1_scores, p_alg1_smax = _fit_alg1_discrete_iid(
            tr_s, tr_y, n_bins=n_bins, split_seed=alg1_seed_p
        )
        z_edges, z_alg1_scores, _ = _fit_alg1_discrete_iid(
            tr_z, tr_y, n_bins=n_bins, split_seed=alg1_seed_z
        )

        env_summary: dict[str, Any] = {}
        z_test_vectors[reg_name] = {}
        alg1_z_test_vectors[reg_name] = {}

        for env_name, te_list in test_environments:
            sc_pool_l, sc_z_l, sc_clip_l = [], [], []
            sc_alg1_pool_l, sc_alg1_z_l, sc_alg1_clip_l, te_y_l = [], [], [], []
            for _, s_p, y_p in te_list:
                q_pool = np.asarray(cal_pool.predict_proba(s_p), dtype=np.float64)
                z_p = _within_prompt_zscore(s_p[None, :])[0]
                q_z = np.asarray(cal_z.predict_proba(z_p), dtype=np.float64)
                q_clip = np.where(s_p > tr_hi_cap, -1.0, q_pool)

                bp_z = np.searchsorted(z_edges, z_p, side="right")
                qa1_z = z_alg1_scores[bp_z]
                bp_p = np.searchsorted(p_edges, s_p, side="right")
                qa1_pool = p_alg1_scores[bp_p]
                qa1_clip = np.where(s_p > p_alg1_smax, -1.0, qa1_pool)

                sc_pool_l.append(q_pool)
                sc_z_l.append(q_z)
                sc_clip_l.append(q_clip)
                sc_alg1_pool_l.append(qa1_pool)
                sc_alg1_z_l.append(qa1_z)
                sc_alg1_clip_l.append(qa1_clip)
                te_y_l.append(y_p)

            vec_raw = raw_per_env[env_name]
            vec_pool = _eval_per_problem_bon(sc_pool_l, te_y_l)
            vec_z = _eval_per_problem_bon(sc_z_l, te_y_l)
            vec_clip = _eval_per_problem_bon(sc_clip_l, te_y_l)
            vec_alg1_pool = _eval_per_problem_bon(sc_alg1_pool_l, te_y_l)
            vec_alg1_z = _eval_per_problem_bon(sc_alg1_z_l, te_y_l)
            vec_alg1_clip = _eval_per_problem_bon(sc_alg1_clip_l, te_y_l)

            z_test_vectors[reg_name][env_name] = vec_z
            alg1_z_test_vectors[reg_name][env_name] = vec_alg1_z
            env_summary[env_name] = {
                "raw_bon_acc": float(np.mean(vec_raw)),
                "pooled_kl_lcb_acc": float(np.mean(vec_pool)),
                "within_z_kl_lcb_acc": float(np.mean(vec_z)),
                "support_clipped_acc": float(np.mean(vec_clip)),
                "alg1_pooled_kl_lcb_acc": float(np.mean(vec_alg1_pool)),
                "alg1_within_z_acc": float(np.mean(vec_alg1_z)),
                "alg1_support_clipped_acc": float(np.mean(vec_alg1_clip)),
                "within_z_minus_raw": _paired_ttest(vec_z, vec_raw),
                "within_z_minus_pooled": _paired_ttest(vec_z, vec_pool),
            }

        reg_entry: dict[str, Any] = {
            "n_train_problems": len(tr_items),
            "n_train_candidates": int(tr_s.size),
            "pi_0_train": float(np.mean(tr_corrs == 0)),
            "pi_1_train": float(np.mean(tr_corrs == tr_lens)),
            "pi_int_train": float(np.mean((tr_corrs > 0) & (tr_corrs < tr_lens))),
            "train_raw_hi_cap_998": tr_hi_cap,
            "alg1_within_z_peak_minus_right": float(np.max(z_alg1_scores) - z_alg1_scores[-1]),
            "test_envs": env_summary,
        }
        if include_full_diagnostics:
            reg_entry["cal_z_curve"] = _describe_calibrator_curve(cal_z)
            reg_entry["cal_pool_curve"] = _describe_calibrator_curve(cal_pool)
            reg_entry["alg1_within_z_diag"] = _describe_alg1_z_diagnostics(
                tr_z, tr_y, tr_p_cand, n_bins=n_bins, split_seed=alg1_seed_z
            )
            reg_entry["alg1_within_z_bin_scores"] = [float(v) for v in z_alg1_scores]
        regime_results[reg_name] = reg_entry

    paired_vs_full: dict[str, Any] = {}
    paired_pi0zero_vs_matched: dict[str, Any] = {}
    for env_name, _ in test_environments:
        paired_pi0zero_vs_matched[env_name] = {
            "continuous": _paired_ttest(
                z_test_vectors["filtered_pi0_zero"][env_name],
                z_test_vectors["size_matched_full"][env_name],
            ),
            "alg1_discrete": _paired_ttest(
                alg1_z_test_vectors["filtered_pi0_zero"][env_name],
                alg1_z_test_vectors["size_matched_full"][env_name],
            ),
        }
    for reg_name in ("filtered_pi0_zero", "size_matched_full", "filtered_interior_only"):
        paired_vs_full[reg_name] = {}
        for env_name, _ in test_environments:
            paired_vs_full[reg_name][env_name] = _paired_ttest(
                z_test_vectors[reg_name][env_name], z_test_vectors["full_train"][env_name]
            )

    return {
        "split_seed": int(split_seed),
        "n_test_problems": int(test_idx.size),
        "n_budget": 64,
        "regimes": regime_results,
        "within_z_paired_vs_full_train": paired_vs_full,
        "filtered_pi0_zero_vs_size_matched": paired_pi0zero_vs_matched,
    }


def run_module_c_counterfactual_pi0_ablation(all_probs_ace: list[dict[str, Any]]) -> dict[str, Any]:
    """模块 C：seed=835 主消融 + 完整 7-Regime 分布漂移矩阵 + 8 个随机划分种子稳健性审计。"""
    print("[TMLR Module C] Running counterfactual pi_0=0 ablation, 7-regime shift matrix, & 8-seed check...")
    clean_pools, ood_pools, insup_pools = _build_acemath_contaminated_pools(all_probs_ace)

    # 1. seed=835 主划分：同时计算 4-regime 反事实消融与完整 7-regime 分布漂移矩阵（保持 Table 4/5 100% 口径一致）
    main_res = _run_counterfactual_one_seed(
        clean_pools, ood_pools, insup_pools, split_seed=835, n_bins=20, include_full_diagnostics=True
    )
    tr_835, va_835, te_835 = _stratified_50_20_30_split(clean_pools, 835)
    shift_matrix = _run_shift_matrix_one_seed(
        clean_pools, ood_pools, insup_pools, tr_835, va_835, te_835, alg1_seed=300000 + 835 * 100, n_bins=20
    )

    # 2. 8 个随机划分种子的多划分稳健性诊断
    seeds_8 = [101, 202, 303, 404, 505, 606, 707, 835]
    per_seed_records: list[dict[str, Any]] = []
    for sd in seeds_8:
        res_sd = (
            main_res
            if sd == 835
            else _run_counterfactual_one_seed(
                clean_pools, ood_pools, insup_pools, split_seed=sd, n_bins=20, include_full_diagnostics=False
            )
        )
        per_seed_records.append(res_sd)

    def _agg_stats(vals: list[float]) -> dict[str, float]:
        arr = np.asarray(vals, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "sd": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    envs = ("1_clean_test", "3a_ood_unseen_test", "3b_in_support_unseen_test")
    reg_names = ("full_train", "filtered_pi0_zero", "size_matched_full", "filtered_interior_only")
    multi_seed_summary: dict[str, Any] = {"seeds": seeds_8, "by_env": {}}
    for env_name in envs:
        raw_list = [r["regimes"]["full_train"]["test_envs"][env_name]["raw_bon_acc"] for r in per_seed_records]
        env_blk: dict[str, Any] = {"raw_bon_acc": _agg_stats(raw_list), "regimes": {}}
        for reg_name in reg_names:
            cont_list = [r["regimes"][reg_name]["test_envs"][env_name]["within_z_kl_lcb_acc"] for r in per_seed_records]
            alg1_list = [r["regimes"][reg_name]["test_envs"][env_name]["alg1_within_z_acc"] for r in per_seed_records]
            env_blk["regimes"][reg_name] = {
                "continuous_within_z": _agg_stats(cont_list),
                "alg1_within_z": _agg_stats(alg1_list),
            }
        # filtered_pi0_zero - size_matched_full 配对差值的跨种子分布
        diff_cont = [
            r["regimes"]["filtered_pi0_zero"]["test_envs"][env_name]["within_z_kl_lcb_acc"]
            - r["regimes"]["size_matched_full"]["test_envs"][env_name]["within_z_kl_lcb_acc"]
            for r in per_seed_records
        ]
        diff_alg1 = [
            r["regimes"]["filtered_pi0_zero"]["test_envs"][env_name]["alg1_within_z_acc"]
            - r["regimes"]["size_matched_full"]["test_envs"][env_name]["alg1_within_z_acc"]
            for r in per_seed_records
        ]
        env_blk["filtered_minus_size_matched"] = {
            "continuous_within_z": _agg_stats(diff_cont),
            "alg1_within_z": _agg_stats(diff_alg1),
        }
        multi_seed_summary["by_env"][env_name] = env_blk

    return {
        "n_test_problems": main_res["n_test_problems"],
        "n_budget": main_res["n_budget"],
        "regimes": main_res["regimes"],
        "within_z_paired_vs_full_train": main_res["within_z_paired_vs_full_train"],
        "filtered_pi0_zero_vs_size_matched": main_res["filtered_pi0_zero_vs_size_matched"],
        "distribution_shift_matrix": shift_matrix,
        "multi_seed_8_summary": multi_seed_summary,
    }


def main() -> None:
    mod_a = rs2.to_json_safe(run_module_a_general_pi1_theory())
    print("\n=== Module A: General pi_1 >= 0 Summary ===", flush=True)
    c1 = mod_a["case1_compact_peak_loss"]
    print(
        f"Case1 (pi0=0.45, pi1=0.10): L_tail={c1['base']['z_tail_posterior_limit']:.4f}, "
        f"z_peak={c1['curve']['z_peak']:.4f}, q_peak={c1['curve']['q_z_peak']:.4f}, "
        f"N4096 SampleZ={c1['affine_shifted']['sample_z_oracle_acc'][-1]:.4f} vs Raw={c1['affine_shifted']['raw_bon_acc'][-1]:.4f} "
        f"(affine_diff={c1['max_sample_z_acc_diff_under_affine']:.2e})",
        flush=True,
    )
    c2 = mod_a["case2_high_pi1_practical_boundary"]
    c2d = c2["adaptive_global_diag"]
    print(
        f"Case2 (pi0=0.05, pi1=0.50, mu1=0.4): adaptive z_peak={c2d['z_peak']:.4f}, q_peak={c2d['q_z_peak']:.4f}, "
        f"compact={c2d['compact_maximizer_exists']}, full_mono={c2d['practical_window_monotone']}, "
        f"right_mono={c2d['practical_right_window_monotone']} (min_z={c2d['practical_window_min_z']:.2f}, min_q={c2d['practical_window_min_q']:.4f}), "
        f"loss={c2d['asymptotic_ceiling_loss']:.6f}, N4096 SampleZ={c2['practical_grid_eval']['sample_z_oracle_acc'][-1]:.4f} vs Raw={c2['practical_grid_eval']['raw_bon_acc'][-1]:.4f}",
        flush=True,
    )
    c3 = mod_a["case3_filtered_pi0_zero_no_loss"]
    print(
        f"Case3 (pi0=0.0, pi1=0.50): L_tail={c3['eval']['z_tail_posterior_limit']:.4f}, "
        f"N4096 SampleZ={c3['eval']['sample_z_oracle_acc'][-1]:.4f} vs Raw={c3['eval']['raw_bon_acc'][-1]:.4f}",
        flush=True,
    )
    c4 = mod_a["case4_left_tail_overtake_trap"]
    print(
        f"Case4 Left-Tail Trap (pi0=0, pi1=0.10, Delta=15): "
        f"N4096 Raw={c4['unconstrained_two_sided']['raw_bon_acc'][-1]:.4f}, "
        f"Unconstrained SampleZ={c4['unconstrained_two_sided']['sample_z_oracle_acc'][-1]:.4f}, "
        f"Right-Half (Z>=0) SampleZ={c4['right_half_constrained_z_ge_0']['sample_z_oracle_acc'][-1]:.4f}",
        flush=True,
    )

    all_probs_ace = _load_acemath_problems()
    mod_c = rs2.to_json_safe(run_module_c_counterfactual_pi0_ablation(all_probs_ace))
    print("\n=== Module C: Counterfactual pi_0=0 Ablation (seed=835, shuffled i.i.d. Alg1) ===", flush=True)
    for reg_name, reg_val in mod_c["regimes"].items():
        zc = reg_val["cal_z_curve"]
        a1d = reg_val["alg1_within_z_diag"]
        e1 = reg_val["test_envs"]["1_clean_test"]
        e3a = reg_val["test_envs"]["3a_ood_unseen_test"]
        e3b = reg_val["test_envs"]["3b_in_support_unseen_test"]
        b16_20 = a1d["bin16_to_bin20"]
        print(
            f"  {reg_name:23s} | Cont Peak->Tail={zc['peak_lcb']:.4f}->{zc['right_tail_lcb']:.4f} | "
            f"Alg1 Peak->Tail={a1d['peak_lcb']:.4f}->{a1d['right_tail_lcb']:.4f} (bin16->20 p: {b16_20['bin16_mean_p']:.3f}->{b16_20['bin20_mean_p']:.3f}) | "
            f"Cont[Clean={e1['within_z_kl_lcb_acc']:.4f}, 3a={e3a['within_z_kl_lcb_acc']:.4f}, 3b={e3b['within_z_kl_lcb_acc']:.4f}] | "
            f"Alg1[Clean={e1['alg1_within_z_acc']:.4f}, 3a={e3a['alg1_within_z_acc']:.4f}, 3b={e3b['alg1_within_z_acc']:.4f}]",
            flush=True,
        )

    print("\n=== Module C: 8-Seed Mean ± SD Summary (filtered_pi0_zero - size_matched_full) ===", flush=True)
    ms = mod_c["multi_seed_8_summary"]["by_env"]
    for env_name in ("1_clean_test", "3a_ood_unseen_test", "3b_in_support_unseen_test"):
        dc = ms[env_name]["filtered_minus_size_matched"]["continuous_within_z"]
        da = ms[env_name]["filtered_minus_size_matched"]["alg1_within_z"]
        print(
            f"  {env_name:26s} | Cont diff={dc['mean']*100:+.2f}±{dc['sd']*100:.2f} pp | "
            f"Alg1 diff={da['mean']*100:+.2f}±{da['sd']*100:.2f} pp",
            flush=True,
        )

    print("\n=== Module C: 7-Regime Shift Matrix (seed=835, shuffled i.i.d. Alg1) ===", flush=True)
    for r_key, r_val in mod_c["distribution_shift_matrix"].items():
        print(
            f"  {r_key:26s} | sel={r_val['selected_branch_cont']:15s} | Raw={r_val['raw_bon_n64']:.4f} | "
            f"Pool={r_val['pooled_kl_lcb_n64']:.4f} [{r_val['alg1_pooled_kl_lcb_n64']:.4f}] | "
            f"Z={r_val['within_prompt_z_n64']:.4f} [{r_val['alg1_within_prompt_z_n64']:.4f}] | "
            f"Hyb={r_val['hybrid_fixed_n64']:.4f} [{r_val['alg1_hybrid_fixed_n64']:.4f}] | "
            f"Clip={r_val['support_clipped_n64']:.4f} [{r_val['alg1_support_clipped_n64']:.4f}]",
            flush=True,
        )

    mod_b = rs2.to_json_safe(run_module_b_real_benchmarks_boundary_audit(all_probs_ace))
    out = {
        "module_a_general_pi1_theory": mod_a,
        "module_b_real_benchmarks_boundary_audit": mod_b,
        "module_c_counterfactual_pi0_ablation": mod_c,
    }
    out_path = PROJECT_ROOT / "results" / "tmlr_diagnostics.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved TMLR diagnostics JSON:", out_path, flush=True)

    print("\n=== Module B: 5 Benchmark Configurations Audit ===", flush=True)
    for b_key, b_val in mod_b["benchmarks"].items():
        bk = b_val["binned_k20"]
        print(
            f"  {b_val['benchmark']:22s} (P={b_val['n_problems']:4d}): pi0={b_val['pi_0_hat_all_wrong']:.4f}, "
            f"pi1={b_val['pi_1_hat_all_right']:.4f}, tie={b_val['prompt_tie_rate']:.3f}, "
            f"skew+={b_val['z_pos_skewness']:+.3f}, kurt+={b_val['z_pos_excess_kurtosis']:+.2f}, "
            f"q_peak={bk['q_z_peak_hat']:.4f} (bin {bk['peak_bin_1idx']}/{bk['k_effective']}), q_tail={bk['q_z_right_tail_hat']:.4f}, "
            f"bin20_zero={bk['right_tail_zero_prompt_frac']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
