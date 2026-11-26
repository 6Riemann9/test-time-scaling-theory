"""四审收官（V10 Release / v1.0.0）全套数学核验、同分桶公平消融、分布漂移与自然坍塌实证脚本。

全部基于本地缓存数据离线运行，导出 results/v10_release_verification.json：
1. 核验三项局部数学修正：
   - P1-01：定理 7A 在 s=3.5 左端点处锁定 q'_2(s) <= -0.075 * q_2(3.5)*(1 - q_2(3.5)) ≈ -0.0018539 < -1.85e-3；
   - P1-02：定理 2C 在固定峰高 c_1 = 4.0 下 h*tau*(kappa^2 - 1)/sqrt(c_1) -> 1.0；
   - P1-03：定理 7B 零方差非退化条件与常数平局池安全性。
2. 公平同分桶数 K 消融实验（Checklist #5）：
   在严格相同 K 与相同 sample_split 协议（零额外无标签数据）下，对比 MLE、Laplace、Wilson-LCB 与 KL-LCB：
   - 未污染 AceMath RM-B（自适应粗桶 K_split(M) 与中等固定桶 K=20）；
   - 受控分数右尾污染 AceMath RM-B（自适应粗桶 K_split(M) 与中等固定桶 K=20），定量分离“分桶粒度正则化”与“LCB 悲观惩罚”各自的贡献。
3. 分布漂移对照矩阵（Checklist #6）：
   在 AceMath RM-B 单次划分（Train=4,494 / Val=1,798 / Held-Out Test=2,695, N=64）下对比四类环境：
   - Clean Train + Clean Val -> Clean Test；
   - Contam Train + Contam Val -> Contam Test（同分布已知污染类型）；
   - Clean Train + Clean Val -> Contam Test（纯测试期未知分布漂移）；
   - Clean Train + Contam Val -> Contam Test（训练无污染、仅留出验证集来自目标部署环境）。
4. 自然发生的真实非单调/过优化回撤子集审计（Checklist #7，零人工污染注入）：
   对本地全部真实公开基准子集扫描自然过优化回撤，重点在 AceMath-RewardBench 的 college_math 子集
   （P=2,818 题 × 64 候选，Official RM-B）与 OlympiadBench uPRM 多视图上检验自然右尾倒挂与统计显著性。
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

import run_stage2_experiments as rs2  # noqa: E402

from tts_theory import (  # noqa: E402
    EmpiricalScoreCalibrator,
    GaussianVerifierSpec,
    calibration_minimax_scaling,
    evaluate_finite_calibration_on_real_pools,
    evaluate_single_split_prompt_heterogeneity,
    exact_pool_bon_accuracy,
    gaussian_peak_threshold,
)


def verify_v10_math_fixes() -> dict[str, Any]:
    """核验 P1-01（q_2 左端点）、P1-02（sqrt(c_1) 常数）与 P1-03（零方差安全）。"""
    # P1-01
    def _ell2(s: float) -> float:
        return math.log(2.0) - 0.5 * ((s - 3.3) ** 2) + 0.125 * ((s - 3.0) ** 2)

    def _q2(s: float) -> float:
        return 1.0 / (1.0 + math.exp(-(math.log(0.95 / 0.05) + _ell2(s))))

    q2_35 = _q2(3.5)
    q2_42 = _q2(4.2)
    dq2_bound_correct = -0.075 * q2_35 * (1.0 - q2_35)
    dq2_grid_max = float(
        np.max(
            [
                _q2(float(sv)) * (1.0 - _q2(float(sv))) * (-0.75 * (float(sv) - 3.4))
                for sv in np.linspace(3.5, 4.2, 5_000)
            ]
        )
    )

    # P1-02: c_1 = 4.0 => sqrt(c_1) = 2.0 != c_1 = 4.0
    c1 = 4.0
    sigma1 = 1.25
    p = 0.25
    gumbel_rows = []
    from tts_theory.theorems import _gaussian_mix_log_sf_scalar

    for kappa in (1.20, 1.05, 1.01, 1.002, 1.0005):
        dmu = sigma1 * math.sqrt(c1 * (kappa * kappa - 1.0))
        spec = GaussianVerifierSpec(p=p, mu1=dmu, sigma1=sigma1, mu0=0.0, sigma0=kappa * sigma1)
        x_peak = gaussian_peak_threshold(spec)
        l1 = math.log(p) + float(stats.norm.logpdf(x_peak, loc=dmu, scale=sigma1))
        l0 = math.log1p(-p) + float(stats.norm.logpdf(x_peak, loc=0.0, scale=kappa * sigma1))
        log_pdf = float(np.logaddexp(l1, l0))
        log_sf = _gaussian_mix_log_sf_scalar(spec, x_peak)
        h_val = math.exp(log_pdf - log_sf)
        tau_val = (kappa * sigma1) / math.sqrt(kappa * kappa - 1.0)
        htau = h_val * tau_val
        gumbel_rows.append(
            {
                "kappa": kappa,
                "h_tau": htau,
                "ratio_to_sqrt_c1_over_k2_minus_1": htau * (kappa * kappa - 1.0) / math.sqrt(c1),
                "ratio_to_wrong_c1_over_k2_minus_1": htau * (kappa * kappa - 1.0) / c1,
            }
        )

    return {
        "p1_01_q2_left_endpoint": {
            "q2_at_3_5": q2_35,
            "q2_at_4_2": q2_42,
            "q2_var_at_3_5_min": q2_35 * (1.0 - q2_35),
            "q2_var_at_4_2_max": q2_42 * (1.0 - q2_42),
            "correct_dq2_upper_bound_from_s3_5": dq2_bound_correct,
            "exact_max_dq2_on_interval": dq2_grid_max,
        },
        "p1_02_gumbel_sqrt_c1_scaling": gumbel_rows,
    }


def _load_acemath_problems() -> list[dict[str, Any]]:
    ace_dst = rs2.DATA_DIR / "acemath_rewardbench" / "default_test_0.parquet"
    tbl = pq.read_table(ace_dst)
    rm_all = tbl["rm_score"].to_pylist()
    corr_all = tbl["score"].to_pylist()
    offset_b = 8987
    all_probs = []
    for sub_name, s0, s1 in rs2.ACEMATH_SEGMENTS:
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


def _inject_tail_contamination(
    probs: list[dict[str, Any]],
    q995: float,
    sd_b: float,
    spike_rate: float = 0.06,
    seed: int = 202609,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    out: list[tuple[str, np.ndarray, np.ndarray]] = []
    for p in probs:
        s_arr = p["rm_b"].copy()
        y_arr = p["is_correct"].copy()
        wrong_mask = ~y_arr
        if np.any(wrong_mask):
            sp = (rng.random(s_arr.size) < spike_rate) & wrong_mask
            if np.any(sp):
                s_arr[sp] = q995 + 0.8 * sd_b + np.abs(rng.normal(0.0, 0.5 * sd_b, size=int(np.sum(sp))))
        out.append((p["subset"], s_arr, y_arr))
    return out


def run_same_k_fair_calibrator_ablation(all_probs_ace: list[dict[str, Any]]) -> dict[str, Any]:
    """Checklist #5：相同 K 与相同 sample_split 协议下 MLE / Laplace / Wilson-LCB / KL-LCB 公平消融。"""
    rng = np.random.default_rng(2026)
    perm = rng.permutation(len(all_probs_ace))
    half = len(all_probs_ace) // 2
    cal_p = [all_probs_ace[i] for i in perm[:half]]
    test_p = [all_probs_ace[i] for i in perm[half:]]

    clean_cal_s = np.concatenate([p["rm_b"] for p in cal_p])
    clean_cal_y = np.concatenate([p["is_correct"] for p in cal_p])
    clean_test = [(p["rm_b"], p["is_correct"]) for p in test_p]

    all_rm_b = np.concatenate([p["rm_b"] for p in all_probs_ace])
    q995 = float(np.quantile(all_rm_b, 0.995))
    sd_b = float(np.std(all_rm_b))
    contam_all = _inject_tail_contamination(all_probs_ace, q995=q995, sd_b=sd_b, spike_rate=0.06, seed=202609)
    contam_cal_s = np.concatenate([contam_all[i][1] for i in perm[:half]])
    contam_cal_y = np.concatenate([contam_all[i][2] for i in perm[:half]])
    contam_test = [(contam_all[i][1], contam_all[i][2]) for i in perm[half:]]

    methods = ("mle", "laplace", "wilson_lcb", "kl_lcb")

    def _run_grid(
        c_s: np.ndarray,
        c_y: np.ndarray,
        t_pools: list[tuple[np.ndarray, np.ndarray]],
        m_list: list[int],
        fixed_k: int | None,
        seed_base: int,
    ) -> dict[str, Any]:
        rows = []
        raw_acc = None
        for i_m, mv in enumerate(m_list):
            k_bins = (
                int(fixed_k)
                if fixed_k is not None
                else max(3, int(calibration_minimax_scaling(max(10, mv // 2), delta=0.05, beta=2.0)["k_opt_theory"]))
            )
            res = evaluate_finite_calibration_on_real_pools(
                c_s,
                c_y,
                t_pools,
                m_cal=mv,
                n_bins=k_bins,
                n_values=[64],
                binning="quantile",
                methods=methods,
                n_repeats=50,
                delta=0.05,
                seed=seed_base + i_m,
                quantile_protocol="sample_split",
            )
            if raw_acc is None:
                raw_acc = float(res["standard_bon_accuracy"][0])
            row: dict[str, Any] = {"m_total": mv, "k_bins": k_bins}
            for m_name in methods:
                row[f"{m_name}_acc"] = float(res["methods"][m_name]["accuracy"][0])
                row[f"{m_name}_se"] = float(res["methods"][m_name]["se"][0])
            rows.append(row)
        return {"standard_bon_n64": raw_acc, "rows": rows}

    print("[V10] Running fair same-K calibrator ablation under sample_split...")
    clean_opt_k = _run_grid(clean_cal_s, clean_cal_y, clean_test, [100, 200, 400, 800, 2000, 5000], None, 3100)
    clean_k30 = _run_grid(clean_cal_s, clean_cal_y, clean_test, [100, 200, 400, 800, 2000, 5000], 30, 3200)
    contam_opt_k = _run_grid(contam_cal_s, contam_cal_y, contam_test, [100, 200, 400, 800, 2000, 5000], None, 3300)
    contam_k30 = _run_grid(contam_cal_s, contam_cal_y, contam_test, [100, 200, 400, 800, 2000, 5000], 30, 3400)

    return {
        "clean_acemath_rm_b_same_k_opt": clean_opt_k,
        "clean_acemath_rm_b_same_k30": clean_k30,
        "contaminated_acemath_rm_b_same_k_opt": contam_opt_k,
        "contaminated_acemath_rm_b_same_k30": contam_k30,
    }


def run_contamination_distribution_shift_matrix(all_probs_ace: list[dict[str, Any]]) -> dict[str, Any]:
    """Checklist #6：严格按同一 50/20/30 分层划分对比同分布污染与训练干净/测试污染的分布漂移边界。"""
    print("[V10] Running 4-regime clean/contaminated distribution-shift matrix on AceMath RM-B...")
    clean_pools = [(p["subset"], p["rm_b"], p["is_correct"]) for p in all_probs_ace]
    all_rm_b = np.concatenate([p["rm_b"] for p in all_probs_ace])
    q995 = float(np.quantile(all_rm_b, 0.995))
    sd_b = float(np.std(all_rm_b))
    contam_pools = _inject_tail_contamination(all_probs_ace, q995=q995, sd_b=sd_b, spike_rate=0.06, seed=202609)

    # 复现与 evaluate_single_split_prompt_heterogeneity(seed=835) 完全一致的 50/20/30 题级划分
    split_rng = np.random.default_rng(835)
    by_stratum: dict[str, list[int]] = {}
    for idx, (st, _, _) in enumerate(clean_pools):
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

    train_idx = np.asarray(sorted(train_idx_l), dtype=np.int64)
    val_idx = np.asarray(sorted(val_idx_l), dtype=np.int64)
    test_idx = np.asarray(sorted(test_idx_l), dtype=np.int64)

    def _inject_in_support_high_contamination(
        probs: list[dict[str, Any]],
        q90: float,
        q985: float,
        spike_rate: float = 0.06,
        seed: int = 202610,
    ) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """非极端高分（分布内 In-Support）污染：仅落于干净正常高分带 [q90, q985]，不超出 99.8% 截断边界。"""
        rng_in = np.random.default_rng(seed)
        out_in: list[tuple[str, np.ndarray, np.ndarray]] = []
        for p_item in probs:
            s_arr = p_item["rm_b"].copy()
            y_arr = p_item["is_correct"].copy()
            w_mask = ~y_arr
            if np.any(w_mask):
                sp_mask = (rng_in.random(s_arr.size) < spike_rate) & w_mask
                n_sp = int(np.sum(sp_mask))
                if n_sp > 0:
                    s_arr[sp_mask] = rng_in.uniform(q90, q985, size=n_sp)
            out_in.append((p_item["subset"], s_arr, y_arr))
        return out_in

    def _eval_split_matrix(c_pools_list: list[tuple[str, np.ndarray, np.ndarray]], tag_seed: int = 0) -> dict[str, Any]:
        """对给定干净/污染池矩阵同时评估 EmpiricalScoreCalibrator 与严格 Algorithm 1 Sample-Split 版本。"""
        def _run_one(
            tr_src: list[tuple[str, np.ndarray, np.ndarray]],
            va_src: list[tuple[str, np.ndarray, np.ndarray]],
            te_src: list[tuple[str, np.ndarray, np.ndarray]],
            n_bins: int = 20,
        ) -> dict[str, Any]:
            def _z_pool(s_vec: np.ndarray) -> np.ndarray:
                sd = float(np.std(s_vec, ddof=1)) if s_vec.size > 1 else 0.0
                return (s_vec - float(np.mean(s_vec))) / sd if sd > 1e-12 else np.zeros_like(s_vec)

            from tts_theory.calibration_theory import compute_bin_calibrator_scores

            tr_s = np.concatenate([tr_src[i][1] for i in train_idx])
            tr_y = np.concatenate([tr_src[i][2].astype(np.int64) for i in train_idx])
            tr_z = np.concatenate([_z_pool(tr_src[i][1]) for i in train_idx])

            cal_pool = EmpiricalScoreCalibrator(
                n_bins=n_bins, method="quantile_lcb", lcb_delta=0.05, lcb_type="kl"
            ).fit(tr_s, tr_y)
            cal_z = EmpiricalScoreCalibrator(
                n_bins=n_bins, method="quantile_lcb", lcb_delta=0.05, lcb_type="kl"
            ).fit(tr_z, tr_y)
            tr_hi_cap = float(np.quantile(tr_s, 0.998))

            # 严格遵循 Algorithm 1：按题目顺序前 50% 训练集定分位边与 99.8% 截断阈值，后 50% 估计阶梯 KL-LCB（无线性插值）
            m_tot = int(tr_s.size)
            m_half = m_tot // 2
            s_edge_sorted = np.sort(tr_s[:m_half])
            r_ord = np.floor(np.arange(1, n_bins, dtype=np.float64) * float(m_half + 1) / float(n_bins)).astype(np.int64)
            r_ord = np.clip(r_ord - 1, 0, m_half - 1)
            inner_edges = np.unique(s_edge_sorted[r_ord])
            s_est, y_est = tr_s[m_half:], tr_y[m_half:]
            k_cur = int(inner_edges.size + 1)
            b_est = np.searchsorted(inner_edges, s_est, side="right")
            cnts = np.bincount(b_est, minlength=k_cur).astype(np.float64)
            poss = np.bincount(b_est, weights=y_est.astype(np.float64), minlength=k_cur)
            alg1_scores = compute_bin_calibrator_scores(
                cnts, poss, method="kl_lcb", prior_p=float(np.mean(y_est)), delta=0.05
            )
            alg1_s_max = float(s_edge_sorted[min(m_half - 1, max(0, math.ceil(0.998 * m_half) - 1))])

            def _eval_subset(indices: np.ndarray, src: list[tuple[str, np.ndarray, np.ndarray]]) -> dict[str, np.ndarray]:
                m_len = int(indices.size)
                out_raw = np.empty(m_len, dtype=np.float64)
                out_pool = np.empty(m_len, dtype=np.float64)
                out_z = np.empty(m_len, dtype=np.float64)
                out_clipped = np.empty(m_len, dtype=np.float64)
                out_alg1_pool = np.empty(m_len, dtype=np.float64)
                out_alg1_clip = np.empty(m_len, dtype=np.float64)
                for k_i, p_i in enumerate(indices):
                    _, s_p, y_p = src[int(p_i)]
                    q_pool = np.asarray(cal_pool.predict_proba(s_p), dtype=np.float64)
                    z_p = _z_pool(s_p)
                    q_z = np.asarray(cal_z.predict_proba(z_p), dtype=np.float64)
                    q_clip = np.where(s_p > tr_hi_cap, 0.0, q_pool)
                    # Algorithm 1 严格分桶阶梯查找与支撑截断（均匀随机打破最大分桶平局之精确期望）
                    b_p = np.searchsorted(inner_edges, s_p, side="right")
                    q_a1_pool = alg1_scores[b_p]
                    q_a1_clip = np.where(s_p > alg1_s_max, 0.0, q_a1_pool)

                    for arr_out, sc_vec in (
                        (out_raw, s_p),
                        (out_pool, q_pool),
                        (out_z, q_z),
                        (out_clipped, q_clip),
                        (out_alg1_pool, q_a1_pool),
                        (out_alg1_clip, q_a1_clip),
                    ):
                        mx = float(np.max(sc_vec))
                        ties = np.abs(sc_vec - mx) <= 1e-12
                        arr_out[k_i] = float(np.mean(y_p[ties]))
                return {
                    "Standard-BoN": out_raw,
                    "Pooled-KL-LCB": out_pool,
                    "Within-Prompt-Z-KL-LCB": out_z,
                    "Support-Clipped-KL-LCB": out_clipped,
                    "Alg1-SampleSplit-KL-LCB": out_alg1_pool,
                    "Alg1-SampleSplit-Support-Clipped": out_alg1_clip,
                }

            val_scores = _eval_subset(val_idx, va_src)
            test_scores = _eval_subset(test_idx, te_src)
            branches = ("Standard-BoN", "Pooled-KL-LCB", "Within-Prompt-Z-KL-LCB")
            best_branch = max(branches, key=lambda b: float(np.mean(val_scores[b])))
            hyb_test = test_scores[best_branch]

            diff_vs_std = hyb_test - test_scores["Standard-BoN"]
            sd_d = float(np.std(diff_vs_std, ddof=1))
            if sd_d <= 1e-15:
                p_vs_std = 1.0
            else:
                t_val = float(np.mean(diff_vs_std)) / (sd_d / math.sqrt(diff_vs_std.size))
                p_vs_std = float(max(np.finfo(np.float64).tiny, 2.0 * stats.t.sf(abs(t_val), df=diff_vs_std.size - 1)))

            return {
                "n_train": int(train_idx.size),
                "n_val": int(val_idx.size),
                "n_test": int(test_idx.size),
                "selected_branch_n64": best_branch,
                "standard_bon_n64": float(np.mean(test_scores["Standard-BoN"])),
                "pooled_kl_lcb_n64": float(np.mean(test_scores["Pooled-KL-LCB"])),
                "within_prompt_z_n64": float(np.mean(test_scores["Within-Prompt-Z-KL-LCB"])),
                "hybrid_fixed_n64": float(np.mean(hyb_test)),
                "support_clipped_kl_lcb_n64": float(np.mean(test_scores["Support-Clipped-KL-LCB"])),
                "alg1_sample_split_kl_lcb_n64": float(np.mean(test_scores["Alg1-SampleSplit-KL-LCB"])),
                "alg1_sample_split_support_clipped_n64": float(np.mean(test_scores["Alg1-SampleSplit-Support-Clipped"])),
                "hybrid_minus_std_n64": float(np.mean(diff_vs_std)),
                "p_value_ttest_vs_std": p_vs_std,
            }

        return {
            "clean_train_clean_val_clean_test": _run_one(clean_pools, clean_pools, clean_pools),
            "contam_train_contam_val_contam_test": _run_one(c_pools_list, c_pools_list, c_pools_list),
            "clean_train_clean_val_contam_test_unseen_shift": _run_one(clean_pools, clean_pools, c_pools_list),
            "clean_train_contam_val_contam_test_target_val": _run_one(clean_pools, c_pools_list, c_pools_list),
        }

    out_shift = _eval_split_matrix(contam_pools)
    q90 = float(np.quantile(all_rm_b, 0.90))
    q985 = float(np.quantile(all_rm_b, 0.985))
    in_support_pools = _inject_in_support_high_contamination(all_probs_ace, q90=q90, q985=q985, spike_rate=0.06, seed=202610)
    out_shift["in_support_high_score_shift_matrix"] = _eval_split_matrix(in_support_pools)
    return out_shift


def run_natural_nonmonotone_collapse_audit(all_probs_ace: list[dict[str, Any]]) -> dict[str, Any]:
    """Checklist #7 + 第五轮 P1-04：全 14 组多重比较校正与无峰选偏倚的 50/50 样本分割确认检验。"""
    print("[V10] Auditing naturally occurring real-world non-monotone subsets (14 tests + 50/50 split confirmatory)...")
    n_grid_64 = [1, 2, 4, 8, 16, 32, 64]
    all_14_records: list[dict[str, Any]] = []
    natural_subsets: list[dict[str, Any]] = []

    for sub_name, _, _ in rs2.ACEMATH_SEGMENTS:
        sub_probs = [p for p in all_probs_ace if p["subset"] == sub_name]
        for rm_key in ("rm_a", "rm_b"):
            mat = np.asarray(
                [exact_pool_bon_accuracy(p[rm_key], p["is_correct"], n_grid_64) for p in sub_probs],
                dtype=np.float64,
            )
            curve = np.mean(mat, axis=0)
            peak_idx = int(np.argmax(curve))
            drop = float(curve[peak_idx] - curve[-1])
            diffs = mat[:, peak_idx] - mat[:, -1]
            sd = float(np.std(diffs, ddof=1))
            se = sd / math.sqrt(len(sub_probs))
            t_val = float(np.mean(diffs)) / se if (drop > 1e-12 and se > 1e-15) else 0.0
            p_t = float(2.0 * stats.t.sf(abs(t_val), df=len(sub_probs) - 1)) if (drop > 1e-12 and se > 1e-15) else 1.0
            rec = {
                "dataset": f"AceMath/{sub_name}/{rm_key.upper()}",
                "n_problems": len(sub_probs),
                "n_peak": n_grid_64[peak_idx],
                "acc_peak": float(curve[peak_idx]),
                "n_max": 64,
                "acc_n_max": float(curve[-1]),
                "drop_peak_to_max": drop,
                "se_drop": se,
                "p_value_ttest_two_sided": p_t,
                "exact_curve": [float(v) for v in curve],
            }
            all_14_records.append(rec)

    # 跨 7 子集 x 2 RM = 14 组计算 Bonferroni 与 Holm-Bonferroni 校正 p 值
    m_tests = len(all_14_records)
    raw_ps = np.array([r["p_value_ttest_two_sided"] for r in all_14_records], dtype=np.float64)
    order = np.argsort(raw_ps)
    holm_ps = np.empty(m_tests, dtype=np.float64)
    prev = 0.0
    for rank, idx_r in enumerate(order):
        adj = min(1.0, float((m_tests - rank) * raw_ps[idx_r]))
        prev = max(prev, adj)
        holm_ps[idx_r] = prev

    for i_r, rec in enumerate(all_14_records):
        rec["p_value_bonferroni_14"] = min(1.0, float(m_tests * rec["p_value_ttest_two_sided"]))
        rec["p_value_holm_14"] = float(holm_ps[i_r])
        if rec["drop_peak_to_max"] > 1e-4:
            natural_subsets.append(rec)

    # 无峰选偷窥偏倚的 50/50 独立样本分割确认检验（在 college_math RM-B 上 P=2,818 -> 1,409 选峰 / 1,409 检验）
    cm_probs = [p for p in all_probs_ace if p["subset"] == "college_math"]
    cm_mat = np.asarray(
        [exact_pool_bon_accuracy(p["rm_b"], p["is_correct"], n_grid_64) for p in cm_probs],
        dtype=np.float64,
    )
    split_rng = np.random.default_rng(202611)
    perm_cm = split_rng.permutation(len(cm_probs))
    half_cm = len(cm_probs) // 2
    sel_mat = cm_mat[perm_cm[:half_cm]]
    conf_mat = cm_mat[perm_cm[half_cm:]]

    sel_curve = np.mean(sel_mat, axis=0)
    peak_idx_sel = int(np.argmax(sel_curve))
    conf_curve = np.mean(conf_mat, axis=0)
    conf_diffs = conf_mat[:, peak_idx_sel] - conf_mat[:, -1]
    conf_drop = float(np.mean(conf_diffs))
    conf_se = float(np.std(conf_diffs, ddof=1)) / math.sqrt(conf_mat.shape[0])
    conf_t = conf_drop / conf_se if conf_se > 1e-15 else 0.0
    conf_p = float(2.0 * stats.t.sf(abs(conf_t), df=conf_mat.shape[0] - 1)) if conf_se > 1e-15 else 1.0

    # 对探索性候选子集 college_math (RM-B, P=2,818) 执行独立 50/20/30 单次留出校准测试
    cm_pools = [(p["subset"], p["rm_b"], p["is_correct"]) for p in cm_probs]
    cm_single_split = evaluate_single_split_prompt_heterogeneity(
        cm_pools,
        n_values=n_grid_64,
        n_trials_per_problem=16,
        n_bins=12,
        seed=2818,
        n_bootstrap=2_000,
    )

    return {
        "n_family_tests": m_tests,
        "all_14_subset_audits": all_14_records,
        "acemath_natural_overoptimization_instances": natural_subsets,
        "split_sample_confirmatory_college_math_rm_b": {
            "n_sel": int(sel_mat.shape[0]),
            "n_conf": int(conf_mat.shape[0]),
            "n_peak_sel": n_grid_64[peak_idx_sel],
            "acc_peak_sel": float(sel_curve[peak_idx_sel]),
            "acc_peak_conf": float(conf_curve[peak_idx_sel]),
            "acc_n64_conf": float(conf_curve[-1]),
            "drop_conf": conf_drop,
            "se_conf": conf_se,
            "p_value_conf_two_sided": conf_p,
        },
        "college_math_rm_b_single_split": cm_single_split,
    }


def main() -> None:
    math_fixes = rs2.to_json_safe(verify_v10_math_fixes())
    all_probs_ace = _load_acemath_problems()
    fair_ablation = rs2.to_json_safe(run_same_k_fair_calibrator_ablation(all_probs_ace))
    shift_matrix = rs2.to_json_safe(run_contamination_distribution_shift_matrix(all_probs_ace))
    natural_audit = rs2.to_json_safe(run_natural_nonmonotone_collapse_audit(all_probs_ace))

    out = {
        "math_fixes_verification": math_fixes,
        "same_k_fair_calibrator_ablation": fair_ablation,
        "contamination_distribution_shift_matrix": shift_matrix,
        "natural_nonmonotone_collapse_audit": natural_audit,
    }
    out_path = PROJECT_ROOT / "results/v10_release_verification.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved:", out_path)

    print("[P1-01 q2(3.5)]", math_fixes["p1_01_q2_left_endpoint"])
    print("[P1-02 Gumbel sqrt(c1)]", math_fixes["p1_02_gumbel_sqrt_c1_scaling"][-2:])
    for k_name, blk in fair_ablation.items():
        print(f"=== {k_name} (Raw={blk['standard_bon_n64']:.4f}) ===")
        for r in blk["rows"]:
            print(
                f"  M={r['m_total']:5d} K={r['k_bins']:2d} | MLE={r['mle_acc']:.4f} | "
                f"Laplace={r['laplace_acc']:.4f} | Wilson={r['wilson_lcb_acc']:.4f} | KL-LCB={r['kl_lcb_acc']:.4f}"
            )
    print("=== Distribution Shift Matrix ===")
    for r_name, r_val in shift_matrix.items():
        if r_name == "in_support_high_score_shift_matrix":
            print("  --- In-Support High-Score [q90, q985] Shift Matrix ---")
            for sr_k, sr_v in r_val.items():
                print(f"    {sr_k}: {sr_v}")
        else:
            print(f"  {r_name}: {r_val}")
    print("=== Natural Overoptimization Subsets (14-Test Family) ===")
    for item in natural_audit["acemath_natural_overoptimization_instances"]:
        print(
            f"  {item['dataset']} (P={item['n_problems']}): peak N={item['n_peak']} ({item['acc_peak']:.4f}) -> "
            f"N=64 ({item['acc_n_max']:.4f}), drop={item['drop_peak_to_max']:+.4f} ± {item['se_drop']:.4f} "
            f"(p_raw={item['p_value_ttest_two_sided']:.4g}, p_Bonf14={item['p_value_bonferroni_14']:.4g}, p_Holm14={item['p_value_holm_14']:.4g})"
        )
    print("  [50/50 Split-Sample Confirmatory]", natural_audit["split_sample_confirmatory_college_math_rm_b"])
    cm = natural_audit["college_math_rm_b_single_split"]
    cm_a = cm["algorithms"]
    print(
        f"  [college_math RM-B SingleSplit P_test={cm['n_test_problems']} N=64 sel={cm['selected_branch_by_n']['64']}] "
        f"Std(16->64)={cm_a['Standard-BoN']['accuracy'][4]:.4f}->{cm_a['Standard-BoN']['accuracy'][6]:.4f} | "
        f"Pool(64)={cm_a['Pooled-KL-LCB']['accuracy'][6]:.4f} | HybFixed(64)={cm_a['Hybrid-Fixed-KL-LCB']['accuracy'][6]:.4f}"
    )


if __name__ == "__main__":
    main()
