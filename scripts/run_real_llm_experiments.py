"""真实大模型与真实开源 Reward Model 端到端实证实验脚本（V3）。

自动下载并本地缓存两个真实公开 HuggingFace 基准（已存在则直接复用）：
1. `nvidia/AceMath-RewardBench` 官方预打分 Parquet（8,987 题 × 64 候选 = 575,168 条真实解答，
   涵盖 7 大数学基准：college_math, gaokao2023en, gsm8k, math500, minerva_math, mmlu_stem, olympiadbench，
   并自带两套独立真实开源 Reward Model 打分 RM-A 与 RM-B）。
2. `HuggingFaceH4/Llama-3.2-1B-Instruct-best-of-N-completions` 与 `Llama-3.2-3B-Instruct-best-of-N-completions`
   （MATH-500 上每题 256 条真实解答 + RLHFlow-8B-PRM 逐步分数，N=1..256）。

执行实验 R1–R4 并将全部真实实证结果写入 `results/real_llm_results.json`。
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
import numpy as np
import pyarrow.parquet as pq
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tts_theory.real_eval import (
    diagnose_multi_verifier_correlations,
    diagnose_real_verifier_distribution,
    evaluate_discrete_algorithms_on_real_pools,
    evaluate_finite_calibration_on_real_pools,
    exact_pool_bon_accuracy,
    extract_boxed_answer,
    is_math_answer_correct,
    normalize_math_answer,
)

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

ACEMATH_SEGMENTS = (
    ("college_math", 0, 2818),
    ("gaokao2023en", 2818, 3203),
    ("gsm8k", 3203, 4522),
    ("math500", 4522, 5022),
    ("minerva_math", 5022, 5294),
    ("mmlu_stem", 5294, 8312),
    ("olympiadbench", 8312, 8987),
)


def to_json_safe(obj: Any) -> Any:
    """将 NumPy 标量与数组递归转为原生 JSON 类型。"""
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return None if (np.isnan(val) or np.isinf(val)) else val
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def download_acemath_scored_parquet() -> Path:
    """下载或复用 nvidia/AceMath-RewardBench 带 RM-A / RM-B 预打分的 Parquet 文件 (24.6 MB)。"""
    dst = DATA_DIR / "acemath_rewardbench" / "default_test_0.parquet"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        print("正在下载 nvidia/AceMath-RewardBench 官方预打分 Parquet (24.6 MB) ...")
        url = "https://huggingface.co/api/datasets/nvidia/AceMath-RewardBench/parquet/default/test/0.parquet"
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        dst.write_bytes(r.content)
    return dst


def download_hf_h4_completions() -> dict[str, dict[str, Path]]:
    """下载或复用 Llama-3.2-1B / 3B MATH-500 (n=256, seed=0) Parquet 及官方 Evals 文件。"""
    repos = {
        "llama_1b": "HuggingFaceH4/Llama-3.2-1B-Instruct-best-of-N-completions",
        "llama_3b": "HuggingFaceH4/Llama-3.2-3B-Instruct-best-of-N-completions",
    }
    base_cfg = "HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-256--max_tokens-2048--bsz-8--seed-0--agg_strategy-last"
    out: dict[str, dict[str, Path]] = {}
    for key, repo_id in repos.items():
        local_root = DATA_DIR / f"hf_h4_{key}"
        local_root.mkdir(parents=True, exist_ok=True)
        main_rel = f"{base_cfg}/train-00000-of-00001.parquet"
        eval_rel = f"{base_cfg}--evals/train-00000-of-00001.parquet"
        for rel in (main_rel, eval_rel):
            dst = local_root / rel
            if not dst.exists():
                print(f"正在下载 {repo_id}/{rel} ...")
                hf_hub_download(
                    repo_id=repo_id,
                    filename=rel,
                    repo_type="dataset",
                    local_dir=str(local_root),
                )
        out[key] = {"main": local_root / main_rel, "evals": local_root / eval_rel}
    return out


def load_acemath_dual_rm(parquet_path: Path) -> dict[str, list[dict[str, Any]]]:
    """加载 AceMath-RewardBench 7 大子集及双独立开源 Reward Model（RM-A 与 RM-B）打分。"""
    tbl = pq.read_table(parquet_path)
    rm_all = tbl["rm_score"].to_pylist()
    pred_all = tbl["pred"].to_pylist()
    corr_all = tbl["score"].to_pylist()

    offset_b = 8987
    by_subset: dict[str, list[dict[str, Any]]] = {}
    for sub_name, s0, s1 in ACEMATH_SEGMENTS:
        probs = []
        for idx in range(s0, s1):
            rm_a = np.asarray(rm_all[idx], dtype=np.float64)
            rm_b = np.asarray(rm_all[offset_b + idx], dtype=np.float64)
            y = np.asarray(corr_all[idx], dtype=bool)
            preds = [str(x) if x is not None else "" for x in pred_all[idx]]
            probs.append({
                "subset": sub_name,
                "rm_a": rm_a,
                "rm_b": rm_b,
                "is_correct": y,
                "preds": preds,
            })
        by_subset[sub_name] = probs
    return by_subset


def run_acemath_experiments(by_subset: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """运行实验 R1（7 大基准分布诊断）、R2（定理 6 有限校准与算法对比）及跨 RM 定理 5 验证。"""
    n_values = [1, 2, 4, 8, 16, 32, 64]
    r1_benchmarks: dict[str, Any] = {}
    all_problems: list[dict[str, Any]] = []

    for sub, probs in by_subset.items():
        diag_a = diagnose_real_verifier_distribution(
            [p["rm_a"] for p in probs], [p["is_correct"] for p in probs]
        )
        diag_b = diagnose_real_verifier_distribution(
            [p["rm_b"] for p in probs], [p["is_correct"] for p in probs]
        )
        curves_a = np.array([exact_pool_bon_accuracy(p["rm_a"], p["is_correct"], n_values) for p in probs])
        curves_b = np.array([exact_pool_bon_accuracy(p["rm_b"], p["is_correct"], n_values) for p in probs])
        r1_benchmarks[sub] = {
            "rm_a_diag": diag_a,
            "rm_b_diag": diag_b,
            "n_values": n_values,
            "rm_a_bon_accuracy": np.mean(curves_a, axis=0),
            "rm_b_bon_accuracy": np.mean(curves_b, axis=0),
        }
        all_problems.extend(probs)

    # 真实双独立 RM（RM-A + RM-B）相关性矩阵与定理 5 融合验证
    all_a = np.concatenate([p["rm_a"] for p in all_problems])
    all_b = np.concatenate([p["rm_b"] for p in all_problems])
    all_y = np.concatenate([p["is_correct"] for p in all_problems])
    dual_corr = diagnose_multi_verifier_correlations({"RM-A": all_a, "RM-B": all_b}, all_y)

    mu_a, sd_a = float(np.mean(all_a)), max(1e-8, float(np.std(all_a)))
    mu_b, sd_b = float(np.mean(all_b)), max(1e-8, float(np.std(all_b)))
    ens_curves = {k: np.zeros(len(n_values)) for k in ("RM-A", "RM-B", "Mean-Ensemble (A+B)", "Min-Ensemble (A+B)")}
    for p in all_problems:
        za = (p["rm_a"] - mu_a) / sd_a
        zb = (p["rm_b"] - mu_b) / sd_b
        ens_curves["RM-A"] += exact_pool_bon_accuracy(p["rm_a"], p["is_correct"], n_values)
        ens_curves["RM-B"] += exact_pool_bon_accuracy(p["rm_b"], p["is_correct"], n_values)
        ens_curves["Mean-Ensemble (A+B)"] += exact_pool_bon_accuracy(0.5 * (za + zb), p["is_correct"], n_values)
        ens_curves["Min-Ensemble (A+B)"] += exact_pool_bon_accuracy(np.minimum(za, zb), p["is_correct"], n_values)
    for k in ens_curves:
        ens_curves[k] = ens_curves[k] / float(len(all_problems))

    # 严格按题 50%/50% 划分校准题库与留出测试题库（以更具挑战的 RM-B 作为校准主评估，同时保留全部 7 大基准）
    rng = np.random.default_rng(2026)
    perm = rng.permutation(len(all_problems))
    half = len(all_problems) // 2
    cal_probs = [all_problems[i] for i in perm[:half]]
    test_probs = [all_problems[i] for i in perm[half:]]

    cal_scores = np.concatenate([p["rm_b"] for p in cal_probs])
    cal_correct = np.concatenate([p["is_correct"] for p in cal_probs])
    test_pools = [(p["rm_b"], p["is_correct"]) for p in test_probs]
    test_pools_ans = [(p["rm_b"], p["is_correct"], p["preds"]) for p in test_probs]

    # R2a: M=400 有限校准集实验（等宽 K=20 与 等频 K=40）
    r2_m400_uniform = evaluate_finite_calibration_on_real_pools(
        cal_scores, cal_correct, test_pools,
        m_cal=400, n_bins=20, n_values=n_values,
        binning="uniform", n_repeats=150, delta=0.05, seed=42,
    )
    r2_m400_quantile = evaluate_finite_calibration_on_real_pools(
        cal_scores, cal_correct, test_pools,
        m_cal=400, n_bins=40, n_values=n_values,
        binning="quantile", n_repeats=150, delta=0.05, seed=43,
    )

    # R2b: 跨 M 扫描（N=64 下对比 Optimal K*(M)+Wilson-LCB vs Fixed K=60+MLE vs Standard-BoN）
    m_grid = [200, 400, 800, 2000, 5000, 15000, 40000]
    m_sweep_records = []
    std_bon_64 = float(r2_m400_uniform["standard_bon_accuracy"][-1])
    for idx_m, m_val in enumerate(m_grid):
        k_opt = max(5, int(round(2.0 * (m_val / max(1.0, math.log(m_val))) ** 0.4)))
        res_opt = evaluate_finite_calibration_on_real_pools(
            cal_scores, cal_correct, test_pools,
            m_cal=m_val, n_bins=k_opt, n_values=[64],
            binning="quantile", methods=("wilson_lcb", "mle"),
            n_repeats=100, delta=0.05, seed=100 + idx_m,
        )
        res_fix = evaluate_finite_calibration_on_real_pools(
            cal_scores, cal_correct, test_pools,
            m_cal=m_val, n_bins=60, n_values=[64],
            binning="quantile", methods=("mle", "wilson_lcb"),
            n_repeats=100, delta=0.05, seed=200 + idx_m,
        )
        m_sweep_records.append({
            "m_cal": m_val,
            "k_opt": k_opt,
            "standard_bon_n64": std_bon_64,
            "opt_wilson_n64": float(res_opt["methods"]["wilson_lcb"]["accuracy"][0]),
            "opt_wilson_se": float(res_opt["methods"]["wilson_lcb"]["se"][0]),
            "opt_mle_n64": float(res_opt["methods"]["mle"]["accuracy"][0]),
            "fixed60_mle_n64": float(res_fix["methods"]["mle"]["accuracy"][0]),
            "fixed60_mle_se": float(res_fix["methods"]["mle"]["se"][0]),
            "fixed60_wilson_n64": float(res_fix["methods"]["wilson_lcb"]["accuracy"][0]),
        })

    # R2c: 真实提取答案下的离散抗坍塌算法对比（每题 25 次，4,494 × 25 = 112,350 次试验）
    r2_discrete = evaluate_discrete_algorithms_on_real_pools(
        cal_scores, cal_correct, test_pools_ans,
        n_values=[1, 4, 16, 64], n_trials_per_problem=25, n_bins=25, lcb_delta=0.05, seed=99,
    )

    return {
        "r1_benchmarks": r1_benchmarks,
        "r1_dual_rm_theorem5": {
            "correlation_diagnosis": dual_corr,
            "n_values": n_values,
            "ensemble_curves": ens_curves,
        },
        "r2_split_info": {
            "n_total_problems": len(all_problems),
            "n_cal_problems": len(cal_probs),
            "n_test_problems": len(test_probs),
            "n_cal_candidates": int(cal_scores.size),
        },
        "r2_m400_uniform_k20": r2_m400_uniform,
        "r2_m400_quantile_k40": r2_m400_quantile,
        "r2_m_sweep_n64": m_sweep_records,
        "r2_discrete_algorithms": r2_discrete,
    }


def _parse_h4_model_dataset(main_path: Path, eval_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """解析 HuggingFaceH4 MATH-500 (n=256) 数据集并提取官方 --evals 基准曲线。"""
    tbl = pq.read_table(main_path, columns=["answer", "completions", "scores", "agg_scores"])
    eval_tbl = pq.read_table(eval_path)

    answers = tbl["answer"].to_pylist()
    completions = tbl["completions"].to_pylist()
    scores_list = tbl["scores"].to_pylist()
    agg_scores = tbl["agg_scores"].to_pylist()

    problems: list[dict[str, Any]] = []
    for p_idx in range(len(answers)):
        gold = str(answers[p_idx])
        comps = completions[p_idx]
        step_scores_all = scores_list[p_idx]
        agg_arr = np.asarray(agg_scores[p_idx], dtype=np.float64)

        judge_cache: dict[str, bool] = {}
        extracted: list[str] = []
        is_corr = np.empty(len(comps), dtype=bool)
        prm_last = np.empty(len(comps), dtype=np.float64)
        prm_min = np.empty(len(comps), dtype=np.float64)
        prm_mean = np.empty(len(comps), dtype=np.float64)
        prm_prod = np.empty(len(comps), dtype=np.float64)

        for c_idx, txt in enumerate(comps):
            ans_str = extract_boxed_answer(txt)
            extracted.append(ans_str)
            norm = normalize_math_answer(ans_str)
            if norm not in judge_cache:
                judge_cache[norm] = is_math_answer_correct(ans_str, gold)
            is_corr[c_idx] = judge_cache[norm]

            st = np.asarray(step_scores_all[c_idx], dtype=np.float64)
            if st.size == 0:
                prm_last[c_idx] = 0.0
                prm_min[c_idx] = 0.0
                prm_mean[c_idx] = 0.0
                prm_prod[c_idx] = 0.0
            else:
                prm_last[c_idx] = float(st[-1])
                prm_min[c_idx] = float(np.min(st))
                prm_mean[c_idx] = float(np.mean(st))
                prm_prod[c_idx] = float(np.prod(np.clip(st, 1e-6, 1.0)))

        problems.append({
            "gold": gold,
            "preds": extracted,
            "is_correct": is_corr,
            "agg_scores": agg_arr,
            "prm_last": prm_last,
            "prm_min": prm_min,
            "prm_mean": prm_mean,
            "prm_prod": prm_prod,
        })

    # 提取官方 evals 汇总表并按 n 排序
    ev_n = eval_tbl["n"].to_pylist()
    ev_naive = eval_tbl["acc_naive"].to_pylist()
    ev_weighted = eval_tbl["acc_weighted"].to_pylist()
    ev_maj = eval_tbl["acc_maj"].to_pylist()
    order = np.argsort(ev_n)
    official_table = [
        {
            "n": int(ev_n[i]),
            "acc_naive": float(ev_naive[i]) / 100.0,
            "acc_weighted": float(ev_weighted[i]) / 100.0,
            "acc_maj": float(ev_maj[i]) / 100.0,
        }
        for i in order
    ]

    max_agg_diff = float(np.max([np.max(np.abs(p["agg_scores"] - p["prm_last"])) for p in problems]))
    check_info = {
        "n_problems": len(problems),
        "n_candidates_per_problem": len(problems[0]["preds"]),
        "max_agg_vs_prm_last_diff": max_agg_diff,
        "official_evals_curve": official_table,
    }
    return problems, check_info


def run_math500_model_experiments(model_name: str, problems: list[dict[str, Any]], check_info: dict[str, Any]) -> dict[str, Any]:
    """运行实验 R3（多 PRM 视图相关性与定理 5 诊断）与 R4（5-Fold 交叉验证对比，N=1..256）。"""
    n_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    v_last = np.concatenate([p["prm_last"] for p in problems])
    v_min = np.concatenate([p["prm_min"] for p in problems])
    v_mean = np.concatenate([p["prm_mean"] for p in problems])
    v_prod = np.concatenate([p["prm_prod"] for p in problems])
    y_all = np.concatenate([p["is_correct"] for p in problems])

    multi_diag = diagnose_multi_verifier_correlations(
        {"PRM-Last": v_last, "PRM-Min": v_min, "PRM-Mean": v_mean, "PRM-Prod": v_prod},
        y_all,
    )
    single_last_diag = diagnose_real_verifier_distribution(
        [p["prm_last"] for p in problems], [p["is_correct"] for p in problems]
    )
    single_min_diag = diagnose_real_verifier_distribution(
        [p["prm_min"] for p in problems], [p["is_correct"] for p in problems]
    )

    stats_norm = {}
    for key in ("prm_last", "prm_min", "prm_mean", "prm_prod"):
        vec = np.concatenate([p[key] for p in problems])
        stats_norm[key] = (float(np.mean(vec)), max(1e-8, float(np.std(vec))))

    ens_curves = {k: np.zeros(len(n_values)) for k in ("PRM-Last", "PRM-Min", "Mean-Ensemble (4 views)", "Min-Ensemble (4 views)")}
    for p in problems:
        z_stack = np.vstack([
            (p[k] - stats_norm[k][0]) / stats_norm[k][1]
            for k in ("prm_last", "prm_min", "prm_mean", "prm_prod")
        ])
        ens_curves["PRM-Last"] += exact_pool_bon_accuracy(p["prm_last"], p["is_correct"], n_values)
        ens_curves["PRM-Min"] += exact_pool_bon_accuracy(p["prm_min"], p["is_correct"], n_values)
        ens_curves["Mean-Ensemble (4 views)"] += exact_pool_bon_accuracy(np.mean(z_stack, axis=0), p["is_correct"], n_values)
        ens_curves["Min-Ensemble (4 views)"] += exact_pool_bon_accuracy(np.min(z_stack, axis=0), p["is_correct"], n_values)
    for k in ens_curves:
        ens_curves[k] = ens_curves[k] / float(len(problems))

    # R4: 严格 5-Fold 按题交叉验证（N=1..256）
    n_probs = len(problems)
    rng = np.random.default_rng(2026 if "1b" in model_name else 2027)
    fold_perm = rng.permutation(n_probs)
    n_folds = 5
    fold_size = n_probs // n_folds

    agg_hits: dict[str, np.ndarray] | None = None
    total_trials = 0
    for f_idx in range(n_folds):
        val_idx = set(fold_perm[f_idx * fold_size : (f_idx + 1) * fold_size].tolist())
        cal_p = [problems[i] for i in range(n_probs) if i not in val_idx]
        test_p = [problems[i] for i in range(n_probs) if i in val_idx]

        cal_s = np.concatenate([p["prm_last"] for p in cal_p])
        cal_y = np.concatenate([p["is_correct"] for p in cal_p])
        test_tuples = [(p["prm_last"], p["is_correct"], p["preds"]) for p in test_p]

        fold_res = evaluate_discrete_algorithms_on_real_pools(
            cal_s, cal_y, test_tuples,
            n_values=n_values, n_trials_per_problem=30, n_bins=25, lcb_delta=0.05, seed=500 + f_idx,
        )
        n_eval_fold = fold_res["n_problems"] * fold_res["n_trials_per_problem"]
        total_trials += n_eval_fold
        if agg_hits is None:
            agg_hits = {
                name: np.asarray(d["accuracy"], dtype=np.float64) * n_eval_fold
                for name, d in fold_res["algorithms"].items()
            }
        else:
            for name, d in fold_res["algorithms"].items():
                agg_hits[name] += np.asarray(d["accuracy"], dtype=np.float64) * n_eval_fold

    assert agg_hits is not None
    r4_algos = {}
    for name, hit_sum in agg_hits.items():
        acc = hit_sum / float(total_trials)
        se = np.sqrt(acc * (1.0 - acc) / float(total_trials))
        r4_algos[name] = {
            "accuracy": acc,
            "se": se,
            "ci_low": np.clip(acc - 1.96 * se, 0.0, 1.0),
            "ci_high": np.clip(acc + 1.96 * se, 0.0, 1.0),
        }

    # 校验自算结果与官方 evals 偏差
    off_n1 = check_info["official_evals_curve"][0]["acc_naive"]
    off_n256 = check_info["official_evals_curve"][-1]["acc_naive"]
    off_maj256 = check_info["official_evals_curve"][-1]["acc_maj"]
    check_info["parsed_pass_at_1"] = float(ens_curves["PRM-Last"][0])
    check_info["parsed_naive_at_256"] = float(ens_curves["PRM-Last"][-1])
    check_info["parsed_maj_at_256"] = float(r4_algos["Majority-Voting"]["accuracy"][-1])
    check_info["diff_vs_official_n1"] = abs(check_info["parsed_pass_at_1"] - off_n1)
    check_info["diff_vs_official_naive256"] = abs(check_info["parsed_naive_at_256"] - off_n256)
    check_info["diff_vs_official_maj256"] = abs(check_info["parsed_maj_at_256"] - off_maj256)

    return {
        "sanity_check": check_info,
        "r3_single_prm_last_diag": single_last_diag,
        "r3_single_prm_min_diag": single_min_diag,
        "r3_multi_prm_correlation": multi_diag,
        "r3_ensemble_exact_bon": {"n_values": n_values, "curves": ens_curves},
        "r4_5fold_algorithms": {"n_values": n_values, "total_trials": total_trials, "algorithms": r4_algos},
    }


def main() -> None:
    t0 = time.perf_counter()
    print("=== 步骤 A：检查/下载 HuggingFace 真实公开基准数据 ===")
    ace_parquet = download_acemath_scored_parquet()
    h4_paths = download_hf_h4_completions()

    print("=== 步骤 B：运行实验 R1 & R2 (AceMath-RewardBench 7 大基准 / 8,987 题 × 双 RM) ===")
    ace_data = load_acemath_dual_rm(ace_parquet)
    acemath_res = run_acemath_experiments(ace_data)

    print("=== 步骤 C：运行实验 R3 & R4 (MATH-500 Llama-3.2-1B & 3B / N=1..256) ===")
    p_1b, chk_1b = _parse_h4_model_dataset(h4_paths["llama_1b"]["main"], h4_paths["llama_1b"]["evals"])
    res_1b = run_math500_model_experiments("llama_1b", p_1b, chk_1b)

    p_3b, chk_3b = _parse_h4_model_dataset(h4_paths["llama_3b"]["main"], h4_paths["llama_3b"]["evals"])
    res_3b = run_math500_model_experiments("llama_3b", p_3b, chk_3b)

    elapsed = time.perf_counter() - t0
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "real_llm_results.json"
    payload = to_json_safe({
        "metadata": {"elapsed_seconds": elapsed},
        "acemath_rewardbench": acemath_res,
        "math500_llama_3_2_1b": res_1b,
        "math500_llama_3_2_3b": res_3b,
    })
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 80)
    print(f"真实大模型与开源 RM 实证实验全部完成！保存至: {out_path} (耗时 {elapsed:.2f}s)")
    for sub, d in acemath_res["r1_benchmarks"].items():
        da, db = d["rm_a_diag"], d["rm_b_diag"]
        print(
            f"[AceMath {sub:13s}] p={db['p_hat']:.3f} | kappa_A={da['kappa']:.3f} | kappa_B={db['kappa']:.3f} | "
            f"BoN_B(1->64): {d['rm_b_bon_accuracy'][0]:.3f} -> {d['rm_b_bon_accuracy'][-1]:.3f}"
        )
    dr = acemath_res["r1_dual_rm_theorem5"]
    cd = dr["correlation_diagnosis"]
    ec = dr["ensemble_curves"]
    print(
        f"[AceMath Dual-RM A+B] rho1={cd['mean_rho1']:.3f}, rho0={cd['mean_rho0']:.3f}, "
        f"kappa={cd['mean_kappa']:.3f}, regime={cd['theorem5a_diagnosis']['regime']} | "
        f"N=64: A={ec['RM-A'][-1]:.4f}, B={ec['RM-B'][-1]:.4f}, Mean={ec['Mean-Ensemble (A+B)'][-1]:.4f}, Min={ec['Min-Ensemble (A+B)'][-1]:.4f}"
    )
    r2u = acemath_res["r2_m400_uniform_k20"]["methods"]
    print(
        f"[AceMath M=400 Uniform K=20 N=64] StdBoN={acemath_res['r2_m400_uniform_k20']['standard_bon_accuracy'][-1]:.4f} | "
        f"MLE={r2u['mle']['accuracy'][-1]:.4f} | Wilson-LCB={r2u['wilson_lcb']['accuracy'][-1]:.4f} | "
        f"Bernstein-LCB={r2u['empirical_bernstein_lcb']['accuracy'][-1]:.4f}"
    )
    r2d = acemath_res["r2_discrete_algorithms"]["algorithms"]
    print(
        f"[AceMath Discrete N=64] StdBoN={r2d['Standard-BoN']['accuracy'][-1]:.4f} | "
        f"MajVote={r2d['Majority-Voting']['accuracy'][-1]:.4f} | "
        f"CalWeighted-LCB={r2d['Calibrated-Weighted-BoN (Wilson-LCB)']['accuracy'][-1]:.4f} | "
        f"TwoStage-LCB={r2d['Two-Stage-Verified-Majority (Wilson-LCB)']['accuracy'][-1]:.4f}"
    )
    for m_key, m_res in (("Llama-3.2-1B", res_1b), ("Llama-3.2-3B", res_3b)):
        chk = m_res["sanity_check"]
        md = m_res["r3_multi_prm_correlation"]
        r4 = m_res["r4_5fold_algorithms"]["algorithms"]
        print(
            f"[{m_key} Check] agg_diff={chk['max_agg_vs_prm_last_diff']:.2e} | "
            f"diff_n1={chk['diff_vs_official_n1']:.4f}, diff_naive256={chk['diff_vs_official_naive256']:.4f}, "
            f"diff_maj256={chk['diff_vs_official_maj256']:.4f}"
        )
        print(
            f"[{m_key} R3 Corr] rho1={md['mean_rho1']:.3f}, rho0={md['mean_rho0']:.3f}, "
            f"mean_kappa={md['mean_kappa']:.3f}, regime={md['theorem5a_diagnosis']['regime']}"
        )
        print(
            f"[{m_key} R4 N=256] StdBoN={r4['Standard-BoN']['accuracy'][-1]:.4f} | "
            f"MajVote={r4['Majority-Voting']['accuracy'][-1]:.4f} | "
            f"CalWeighted-LCB={r4['Calibrated-Weighted-BoN (Wilson-LCB)']['accuracy'][-1]:.4f} | "
            f"TwoStage-LCB={r4['Two-Stage-Verified-Majority (Wilson-LCB)']['accuracy'][-1]:.4f}"
        )


if __name__ == "__main__":
    main()
