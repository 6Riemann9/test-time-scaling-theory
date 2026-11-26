"""Stage 2 真实开源多 RM / PRM 跨领域大规模实证验证脚本（纯 CPU，零 GPU）。

全面覆盖四大领域、10+ 开源 RM/PRM 家族（0.6B–72B），检验 V4/V5 三大原创理论支柱：
- S1a: `allenai/reward-bench-2-results`（通用 6 域 1,865 题 + Ties 多候选：6-异构家族 vs 4-同源家族 + DeBERTa 脆弱注入）
- S1b: `allenai/reward-bench-results`（HumanEval-Pack 6 语言代码 984 题 + Math-PRM + Safety + 全域 2,985 题）
- S2: `sibasmarakp` OlympiadBench (`Qwen2.5-7B-Instruct` × `uPRM-70B` vs `uPRM-32B`, 674 题 × 128 = 86,272 条 + 官方曲线核验)
- S3: `Human-Centric-Machine-Learning/strategic-ttc-data` (`GPQA` 7B/8B + `GSM8K` 7B + `AIME` 7B, `ArmoRM-Llama3-8B`, N=1..128)
- S4: `nvidia/AceMath-RewardBench`（7 难度域 8,987 题 × 64 = 575,168 条：8-SOTA 校准器、2D (M,N) 网格与 7 域辛普森遮蔽）

输出：`results/stage2_real_results.json` 与 `results/stage2_summary.csv`。
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys
import time
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
    diagnose_real_verifier_distribution,
    evaluate_discrete_algorithms_on_real_pools,
    evaluate_heterogeneous_multi_rm_on_real_pools,
    evaluate_joint_m_n_calibration_on_real_pools,
    evaluate_real_prompt_heterogeneity,
    exact_pool_bon_accuracy,
    extract_boxed_answer,
    is_math_answer_correct_any,
    normalize_math_answer,
)

DATA_DIR = PROJECT_ROOT / "data"
STAGE2_DIR = DATA_DIR / "stage2"
CACHE_DIR = STAGE2_DIR / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"


def to_json_safe(obj: Any) -> Any:
    """将 NumPy 标量与数组递归转为原生 JSON 安全类型。"""
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


def download_hf_file_retry(
    repo_id: str,
    filename: str,
    local_root: Path,
    max_retries: int = 6,
) -> Path:
    """带指数退避重试的 HuggingFace 数据集文件下载与本地复用。"""
    local_root.mkdir(parents=True, exist_ok=True)
    target = local_root / filename
    if target.exists() and target.stat().st_size > 0:
        return target

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            print(f"  [下载] {repo_id} :: {filename} (尝试 {attempt + 1}/{max_retries}) ...")
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=str(local_root),
            )
            return Path(downloaded)
        except Exception as exc:
            last_err = exc
            wait_s = min(24.0, 1.5 * (2**attempt))
            print(f"  [重试等待 {wait_s:.1f}s] 下载出错: {exc}")
            time.sleep(wait_s)
    raise RuntimeError(f"下载 {repo_id}/{filename} 在 {max_retries} 次尝试后失败: {last_err}")


def download_url_retry(url: str, dst: Path, max_retries: int = 6) -> Path:
    """带流式原子写入与指数退避重试的 URL 下载器。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    part = dst.with_suffix(dst.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            print(f"  [下载] {url} (尝试 {attempt + 1}/{max_retries}) ...")
            with requests.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with part.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            fh.write(chunk)
            part.replace(dst)
            return dst
        except Exception as exc:
            last_err = exc
            if part.exists():
                part.unlink(missing_ok=True)
            wait_s = min(24.0, 1.5 * (2**attempt))
            print(f"  [重试等待 {wait_s:.1f}s] URL 下载出错: {exc}")
            time.sleep(wait_s)
    raise RuntimeError(f"下载 {url} 失败: {last_err}")


def _save_stage_cache(name: str, data: dict[str, Any]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.json"
    path.write_text(json.dumps(to_json_safe(data), indent=2, ensure_ascii=False), encoding="utf-8")


def _load_stage_cache(name: str) -> dict[str, Any] | None:
    path = CACHE_DIR / f"{name}.json"
    if path.exists() and path.stat().st_size > 0:
        return json.loads(path.read_text(encoding="utf-8"))
    return None


# ==============================================================================
# S1a: allenai/reward-bench-2-results
# ==============================================================================
RB2_HETERO6 = {
    "Skywork-V2-Llama3.1-8B": "eval-set-scores/Skywork/Skywork-Reward-V2-Llama-3.1-8B.json",
    "WorldPM-72B": "eval-set-scores/Qwen/WorldPM-72B.json",
    "InternLM2-20B": "eval-set-scores/internlm/internlm2-20b-reward.json",
    "Starling-34B": "eval-set-scores/Nexusflow/Starling-RM-34B.json",
    "GRM-Gemma2-2B": "eval-set-scores/Ray2333/GRM-gemma2-2B-rewardmodel-ft.json",
    "Mistral-RAFT-7B": "eval-set-scores/hendrydong/Mistral-RM-for-RAFT-GSHF-v0.json",
}
RB2_WEAK_DEBERTA = {
    "DeBERTa-v3-Large": "eval-set-scores/OpenAssistant/reward-model-deberta-v3-large-v2.json",
}
RB2_HOMO4 = {
    "AllenAI-8B-RM": "eval-set-scores/allenai/Llama-3.1-8B-Instruct-RM-RB2.json",
    "Tulu3-8B-DPO": "eval-set-scores/allenai/Llama-3.1-Tulu-3-8B-DPO-RM-RB2.json",
    "Tulu3-8B-RL": "eval-set-scores/allenai/Llama-3.1-Tulu-3-8B-RL-RM-RB2.json",
    "AllenAI-70B-RM": "eval-set-scores/allenai/Llama-3.1-70B-Instruct-RM-RB2.json",
}


def _unwrap_scalar(v: Any) -> float:
    """将可能被单元素列表嵌套包裹的标量打分安全解包为 float。"""
    while isinstance(v, list):
        v = v[0]
    return float(v)


def _load_rb2_model(path: Path) -> dict[tuple[str, str, int], tuple[list[float], int]]:
    """加载单个 RB2 评分文件，按 (id, subset, occ) 建索引以便跨模型严格对齐。"""
    raw = json.loads(path.read_text(encoding="utf-8"))
    ids = raw["id"]
    subsets = raw["subset"]
    scores_all = raw["scores"]
    num_corr = raw["num_correct"]

    seen: Counter[tuple[str, str]] = Counter()
    out: dict[tuple[str, str, int], tuple[list[float], int]] = {}
    for pid, sub, sc, nc in zip(ids, subsets, scores_all, num_corr):
        base_k = (str(pid), str(sub))
        occ = seen[base_k]
        seen[base_k] += 1
        sc_f = [_unwrap_scalar(v) for v in sc]
        nc_i = int(nc)
        if nc_i < 1 or nc_i >= len(sc_f):
            continue
        out[(base_k[0], base_k[1], occ)] = (sc_f, nc_i)
    return out


def _align_rb2_group(
    model_dicts: dict[str, dict[tuple[str, str, int], tuple[list[float], int]]],
    subset_filter: str | None = None,
) -> tuple[list[str], list[np.ndarray], list[np.ndarray], list[str]]:
    names = list(model_dicts.keys())
    common_keys = set.intersection(*(set(d.keys()) for d in model_dicts.values()))
    if subset_filter is not None:
        common_keys = {k for k in common_keys if k[1] == subset_filter}
    ordered_keys = sorted(common_keys)

    mats: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    strata: list[str] = []
    for k in ordered_keys:
        lens = {len(model_dicts[n][k][0]) for n in names}
        ncs = {model_dicts[n][k][1] for n in names}
        if len(lens) != 1 or len(ncs) != 1:
            continue
        l_i = next(iter(lens))
        nc_i = next(iter(ncs))
        mat = np.column_stack([np.asarray(model_dicts[n][k][0], dtype=np.float64) for n in names])
        y = np.zeros(l_i, dtype=bool)
        y[:nc_i] = True
        mats.append(mat)
        ys.append(y)
        strata.append(k[1])
    return names, mats, ys, strata


def run_s1a_rb2(force: bool = False) -> dict[str, Any]:
    if not force:
        cached = _load_stage_cache("s1a_rb2")
        if cached is not None:
            print("[S1a] 命中本地缓存 s1a_rb2.json，跳过重复计算。")
            return cached

    print("\n=== S1a: allenai/reward-bench-2-results (异构 6+1 家族 vs 同源 4 家族) ===")
    rb2_root = STAGE2_DIR / "reward_bench_2"
    all_specs = {**RB2_HETERO6, **RB2_WEAK_DEBERTA, **RB2_HOMO4}
    loaded = {}
    for name, rel in all_specs.items():
        fpath = download_hf_file_retry("allenai/reward-bench-2-results", rel, rb2_root)
        loaded[name] = _load_rb2_model(fpath)

    # 1. 全量 RB2 (N=1..4) 与 Ties 子集 (N=1,2,4,8,11)
    h6_names, h6_mats, h6_ys, h6_strata = _align_rb2_group({k: loaded[k] for k in RB2_HETERO6})
    h7_names, h7_mats, h7_ys, _ = _align_rb2_group({k: loaded[k] for k in {**RB2_HETERO6, **RB2_WEAK_DEBERTA}})
    m4_names, m4_mats, m4_ys, _ = _align_rb2_group({k: loaded[k] for k in RB2_HOMO4})

    t6_names, t6_mats, t6_ys, _ = _align_rb2_group({k: loaded[k] for k in RB2_HETERO6}, subset_filter="Ties")
    t7_names, t7_mats, t7_ys, _ = _align_rb2_group({k: loaded[k] for k in {**RB2_HETERO6, **RB2_WEAK_DEBERTA}}, subset_filter="Ties")
    tm4_names, tm4_mats, tm4_ys, _ = _align_rb2_group({k: loaded[k] for k in RB2_HOMO4}, subset_filter="Ties")

    res_h6_all = evaluate_heterogeneous_multi_rm_on_real_pools(h6_names, h6_mats, h6_ys, n_values=[1, 2, 3, 4], seed=101)
    res_h7_all = evaluate_heterogeneous_multi_rm_on_real_pools(h7_names, h7_mats, h7_ys, n_values=[1, 2, 3, 4], seed=102)
    res_m4_all = evaluate_heterogeneous_multi_rm_on_real_pools(m4_names, m4_mats, m4_ys, n_values=[1, 2, 3, 4], seed=103)

    res_h6_ties = evaluate_heterogeneous_multi_rm_on_real_pools(t6_names, t6_mats, t6_ys, n_values=[1, 2, 4, 8, 11], seed=104)
    res_h7_ties = evaluate_heterogeneous_multi_rm_on_real_pools(t7_names, t7_mats, t7_ys, n_values=[1, 2, 4, 8, 11], seed=105)
    res_m4_ties = evaluate_heterogeneous_multi_rm_on_real_pools(tm4_names, tm4_mats, tm4_ys, n_values=[1, 2, 4, 8, 11], seed=106)

    # 2. RB2 跨 6 域题目异质性检验（以最强单模型 Skywork-V2-Llama3.1-8B 为例）
    pools_strata = [(st, mat[:, 0], y) for st, mat, y in zip(h6_strata, h6_mats, h6_ys)]
    het_rb2 = evaluate_real_prompt_heterogeneity(
        pools_strata, n_values=[1, 2, 3, 4], n_folds=5, n_trials_per_problem=20, n_bins=15, n_simpson_bins=10, seed=107
    )

    out = {
        "hetero6_all": res_h6_all,
        "hetero6_plus_deberta_all": res_h7_all,
        "homo4_all": res_m4_all,
        "hetero6_ties": res_h6_ties,
        "hetero6_plus_deberta_ties": res_h7_ties,
        "homo4_ties": res_m4_ties,
        "rb2_skywork_heterogeneity": het_rb2,
    }
    _save_stage_cache("s1a_rb2", out)
    return out


# ==============================================================================
# S1b: allenai/reward-bench-results (多语言代码 + 数学 + 安全)
# ==============================================================================
RB1_MODELS = {
    "Skywork-Gemma2-27B": "eval-set-scores/Skywork/Skywork-Reward-Gemma-2-27B-v0.2.json",
    "Skywork-Llama3.1-8B": "eval-set-scores/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2.json",
    "Starling-34B": "eval-set-scores/Nexusflow/Starling-RM-34B.json",
    "InternLM2-20B": "eval-set-scores/internlm/internlm2-20b-reward.json",
    "Mistral-DPA-7B": "eval-set-scores/RLHFlow/RewardModel-Mistral-7B-for-DPA-v1.json",
    "DeBERTa-v3-Large": "eval-set-scores/OpenAssistant/reward-model-deberta-v3-large-v2.json",
}

RB1_CODE_SUBSETS = {"hep-python", "hep-cpp", "hep-go", "hep-js", "hep-rust", "hep-java"}
RB1_SAFETY_SUBSETS = {
    "xstest-should-respond",
    "xstest-should-refuse",
    "donotanswer",
    "refusals-dangerous",
    "refusals-offensive",
}


def _load_rb1_model(path: Path) -> dict[tuple[int, str], tuple[float, float]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[tuple[int, str], tuple[float, float]] = {}
    for pid, sub, sc, sr in zip(raw["id"], raw["subset"], raw["scores_chosen"], raw["scores_rejected"]):
        out[(int(pid), str(sub))] = (_unwrap_scalar(sc), _unwrap_scalar(sr))
    return out


def run_s1b_rb1(force: bool = False) -> dict[str, Any]:
    if not force:
        cached = _load_stage_cache("s1b_rb1")
        if cached is not None:
            print("[S1b] 命中本地缓存 s1b_rb1.json，跳过重复计算。")
            return cached

    print("\n=== S1b: allenai/reward-bench-results (HumanEval-Pack 6 语言代码 + Math + Safety) ===")
    rb1_root = STAGE2_DIR / "reward_bench_1"
    loaded = {}
    for name, rel in RB1_MODELS.items():
        fpath = download_hf_file_retry("allenai/reward-bench-results", rel, rb1_root)
        loaded[name] = _load_rb1_model(fpath)

    names6 = list(RB1_MODELS.keys())
    names5 = names6[:-1]  # 不含弱 DeBERTa
    common_keys = sorted(set.intersection(*(set(loaded[n].keys()) for n in names6)))

    def _subset_eval(model_names: list[str], subset_set: set[str] | None, tag_seed: int) -> dict[str, Any]:
        keys = [k for k in common_keys if (subset_set is None or k[1] in subset_set)]
        mats = []
        ys = []
        for k in keys:
            chosen_row = [loaded[n][k][0] for n in model_names]
            rej_row = [loaded[n][k][1] for n in model_names]
            mats.append(np.vstack([chosen_row, rej_row]))
            ys.append(np.array([True, False], dtype=bool))
        return evaluate_heterogeneous_multi_rm_on_real_pools(model_names, mats, ys, n_values=[1, 2], seed=tag_seed)

    out = {
        "n_aligned_total": len(common_keys),
        "code_5strong": _subset_eval(names5, RB1_CODE_SUBSETS, 201),
        "code_6with_deberta": _subset_eval(names6, RB1_CODE_SUBSETS, 202),
        "math_prm_5strong": _subset_eval(names5, {"math-prm"}, 203),
        "safety_5strong": _subset_eval(names5, RB1_SAFETY_SUBSETS, 204),
        "all_5strong": _subset_eval(names5, None, 205),
        "all_6with_deberta": _subset_eval(names6, None, 206),
    }
    _save_stage_cache("s1b_rb1", out)
    return out


# ==============================================================================
# S2: OlympiadBench Qwen2.5-7B-Instruct × uPRM-70B vs uPRM-32B (N=1..128)
# ==============================================================================
def _parse_olympiad_repo(repo_id: str, local_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = "updated-olympiadbench--T-0.7--top_p-0.8--n-128--seed-0--agg_strategy-last"
    main_p = download_hf_file_retry(repo_id, f"{cfg}/train-00000-of-00001.parquet", local_root)
    eval_p = download_hf_file_retry(repo_id, f"{cfg}--evals/train-00000-of-00001.parquet", local_root)

    tbl = pq.read_table(main_p, columns=["id", "answer", "subfield", "completions", "scores", "agg_scores"])
    eval_tbl = pq.read_table(eval_p)

    ids = tbl["id"].to_pylist()
    answers = tbl["answer"].to_pylist()
    subfields = tbl["subfield"].to_pylist()
    completions = tbl["completions"].to_pylist()
    scores_all = tbl["scores"].to_pylist()
    agg_all = tbl["agg_scores"].to_pylist()

    problems: list[dict[str, Any]] = []
    for i in range(len(ids)):
        golds = [str(g) for g in (answers[i] or [])]
        comps = completions[i]
        steps_list = scores_all[i]
        agg_arr = np.asarray(agg_all[i], dtype=np.float64)

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
                judge_cache[norm] = is_math_answer_correct_any(ans_str, golds)
            is_corr[c_idx] = judge_cache[norm]

            st = np.asarray(steps_list[c_idx], dtype=np.float64)
            if st.size == 0:
                prm_last[c_idx] = prm_min[c_idx] = prm_mean[c_idx] = prm_prod[c_idx] = 0.0
            else:
                prm_last[c_idx] = float(st[-1])
                prm_min[c_idx] = float(np.min(st))
                prm_mean[c_idx] = float(np.mean(st))
                prm_prod[c_idx] = float(np.prod(np.clip(st, 1e-6, 1.0)))

        problems.append({
            "id": int(ids[i]),
            "subfield": str(subfields[i] or "Other"),
            "preds": extracted,
            "is_correct": is_corr,
            "agg_scores": agg_arr,
            "prm_last": prm_last,
            "prm_min": prm_min,
            "prm_mean": prm_mean,
            "prm_prod": prm_prod,
        })

    ev_n = eval_tbl["n"].to_pylist()
    ev_naive = eval_tbl["acc_naive"].to_pylist()
    ev_weighted = eval_tbl["acc_weighted"].to_pylist()
    ev_maj = eval_tbl["acc_maj"].to_pylist()
    order = np.argsort(ev_n)
    official_evals = [
        {
            "n": int(ev_n[j]),
            "acc_naive": float(ev_naive[j]) / 100.0,
            "acc_weighted": float(ev_weighted[j]) / 100.0,
            "acc_maj": float(ev_maj[j]) / 100.0,
        }
        for j in order
    ]
    return problems, official_evals


def run_s2_olympiadbench(force: bool = False) -> dict[str, Any]:
    if not force:
        cached = _load_stage_cache("s2_olympiadbench")
        if cached is not None:
            print("[S2] 命中本地缓存 s2_olympiadbench.json，跳过重复计算。")
            return cached

    print("\n=== S2: OlympiadBench Qwen2.5-7B × uPRM-70B vs uPRM-32B (N=1..128) ===")
    oly_root = STAGE2_DIR / "olympiadbench"
    repo_70b = "sibasmarakp/Qwen2.5-7B-Instruct-uPRM-70B-T80-olympiadbench-best_of_n-completions"
    repo_32b = "sibasmarakp/Qwen2.5-7B-Instruct-uPRM-32B-T80-olympiadbench-best_of_n-completions"

    p70, ev70 = _parse_olympiad_repo(repo_70b, oly_root / "uprm_70b")
    p32, ev32 = _parse_olympiad_repo(repo_32b, oly_root / "uprm_32b")

    # 按 id 严格对齐（并验证是否同一批解答）
    map70 = {p["id"]: p for p in p70}
    map32 = {p["id"]: p for p in p32}
    common_ids = sorted(set(map70.keys()) & set(map32.keys()))
    aligned_70 = [map70[pid] for pid in common_ids]
    aligned_32 = [map32[pid] for pid in common_ids]

    n_values = [1, 2, 4, 8, 16, 32, 64, 128]
    # 核对官方 evals 误差
    exact70 = np.mean([exact_pool_bon_accuracy(p["prm_last"], p["is_correct"], n_values) for p in aligned_70], axis=0)
    exact32 = np.mean([exact_pool_bon_accuracy(p["prm_last"], p["is_correct"], n_values) for p in aligned_32], axis=0)
    diag70 = diagnose_real_verifier_distribution([p["prm_last"] for p in aligned_70], [p["is_correct"] for p in aligned_70])
    diag32 = diagnose_real_verifier_distribution([p["prm_last"] for p in aligned_32], [p["is_correct"] for p in aligned_32])

    # 5-Fold CV 离散答案算法对比 (N=1..128)
    rng = np.random.default_rng(2026)
    perm = rng.permutation(len(aligned_70))
    folds = np.array_split(perm, 5)
    agg_hits: dict[str, np.ndarray] | None = None
    total_trials = 0
    for f_idx in range(5):
        test_set = set(folds[f_idx].tolist())
        cal_p = [aligned_70[i] for i in range(len(aligned_70)) if i not in test_set]
        test_p = [aligned_70[i] for i in range(len(aligned_70)) if i in test_set]
        cal_s = np.concatenate([p["prm_last"] for p in cal_p])
        cal_y = np.concatenate([p["is_correct"] for p in cal_p])
        test_tuples = [(p["prm_last"], p["is_correct"], p["preds"]) for p in test_p]
        f_res = evaluate_discrete_algorithms_on_real_pools(
            cal_s, cal_y, test_tuples, n_values=n_values, n_trials_per_problem=25, n_bins=25, seed=600 + f_idx
        )
        n_eval = f_res["n_problems"] * f_res["n_trials_per_problem"]
        total_trials += n_eval
        if agg_hits is None:
            agg_hits = {k: np.asarray(d["accuracy"]) * n_eval for k, d in f_res["algorithms"].items()}
        else:
            for k, d in f_res["algorithms"].items():
                agg_hits[k] += np.asarray(d["accuracy"]) * n_eval

    assert agg_hits is not None
    r_algos = {}
    for k, h_sum in agg_hits.items():
        acc = h_sum / float(total_trials)
        se = np.sqrt(acc * (1.0 - acc) / float(total_trials))
        r_algos[k] = {
            "accuracy": acc,
            "se": se,
            "ci_low": np.clip(acc - 1.96 * se, 0.0, 1.0),
            "ci_high": np.clip(acc + 1.96 * se, 0.0, 1.0),
        }

    # 跨模型多 PRM 融合（70B-Last, 32B-Last, 70B-Min, 32B-Min）vs 同源 70B 4 视图
    cross_mats = [
        np.column_stack([a["prm_last"], b["prm_last"], a["prm_min"], b["prm_min"]])
        for a, b in zip(aligned_70, aligned_32)
    ]
    cross_ys = [a["is_correct"] for a in aligned_70]
    cross_res = evaluate_heterogeneous_multi_rm_on_real_pools(
        ["uPRM-70B-Last", "uPRM-32B-Last", "uPRM-70B-Min", "uPRM-32B-Min"],
        cross_mats,
        cross_ys,
        n_values=n_values,
        seed=610,
    )
    pair_mats = [np.column_stack([a["prm_last"], b["prm_last"]]) for a, b in zip(aligned_70, aligned_32)]
    pair_res = evaluate_heterogeneous_multi_rm_on_real_pools(
        ["uPRM-70B-Last", "uPRM-32B-Last"],
        pair_mats,
        cross_ys,
        n_values=n_values,
        seed=611,
    )
    views70_mats = [
        np.column_stack([a["prm_last"], a["prm_min"], a["prm_mean"], a["prm_prod"]])
        for a in aligned_70
    ]
    views70_res = evaluate_heterogeneous_multi_rm_on_real_pools(
        ["70B-Last", "70B-Min", "70B-Mean", "70B-Prod"],
        views70_mats,
        cross_ys,
        n_values=n_values,
        seed=612,
    )

    # 跨学科子域题目异质性检验（Algebra / Combinatorics / Geometry / Number Theory / Physics）
    het_oly = evaluate_real_prompt_heterogeneity(
        [(p["subfield"], p["prm_last"], p["is_correct"]) for p in aligned_70],
        n_values=n_values,
        n_folds=5,
        n_trials_per_problem=20,
        n_bins=15,
        n_simpson_bins=10,
        seed=613,
    )

    off70_n1, off70_n128, off70_maj128 = ev70[0]["acc_naive"], ev70[-1]["acc_naive"], ev70[-1]["acc_maj"]
    off32_n1, off32_n128 = ev32[0]["acc_naive"], ev32[-1]["acc_naive"]
    sanity = {
        "n_problems": len(aligned_70),
        "official_70b": ev70,
        "official_32b": ev32,
        "parsed_70b_exact_bon": exact70,
        "parsed_32b_exact_bon": exact32,
        "diff_70b_n1": abs(float(exact70[0]) - off70_n1),
        "diff_70b_n128": abs(float(exact70[-1]) - off70_n128),
        "diff_70b_maj128": abs(float(r_algos["Majority-Voting"]["accuracy"][-1]) - off70_maj128),
        "diff_32b_n1": abs(float(exact32[0]) - off32_n1),
        "diff_32b_n128": abs(float(exact32[-1]) - off32_n128),
    }

    out = {
        "sanity_check": sanity,
        "diag_70b_last": diag70,
        "diag_32b_last": diag32,
        "discrete_5fold_70b": {"n_values": n_values, "total_trials": total_trials, "algorithms": r_algos},
        "cross_prm_4views_70b_32b": cross_res,
        "pair_prm_70b_vs_32b": pair_res,
        "same_prm_70b_4views": views70_res,
        "olympiad_subfield_heterogeneity": het_oly,
    }
    _save_stage_cache("s2_olympiadbench", out)
    return out


# ==============================================================================
# S3: Human-Centric-Machine-Learning/strategic-ttc-data (GPQA 7B/8B + GSM8K + AIME)
# ==============================================================================
TTC_FILES = {
    "gpqa_qwen7b": "GPQA/Qwen2.5-7B--temp-0.7--samples-128--max-512.jsonl",
    "gpqa_llama8b": "GPQA/Llama-3.1-8B--temp-0.6--samples-128--max-512.jsonl",
    "gsm8k_qwen7b": "GSM8K/Qwen2.5-7B--temp-0.7--samples-128--max-512.jsonl",
    "aime_qwen7b": "AIME/Qwen2.5-7B--temp-0.7--samples-128--max-512.jsonl",
}


def _stream_parse_strategic_ttc(tag: str, rel_path: str) -> list[dict[str, Any]]:
    """流式解析大型 JSONL，仅保留 qid/rewards/correct/explanations 并保存轻量化缓存。"""
    parsed_cache = STAGE2_DIR / "parsed" / f"{tag}.json"
    if parsed_cache.exists() and parsed_cache.stat().st_size > 0:
        return json.loads(parsed_cache.read_text(encoding="utf-8"))

    raw_path = download_hf_file_retry(
        "Human-Centric-Machine-Learning/strategic-ttc-data",
        rel_path,
        STAGE2_DIR / "strategic_ttc",
    )
    items: list[dict[str, Any]] = []
    with raw_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rew = obj.get("rewards") or []
            corr = obj.get("correct") or []
            expl = obj.get("explanations") or [""] * len(rew)
            if len(rew) == 0 or len(rew) != len(corr):
                continue
            items.append({
                "qid": str(obj.get("qid", len(items))),
                "rewards": [float(x) for x in rew],
                "correct": [bool(x) for x in corr],
                "explanations": [str(x) if x is not None else "" for x in expl],
            })
    parsed_cache.parent.mkdir(parents=True, exist_ok=True)
    parsed_cache.write_text(json.dumps(items), encoding="utf-8")
    return items


def run_s3_strategic_ttc(force: bool = False) -> dict[str, Any]:
    if not force:
        cached = _load_stage_cache("s3_strategic_ttc")
        if cached is not None:
            print("[S3] 命中本地缓存 s3_strategic_ttc.json，跳过重复计算。")
            return cached

    print("\n=== S3: strategic-ttc-data (GPQA 7B/8B + GSM8K 7B + AIME 7B, ArmoRM-8B, N=1..128) ===")
    parsed = {tag: _stream_parse_strategic_ttc(tag, rel) for tag, rel in TTC_FILES.items()}
    n_values = [1, 2, 4, 8, 16, 32, 64, 128]

    # 1. 各基准单 RM 诊断与精确 BoN 曲线
    single_diags: dict[str, Any] = {}
    for tag, items in parsed.items():
        s_list = [np.asarray(x["rewards"], dtype=np.float64) for x in items]
        y_list = [np.asarray(x["correct"], dtype=bool) for x in items]
        diag = diagnose_real_verifier_distribution(s_list, y_list)
        exact_curve = np.mean([exact_pool_bon_accuracy(s, y, n_values) for s, y in zip(s_list, y_list)], axis=0)
        single_diags[tag] = {
            "diag": diag,
            "n_values": n_values,
            "exact_bon_accuracy": exact_curve,
        }

    # 2. GPQA (Qwen2.5-7B & Llama-3.1-8B) 5-Fold 离散答案抗坍塌对比
    discrete_results: dict[str, Any] = {}
    for tag in ("gpqa_qwen7b", "gpqa_llama8b", "gsm8k_qwen7b"):
        items = parsed[tag]
        rng = np.random.default_rng(701 if "gpqa" in tag else 702)
        perm = rng.permutation(len(items))
        folds = np.array_split(perm, 5)
        agg_hits: dict[str, np.ndarray] | None = None
        total_trials = 0
        for f_idx in range(5):
            test_set = set(folds[f_idx].tolist())
            cal_it = [items[i] for i in range(len(items)) if i not in test_set]
            test_it = [items[i] for i in range(len(items)) if i in test_set]
            cal_s = np.concatenate([np.asarray(x["rewards"], dtype=np.float64) for x in cal_it])
            cal_y = np.concatenate([np.asarray(x["correct"], dtype=bool) for x in cal_it])
            test_tuples = [
                (
                    np.asarray(x["rewards"], dtype=np.float64),
                    np.asarray(x["correct"], dtype=bool),
                    x["explanations"],
                )
                for x in test_it
            ]
            f_res = evaluate_discrete_algorithms_on_real_pools(
                cal_s, cal_y, test_tuples, n_values=n_values, n_trials_per_problem=25, n_bins=20, seed=710 + f_idx
            )
            n_eval = f_res["n_problems"] * f_res["n_trials_per_problem"]
            total_trials += n_eval
            if agg_hits is None:
                agg_hits = {k: np.asarray(d["accuracy"]) * n_eval for k, d in f_res["algorithms"].items()}
            else:
                for k, d in f_res["algorithms"].items():
                    agg_hits[k] += np.asarray(d["accuracy"]) * n_eval
        assert agg_hits is not None
        alg_out = {}
        for k, hs in agg_hits.items():
            acc = hs / float(total_trials)
            se = np.sqrt(acc * (1.0 - acc) / float(total_trials))
            alg_out[k] = {"accuracy": acc, "se": se, "ci_low": np.clip(acc - 1.96 * se, 0.0, 1.0), "ci_high": np.clip(acc + 1.96 * se, 0.0, 1.0)}
        discrete_results[tag] = {"n_values": n_values, "total_trials": total_trials, "algorithms": alg_out}

    # 3. GPQA-Qwen2.5-7B 8-SOTA 校准器与 2D (M, N) 检验（50/50 题级划分）
    gpqa_items = parsed["gpqa_qwen7b"]
    rng_g = np.random.default_rng(750)
    perm_g = rng_g.permutation(len(gpqa_items))
    half_g = len(gpqa_items) // 2
    cal_g = [gpqa_items[i] for i in perm_g[:half_g]]
    test_g = [gpqa_items[i] for i in perm_g[half_g:]]
    cal_gs = np.concatenate([np.asarray(x["rewards"], dtype=np.float64) for x in cal_g])
    cal_gy = np.concatenate([np.asarray(x["correct"], dtype=bool) for x in cal_g])
    test_g_pools = [(np.asarray(x["rewards"], dtype=np.float64), np.asarray(x["correct"], dtype=bool)) for x in test_g]
    gpqa_joint = evaluate_joint_m_n_calibration_on_real_pools(
        cal_gs,
        cal_gy,
        test_g_pools,
        m_grid=[100, 200, 400, 800, 2000, 5000, 12000],
        n_values=n_values,
        sota_m_cal=300,
        sota_n_bins=16,
        sota_repeats=40,
        grid_repeats=50,
        seed=751,
    )

    # 4. 跨 3 域 (GSM8K + GPQA + AIME, Qwen2.5-7B + ArmoRM-8B) 真实辛普森悖论与 Z-Score 交叉
    # 为均衡三域规模，各抽齐至相同/均衡题数或保留全量
    pooled_3domain: list[tuple[str, np.ndarray, np.ndarray]] = []
    for dom_name, tag in (("GSM8K", "gsm8k_qwen7b"), ("GPQA", "gpqa_qwen7b"), ("AIME", "aime_qwen7b")):
        for x in parsed[tag]:
            pooled_3domain.append((
                dom_name,
                np.asarray(x["rewards"], dtype=np.float64),
                np.asarray(x["correct"], dtype=bool),
            ))
    het_3domain = evaluate_real_prompt_heterogeneity(
        pooled_3domain,
        n_values=n_values,
        n_folds=5,
        n_trials_per_problem=20,
        n_bins=15,
        n_simpson_bins=12,
        seed=760,
    )

    out = {
        "single_rm_diagnostics": single_diags,
        "discrete_5fold": discrete_results,
        "gpqa_7b_joint_mn_sota": gpqa_joint,
        "cross_3domain_simpson_het": het_3domain,
    }
    _save_stage_cache("s3_strategic_ttc", out)
    return out


# ==============================================================================
# S4: nvidia/AceMath-RewardBench (8,987 题 × 64 = 575,168 条)
# ==============================================================================
ACEMATH_SEGMENTS = (
    ("college_math", 0, 2818),
    ("gaokao2023en", 2818, 3203),
    ("gsm8k", 3203, 4522),
    ("math500", 4522, 5022),
    ("minerva_math", 5022, 5294),
    ("mmlu_stem", 5294, 8312),
    ("olympiadbench", 8312, 8987),
)


def run_s4_acemath(force: bool = False) -> dict[str, Any]:
    if not force:
        cached = _load_stage_cache("s4_acemath")
        if cached is not None:
            print("[S4] 命中本地缓存 s4_acemath.json，跳过重复计算。")
            return cached

    print("\n=== S4: nvidia/AceMath-RewardBench (8,987 题: 8-SOTA, 2D (M,N) & 7 域辛普森) ===")
    ace_dst = DATA_DIR / "acemath_rewardbench" / "default_test_0.parquet"
    ace_url = "https://huggingface.co/api/datasets/nvidia/AceMath-RewardBench/parquet/default/test/0.parquet"
    download_url_retry(ace_url, ace_dst)

    tbl = pq.read_table(ace_dst)
    rm_all = tbl["rm_score"].to_pylist()
    corr_all = tbl["score"].to_pylist()
    offset_b = 8987

    all_probs: list[dict[str, Any]] = []
    for sub_name, s0, s1 in ACEMATH_SEGMENTS:
        for idx in range(s0, s1):
            all_probs.append({
                "subset": sub_name,
                "rm_a": np.asarray(rm_all[idx], dtype=np.float64),
                "rm_b": np.asarray(rm_all[offset_b + idx], dtype=np.float64),
                "is_correct": np.asarray(corr_all[idx], dtype=bool),
            })

    n_values = [1, 2, 4, 8, 16, 32, 64]
    # 1. 严格 50/50 题级划分 -> 8-SOTA 校准器 + 2D (M, N) 网格 + 1/5 vs 2/5 标度对照 (RM-B)
    rng = np.random.default_rng(2026)
    perm = rng.permutation(len(all_probs))
    half = len(all_probs) // 2
    cal_p = [all_probs[i] for i in perm[:half]]
    test_p = [all_probs[i] for i in perm[half:]]
    cal_s = np.concatenate([p["rm_b"] for p in cal_p])
    cal_y = np.concatenate([p["is_correct"] for p in cal_p])
    test_pools = [(p["rm_b"], p["is_correct"]) for p in test_p]

    joint_rm_b = evaluate_joint_m_n_calibration_on_real_pools(
        cal_s,
        cal_y,
        test_pools,
        m_grid=[100, 200, 400, 800, 2000, 5000, 15000, 40000],
        n_values=n_values,
        sota_m_cal=400,
        sota_n_bins=20,
        sota_repeats=30,
        grid_repeats=60,
        seed=810,
    )

    # 2. 双独立 RM (RM-A, RM-B) 5-Fold 广义特征值 vs Min vs Logit-Stack
    dual_res = evaluate_heterogeneous_multi_rm_on_real_pools(
        ["AceMath-RM-A", "AceMath-RM-B"],
        [np.column_stack([p["rm_a"], p["rm_b"]]) for p in all_probs],
        [p["is_correct"] for p in all_probs],
        n_values=n_values,
        seed=820,
    )

    # 3. 7 难度域题目异质性：辛普森遮蔽分解与 N=2..64 题内 Z-Score 交叉
    het_rm_b = evaluate_real_prompt_heterogeneity(
        [(p["subset"], p["rm_b"], p["is_correct"]) for p in all_probs],
        n_values=n_values,
        n_folds=5,
        n_trials_per_problem=12,
        n_bins=15,
        n_simpson_bins=12,
        seed=830,
    )
    het_rm_a = evaluate_real_prompt_heterogeneity(
        [(p["subset"], p["rm_a"], p["is_correct"]) for p in all_probs],
        n_values=n_values,
        n_folds=5,
        n_trials_per_problem=12,
        n_bins=15,
        n_simpson_bins=12,
        seed=831,
    )

    out = {
        "n_total_problems": len(all_probs),
        "joint_mn_sota_rm_b": joint_rm_b,
        "dual_rm_5fold": dual_res,
        "het_7strata_rm_b": het_rm_b,
        "het_7strata_rm_a": het_rm_a,
    }
    _save_stage_cache("s4_acemath", out)
    return out


def export_summary_csv(payload: dict[str, Any], out_csv: Path) -> None:
    """导出 Stage 2 跨数据集扁平核心指标对比表至 CSV。"""
    rows: list[dict[str, Any]] = []

    def _add_cv_block(dataset: str, block: dict[str, Any], n_idx: int = -1) -> None:
        nv = int(block["n_values"][n_idx])
        for m_name, d in block["cv_results"].items():
            rows.append({
                "stage": dataset,
                "metric_group": "multi_rm_5fold",
                "method": m_name,
                "n_budget": nv,
                "accuracy": round(float(d["accuracy"][n_idx]), 6),
                "se": round(float(d["se"][n_idx]), 6),
                "extra": f"rho1={block['symmetric_diag']['mean_rho1']:.3f};rho0={block['symmetric_diag']['mean_rho0']:.3f};eta0={block['copula_eta0']:.3f}",
            })

    if "s1a_rb2" in payload:
        s1a = payload["s1a_rb2"]
        for k in ("hetero6_all", "hetero6_plus_deberta_all", "homo4_all", "hetero6_ties", "hetero6_plus_deberta_ties", "homo4_ties"):
            _add_cv_block(f"S1a_RB2_{k}", s1a[k])

    if "s1b_rb1" in payload:
        s1b = payload["s1b_rb1"]
        for k in ("code_5strong", "code_6with_deberta", "math_prm_5strong", "safety_5strong", "all_5strong", "all_6with_deberta"):
            _add_cv_block(f"S1b_RB1_{k}", s1b[k])

    if "s2_olympiadbench" in payload:
        s2 = payload["s2_olympiadbench"]
        _add_cv_block("S2_Oly_Cross4", s2["cross_prm_4views_70b_32b"])
        _add_cv_block("S2_Oly_Pair70_32", s2["pair_prm_70b_vs_32b"])
        _add_cv_block("S2_Oly_Same70_4Views", s2["same_prm_70b_4views"])
        d5 = s2["discrete_5fold_70b"]
        for m_name, d in d5["algorithms"].items():
            rows.append({
                "stage": "S2_Oly70B_Discrete",
                "metric_group": "discrete_5fold",
                "method": m_name,
                "n_budget": int(d5["n_values"][-1]),
                "accuracy": round(float(d["accuracy"][-1]), 6),
                "se": round(float(d["se"][-1]), 6),
                "extra": f"n_probs={s2['sanity_check']['n_problems']}",
            })

    if "s3_strategic_ttc" in payload:
        s3 = payload["s3_strategic_ttc"]
        for m_name, d in s3["gpqa_7b_joint_mn_sota"]["sota_comparison"].items():
            rows.append({
                "stage": "S3_GPQA7B_SOTA_M300",
                "metric_group": "sota_calibrator",
                "method": m_name,
                "n_budget": int(s3["gpqa_7b_joint_mn_sota"]["n_values"][-1]),
                "accuracy": round(float(d["accuracy"][-1]), 6),
                "se": round(float(d["se"][-1]), 6),
                "extra": "m_cal=300",
            })
        het3 = s3["cross_3domain_simpson_het"]
        for m_name, d in het3["algorithms"].items():
            for idx_n in (1, -1):
                rows.append({
                    "stage": "S3_3Domain_Het",
                    "metric_group": "heterogeneity",
                    "method": m_name,
                    "n_budget": int(het3["n_values"][idx_n]),
                    "accuracy": round(float(d["accuracy"][idx_n]), 6),
                    "se": round(float(d["se"][idx_n]), 6),
                    "extra": f"masked={het3['simpson_decomposition']['masked_transitions']}",
                })

    if "s4_acemath" in payload:
        s4 = payload["s4_acemath"]
        for m_name, d in s4["joint_mn_sota_rm_b"]["sota_comparison"].items():
            rows.append({
                "stage": "S4_AceMath_SOTA_M400",
                "metric_group": "sota_calibrator",
                "method": m_name,
                "n_budget": int(s4["joint_mn_sota_rm_b"]["n_values"][-1]),
                "accuracy": round(float(d["accuracy"][-1]), 6),
                "se": round(float(d["se"][-1]), 6),
                "extra": "m_cal=400,k=20",
            })
        het_b = s4["het_7strata_rm_b"]
        for m_name, d in het_b["algorithms"].items():
            for idx_n in (1, -1):
                rows.append({
                    "stage": "S4_AceMath7_Het_RMB",
                    "metric_group": "heterogeneity",
                    "method": m_name,
                    "n_budget": int(het_b["n_values"][idx_n]),
                    "accuracy": round(float(d["accuracy"][idx_n]), 6),
                    "se": round(float(d["se"][idx_n]), 6),
                    "extra": f"masked={het_b['simpson_decomposition']['masked_transitions']}",
                })

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["stage", "metric_group", "method", "n_budget", "accuracy", "se", "extra"]
        )
        writer.writeheader()
        writer.writerows(rows)


def print_summary(payload: dict[str, Any]) -> None:
    print("\n" + "=" * 86)
    print("Stage 2 跨领域真实大池实证验证汇总")
    print("=" * 86)
    if "s1a_rb2" in payload:
        s1a = payload["s1a_rb2"]
        for tag in ("hetero6_all", "hetero6_plus_deberta_all", "homo4_all", "hetero6_ties", "hetero6_plus_deberta_ties", "homo4_ties"):
            blk = s1a[tag]
            cv = blk["cv_results"]
            sd = blk["symmetric_diag"]
            best_single = max(float(cv[n]["accuracy"][-1]) for n in blk["verifier_names"])
            print(
                f"[S1a {tag:25s}] P={blk['n_problems']:4d}, N={int(blk['n_values'][-1]):2d} | "
                f"rho1={sd['mean_rho1']:.3f}, rho0={sd['mean_rho0']:.3f}, eta0={blk['copula_eta0']:.3f} | "
                f"Best1={best_single:.4f} | Mean={cv['Mean-Ensemble']['accuracy'][-1]:.4f} | "
                f"Min={cv['Min-Ensemble']['accuracy'][-1]:.4f} | GenEig={cv['GenEig-OptLinear']['accuracy'][-1]:.4f} | "
                f"CalMin={cv['Calibrated-Min (KL-LCB)']['accuracy'][-1]:.4f}"
            )
    if "s1b_rb1" in payload:
        s1b = payload["s1b_rb1"]
        for tag in ("code_5strong", "code_6with_deberta", "math_prm_5strong", "safety_5strong", "all_5strong", "all_6with_deberta"):
            blk = s1b[tag]
            cv = blk["cv_results"]
            sd = blk["symmetric_diag"]
            best_single = max(float(cv[n]["accuracy"][-1]) for n in blk["verifier_names"])
            print(
                f"[S1b {tag:20s}] P={blk['n_problems']:4d} | rho1={sd['mean_rho1']:.3f}, rho0={sd['mean_rho0']:.3f}, eta0={blk['copula_eta0']:.3f} | "
                f"Best1={best_single:.4f} | Mean={cv['Mean-Ensemble']['accuracy'][-1]:.4f} | "
                f"Min={cv['Min-Ensemble']['accuracy'][-1]:.4f} | Logit={cv['Logit-Stack']['accuracy'][-1]:.4f} | "
                f"CalMin={cv['Calibrated-Min (KL-LCB)']['accuracy'][-1]:.4f}"
            )
    if "s2_olympiadbench" in payload:
        s2 = payload["s2_olympiadbench"]
        chk = s2["sanity_check"]
        print(
            f"[S2 OlympiadBench Check] N_prob={chk['n_problems']} | "
            f"diff_70b(n1={chk['diff_70b_n1']:.4f}, n128={chk['diff_70b_n128']:.4f}, maj128={chk['diff_70b_maj128']:.4f}) | "
            f"diff_32b(n1={chk['diff_32b_n1']:.4f}, n128={chk['diff_32b_n128']:.4f})"
        )
        d5 = s2["discrete_5fold_70b"]["algorithms"]
        print(
            f"[S2 Oly-70B Discrete N=128] StdBoN={d5['Standard-BoN']['accuracy'][-1]:.4f} | "
            f"Maj={d5['Majority-Voting']['accuracy'][-1]:.4f} | "
            f"CalWeighted-LCB={d5['Calibrated-Weighted-BoN (Wilson-LCB)']['accuracy'][-1]:.4f}"
        )
        for tag in ("pair_prm_70b_vs_32b", "cross_prm_4views_70b_32b", "same_prm_70b_4views"):
            blk = s2[tag]
            cv = blk["cv_results"]
            sd = blk["symmetric_diag"]
            print(
                f"[S2 {tag:24s}] rho1={sd['mean_rho1']:.3f}, rho0={sd['mean_rho0']:.3f} | "
                f"Mean={cv['Mean-Ensemble']['accuracy'][-1]:.4f}, Min={cv['Min-Ensemble']['accuracy'][-1]:.4f}, "
                f"GenEig={cv['GenEig-OptLinear']['accuracy'][-1]:.4f}, CalMin={cv['Calibrated-Min (KL-LCB)']['accuracy'][-1]:.4f}"
            )
    if "s3_strategic_ttc" in payload:
        s3 = payload["s3_strategic_ttc"]
        for tag, d in s3["single_rm_diagnostics"].items():
            dg = d["diag"]
            c = d["exact_bon_accuracy"]
            print(f"[S3 {tag:14s}] P={dg['n_problems']:4d}, p={dg['p_hat']:.3f}, kappa={dg['kappa']:.3f} | BoN(1->32->128): {c[0]:.4f}->{c[5]:.4f}->{c[-1]:.4f}")
        het3 = s3["cross_3domain_simpson_het"]
        alg3 = het3["algorithms"]
        print(
            f"[S3 3-Domain Het] masked={het3['simpson_decomposition']['masked_transitions']} | "
            f"N=2 (Std={alg3['Standard-BoN']['accuracy'][1]:.4f}, Pool={alg3['Pooled-KL-LCB']['accuracy'][1]:.4f}, "
            f"Z={alg3['Within-Prompt-Z-KL-LCB']['accuracy'][1]:.4f}, Hyb={alg3['Hybrid-Adaptive-KL-LCB']['accuracy'][1]:.4f}) | "
            f"N=128 (Std={alg3['Standard-BoN']['accuracy'][-1]:.4f}, Pool={alg3['Pooled-KL-LCB']['accuracy'][-1]:.4f}, "
            f"Z={alg3['Within-Prompt-Z-KL-LCB']['accuracy'][-1]:.4f}, Hyb={alg3['Hybrid-Adaptive-KL-LCB']['accuracy'][-1]:.4f})"
        )
    if "s4_acemath" in payload:
        s4 = payload["s4_acemath"]
        sota = s4["joint_mn_sota_rm_b"]["sota_comparison"]
        sota_str = " | ".join(f"{k}={v['accuracy'][-1]:.4f}" for k, v in sota.items())
        print(f"[S4 AceMath SOTA M=400 N=64] {sota_str}")
        hb = s4["het_7strata_rm_b"]
        ab = hb["algorithms"]
        print(
            f"[S4 AceMath 7-Strata RM-B] masked={hb['simpson_decomposition']['masked_transitions']} | "
            f"N=2 (Pool={ab['Pooled-KL-LCB']['accuracy'][1]:.4f}, Z={ab['Within-Prompt-Z-KL-LCB']['accuracy'][1]:.4f}) | "
            f"N=64 (Std={ab['Standard-BoN']['accuracy'][-1]:.4f}, Pool={ab['Pooled-KL-LCB']['accuracy'][-1]:.4f}, "
            f"Z={ab['Within-Prompt-Z-KL-LCB']['accuracy'][-1]:.4f}, Hyb={ab['Hybrid-Adaptive-KL-LCB']['accuracy'][-1]:.4f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 真实大池实证验证脚本")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        choices=["all", "s1a", "s1b", "s2", "s3", "s4"],
        help="选择要运行的实验阶段（默认 all）",
    )
    parser.add_argument("--force", action="store_true", help="忽略本地阶段缓存并强制重算")
    args = parser.parse_args()

    run_all = "all" in args.stages
    t0 = time.perf_counter()
    payload: dict[str, Any] = {}

    if run_all or "s1a" in args.stages:
        payload["s1a_rb2"] = run_s1a_rb2(force=args.force)
    if run_all or "s1b" in args.stages:
        payload["s1b_rb1"] = run_s1b_rb1(force=args.force)
    if run_all or "s2" in args.stages:
        payload["s2_olympiadbench"] = run_s2_olympiadbench(force=args.force)
    if run_all or "s3" in args.stages:
        payload["s3_strategic_ttc"] = run_s3_strategic_ttc(force=args.force)
    if run_all or "s4" in args.stages:
        payload["s4_acemath"] = run_s4_acemath(force=args.force)

    elapsed = time.perf_counter() - t0
    payload["metadata"] = {"elapsed_seconds": elapsed}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "stage2_real_results.json"
    out_csv = RESULTS_DIR / "stage2_summary.csv"
    out_json.write_text(json.dumps(to_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    export_summary_csv(payload, out_csv)
    print_summary(payload)
    print(f"\nStage 2 全部实验完成！总耗时: {elapsed:.2f}s\nJSON: {out_json}\nCSV:  {out_csv}")


if __name__ == "__main__":
    main()
