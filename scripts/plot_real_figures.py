"""真实大模型与开源 Reward Model 实证结果 300 DPI 英文学术图表绘制脚本（V3）。

从 `results/real_llm_results.json` 读取真实实验数据，生成 2 张出版级英文图表：
1. `results/figures/real_llm_theorem6_calibration.png`
2. `results/figures/real_llm_math500_scaling.png`

运行方式：
    python scripts/plot_real_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 10,
    "axes.titlesize": 11.2,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.2,
    "xtick.labelsize": 9.2,
    "ytick.labelsize": 9.2,
    "legend.fontsize": 8.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.45, color="#B0B8C4")
    ax.set_axisbelow(True)


def plot_real_theorem6_calibration(ace: dict[str, Any], out_path: Path) -> None:
    """绘制图 6：AceMath-RewardBench 上有限校准集 M=400 对比（左）与跨 M 收敛（右）。"""
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.3))
    fig.subplots_adjust(wspace=0.23, bottom=0.28, top=0.86, left=0.07, right=0.97)

    # Panel (a): M=400 等宽 K=20 实证曲线
    ax = axes[0]
    _style_axis(ax)
    r2u = ace["r2_m400_uniform_k20"]
    n_vals = np.asarray(r2u["n_values"], dtype=float)
    std_bon = np.asarray(r2u["standard_bon_accuracy"], dtype=float)
    ax.plot(n_vals, std_bon, color="#1F2937", linestyle="--", linewidth=2.0,
            marker="x", markersize=5.0, label="Standard-BoN (Raw RM Score)", zorder=4)

    styles = [
        ("wilson_lcb", "Wilson-LCB (M=400)", "#0072B2", "o"),
        ("laplace", "Laplace Smoothed (s=2.0)", "#E69F00", "^"),
        ("mle", "Naive Empirical MLE (M=400)", "#D55E00", "D"),
        ("empirical_bernstein_lcb", "Empirical-Bernstein LCB (M=400)", "#009E73", "s"),
    ]
    for key, label, color, marker in styles:
        md = r2u["methods"][key]
        acc = np.asarray(md["accuracy"], dtype=float)
        lo = np.asarray(md["ci_low"], dtype=float)
        hi = np.asarray(md["ci_high"], dtype=float)
        ax.plot(n_vals, acc, color=color, marker=marker, markersize=4.5, linewidth=2.0, label=label, zorder=3)
        ax.fill_between(n_vals, lo, hi, color=color, alpha=0.12, zorder=2)

    ax.set_xscale("log", base=2)
    ax.set_xticks(n_vals)
    ax.set_xticklabels([str(int(x)) for x in n_vals])
    ax.set_xlabel("Sampling Budget $N$ (log$_2$ scale)")
    ax.set_ylabel("Test-Set Pass@1 Accuracy")
    ax.set_title("(a) AceMath-RewardBench: Finite Calibration ($M=400$, Uniform $K=20$)")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2,
        frameon=True, facecolor="white", framealpha=0.95, edgecolor="#D5D8DC",
    )

    # Panel (b): 跨 M 扫描（N=64）
    ax = axes[1]
    _style_axis(ax)
    recs = ace["r2_m_sweep_n64"]
    m_arr = np.array([r["m_cal"] for r in recs], dtype=float)
    std64 = np.array([r["standard_bon_n64"] for r in recs], dtype=float)
    opt_w = np.array([r["opt_wilson_n64"] for r in recs], dtype=float)
    opt_w_err = 1.96 * np.array([r["opt_wilson_se"] for r in recs], dtype=float)
    fix_m = np.array([r["fixed60_mle_n64"] for r in recs], dtype=float)
    fix_m_err = 1.96 * np.array([r["fixed60_mle_se"] for r in recs], dtype=float)
    fix_w = np.array([r["fixed60_wilson_n64"] for r in recs], dtype=float)

    ax.plot(m_arr, std64, "k--", linewidth=1.6, label="Standard-BoN ($N=64$, Uncalibrated)")
    ax.errorbar(m_arr, fix_m, yerr=fix_m_err, fmt="D:", color="#D55E00", linewidth=1.9, markersize=4.8,
                capsize=2.5, label="Fixed Over-Fine $K=60$ + MLE")
    ax.plot(m_arr, fix_w, "^-.", color="#CC79A7", linewidth=1.8, markersize=4.8,
            label="Fixed $K=60$ + Wilson-LCB")
    ax.errorbar(m_arr, opt_w, yerr=opt_w_err, fmt="o-", color="#0072B2", linewidth=2.2, markersize=5.2,
                capsize=2.5, label=r"Optimal $K^*(M) \propto (M/\ln M)^{2/5}$ + Wilson-LCB")

    ax.set_xscale("log")
    ax.set_xlabel("Offline Calibration Sample Size $M$ (log scale)")
    ax.set_ylabel("Holdout Test Pass@1 at $N=64$")
    ax.set_title("(b) Calibration Sample Scaling on 4,494 Holdout Math Problems")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2,
        frameon=True, facecolor="white", framealpha=0.95, edgecolor="#D5D8DC",
    )

    fig.suptitle(
        "Real-World Validation on AceMath-RewardBench (8,987 Problems, 575k Scored Completions)",
        fontsize=12.4, fontweight="bold", y=0.95,
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_real_math500_scaling(res_1b: dict[str, Any], res_3b: dict[str, Any], out_path: Path) -> None:
    """绘制图 7：MATH-500 5-Fold 算法对比（左）与多 PRM 视图相关性诊断矩阵（右）。"""
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.4))
    fig.subplots_adjust(wspace=0.25, bottom=0.28, top=0.86, left=0.065, right=0.96)

    # Panel (a): Llama-3.2-1B & 3B 5-Fold 对比
    ax = axes[0]
    _style_axis(ax)
    r4_1b = res_1b["r4_5fold_algorithms"]
    r4_3b = res_3b["r4_5fold_algorithms"]
    n_vals = np.asarray(r4_1b["n_values"], dtype=float)

    plot_items = [
        (r4_3b, "Two-Stage-Verified-Majority (Wilson-LCB)", "3B: Two-Stage-Majority (LCB)", "#0072B2", "o", "-"),
        (r4_3b, "Calibrated-Weighted-BoN (Wilson-LCB)", "3B: Calibrated-Weighted (LCB)", "#009E73", "s", "--"),
        (r4_3b, "Majority-Voting", "3B: Majority-Voting", "#7F8C8D", "^", ":"),
        (r4_3b, "Standard-BoN", "3B: Standard-BoN (PRM-Last)", "#D55E00", "D", "-."),
        (r4_1b, "Two-Stage-Verified-Majority (Wilson-LCB)", "1B: Two-Stage-Majority (LCB)", "#56B4E9", "o", "-"),
        (r4_1b, "Standard-BoN", "1B: Standard-BoN (PRM-Last)", "#E69F00", "D", "-."),
    ]
    for src, key, label, color, marker, ls in plot_items:
        d = src["algorithms"][key]
        acc = np.asarray(d["accuracy"], dtype=float)
        err = 1.96 * np.asarray(d["se"], dtype=float)
        ax.errorbar(
            n_vals, acc, yerr=err, fmt=marker, linestyle=ls, linewidth=1.9, markersize=4.3,
            color=color, capsize=2.0, label=label, zorder=4,
            markeredgecolor="white", markeredgewidth=0.45,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(n_vals)
    ax.set_xticklabels([str(int(x)) for x in n_vals])
    ax.set_xlabel("Sampling Budget $N$ (log$_2$ scale)")
    ax.set_ylabel("MATH-500 5-Fold Cross-Validated Pass@1")
    ax.set_title("(a) MATH-500 Scaling: Llama-3.2-1B & 3B (N = 1 to 256)")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2,
        frameon=True, facecolor="white", framealpha=0.95, edgecolor="#D5D8DC",
    )

    # Panel (b): 多 PRM 视图相关性矩阵热图（上三角 rho1，下三角 rho0）
    ax = axes[1]
    md = res_3b["r3_multi_prm_correlation"]
    names = md["names"]
    r1 = np.asarray(md["corr_pos"], dtype=float)
    r0 = np.asarray(md["corr_neg"], dtype=float)
    k = len(names)

    # 组合矩阵：上三角填 rho_1 (Correct)，下三角填 rho_0 (Wrong)，对角置 1.0
    combined = np.eye(k, dtype=float)
    for i in range(k):
        for j in range(k):
            if i < j:
                combined[i, j] = r1[i, j]
            elif i > j:
                combined[i, j] = r0[i, j]

    im = ax.imshow(combined, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(k))
    ax.set_yticks(np.arange(k))
    ax.set_xticklabels(names, fontsize=9.2)
    ax.set_yticklabels(names, fontsize=9.2)
    for i in range(k):
        for j in range(k):
            val = combined[i, j]
            tag = ""
            if i < j:
                tag = "\n(pos)"
            elif i > j:
                tag = "\n(neg)"
            txt_color = "white" if val > 0.58 else "#111827"
            ax.text(j, i, f"{val:.3f}{tag}", ha="center", va="center", color=txt_color, fontsize=8.4, fontweight="bold")

    # 绘制对角分割线
    ax.plot([-0.5, k - 0.5], [-0.5, k - 0.5], color="#C0392B", linewidth=1.6, linestyle="--")
    cbar = fig.colorbar(im, ax=ax, pad=0.03)
    cbar.set_label(r"Pairwise Pearson Correlation ($\rho_1$ Upper / $\rho_0$ Lower)", fontsize=9.2)

    diag5 = md["theorem5a_diagnosis"]
    rho1_m = float(md["mean_rho1"])
    rho0_m = float(md["mean_rho0"])
    kap_m = float(md["mean_kappa"])
    ax.set_title(r"(b) 3B PRM Views Correlation ($\rho_1$ Upper vs $\rho_0$ Lower)")
    # 把诊断说明放到图下方空白处（对齐左图图例位置）
    ax.text(
        0.5, -0.25,
        f"Theorem 5A Empirical Diagnosis (Llama-3.2-3B + RLHFlow-8B-PRM):\n"
        f"Mean kappa = {kap_m:.3f}, Mean rho_1 = {rho1_m:.3f}, Mean rho_0 = {rho0_m:.3f}\n"
        f"Regime: {diag5['regime']} (kappa_eff = {diag5['kappa_eff']:.3f})",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.6,
        bbox=dict(boxstyle="round,pad=0.35", fc="#F8F9F9", ec="#BDC3C7", lw=0.9),
    )

    fig.suptitle(
        "Real-World Validation on MATH-500: Llama-3.2-1B / 3B with RLHFlow-8B-PRM",
        fontsize=12.4, fontweight="bold", y=0.95,
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    in_path = RESULTS_DIR / "real_llm_results.json"
    if not in_path.exists():
        raise FileNotFoundError(f"未找到 {in_path}，请先运行 python scripts/run_real_llm_experiments.py")

    data = json.loads(in_path.read_text(encoding="utf-8"))

    fig6 = FIGURES_DIR / "real_llm_theorem6_calibration.png"
    print(f"正在绘制 {fig6} ...")
    plot_real_theorem6_calibration(data["acemath_rewardbench"], fig6)

    fig7 = FIGURES_DIR / "real_llm_math500_scaling.png"
    print(f"正在绘制 {fig7} ...")
    plot_real_math500_scaling(data["math500_llama_3_2_1b"], data["math500_llama_3_2_3b"], fig7)

    print("真实大模型实证图表（图 6 与图 7）已生成完毕。")


if __name__ == "__main__":
    main()
