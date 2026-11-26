"""Stage 2 跨领域真实大池与异构多 RM 实证 300 DPI 英文学术图表绘制脚本。

从 `results/stage2_real_results.json` 读取全部实测数据，生成 3 张出版级英文图表：
1. `results/figures/stage2_heterogeneous_multi_rm.png`（图 8）
2. `results/figures/stage2_joint_mn_and_sota_calibrators.png`（图 9）
3. `results/figures/stage2_real_simpson_and_cross_domain.png`（图 10）
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
    "axes.titlesize": 11.0,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.0,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "legend.fontsize": 8.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.45, color="#B0B8C4")
    ax.set_axisbelow(True)


def plot_fig8_heterogeneous_multi_rm(data: dict[str, Any], out_path: Path) -> None:
    """绘制图 8：跨家族异构 vs 同源 RM 条件相关性与 Copula eta0（左）+ OlympiadBench 7B 双 PRM（右）。"""
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 6.5))
    fig.subplots_adjust(wspace=0.24, bottom=0.30, top=0.86, left=0.06, right=0.97)

    s1a = data["s1a_rb2"]
    s1b = data["s1b_rb1"]
    s2 = data["s2_olympiadbench"]

    # Panel (a): Cross-family vs same-family rho1, rho0, eta0
    ax = axes[0]
    _style_axis(ax)
    groups = [
        ("RB2 Homo-4\n(AllenAI)", s1a["homo4_all"]),
        ("RB2 Hetero-6\n(All)", s1a["hetero6_all"]),
        ("RB2 Ties\n(Hetero-6)", s1a["hetero6_ties"]),
        ("RB1 Code\n(6-Family)", s1b["code_6with_deberta"]),
        ("Oly 70B\n(4-Views)", s2["same_prm_70b_4views"]),
        ("Oly Cross\n(70B vs 32B)", s2["pair_prm_70b_vs_32b"]),
    ]
    labels = [g[0] for g in groups]
    rho1_vals = [float(g[1]["symmetric_diag"]["mean_rho1"]) for g in groups]
    rho0_vals = [float(g[1]["symmetric_diag"]["mean_rho0"]) for g in groups]
    eta0_vals = [float(g[1]["copula_eta0"]) for g in groups]

    x = np.arange(len(labels))
    w = 0.25
    b1 = ax.bar(x - w, rho1_vals, width=w, color="#0072B2", label=r"Correct Corr. $\bar{\rho}_1$")
    b2 = ax.bar(x, rho0_vals, width=w, color="#D55E00", label=r"Wrong Corr. $\bar{\rho}_0$")
    b3 = ax.bar(x + w, eta0_vals, width=w, color="#009E73", label=r"Copula Residual Tail $\eta_0$")

    for bars in (b1, b2, b3):
        for rect in bars:
            h = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                h + 0.015,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.3,
                color="#1F2937",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.4)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Correlation / Tail Dependence Coefficient")
    ax.set_title("(a) Cross-Family vs. Homogeneous Conditional Dependence")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=3,
        frameon=True, facecolor="white", framealpha=0.95, edgecolor="#D5D8DC",
    )

    # Panel (b): OlympiadBench Qwen2.5-7B (N = 1..128)
    ax = axes[1]
    _style_axis(ax)
    d5 = s2["discrete_5fold_70b"]
    cross4 = s2["cross_prm_4views_70b_32b"]
    pair = s2["pair_prm_70b_vs_32b"]
    n_vals = np.asarray(d5["n_values"], dtype=float)

    curves = [
        (d5["algorithms"]["Calibrated-Weighted-BoN (Wilson-LCB)"], "Calibrated-Weighted-BoN (70B, Wilson-LCB)", "#0072B2", "o", "-"),
        (d5["algorithms"]["Majority-Voting"], "Majority-Voting (Unweighted)", "#56B4E9", "^", ":"),
        (cross4["cv_results"]["Logit-Stack"], "Logit-Stack (70B+32B Cross-4)", "#009E73", "s", "-"),
        (cross4["cv_results"]["Mean-Ensemble"], "Mean-Ensemble (70B+32B Cross-4)", "#CC79A7", "P", "--"),
        (pair["cv_results"]["uPRM-70B-Last"], "Single uPRM-70B-Last (5-Fold Exact)", "#D55E00", "D", "-."),
        (pair["cv_results"]["uPRM-32B-Last"], "Single uPRM-32B-Last (5-Fold Exact)", "#7F8C8D", "v", "-."),
    ]
    for src, lbl, col, mk, ls in curves:
        acc = np.asarray(src["accuracy"], dtype=float)
        err = 1.96 * np.asarray(src["se"], dtype=float)
        ax.errorbar(
            n_vals, acc, yerr=err, fmt=mk, linestyle=ls, linewidth=1.9, markersize=4.4,
            color=col, capsize=2.0, label=lbl, zorder=4, markeredgecolor="white", markeredgewidth=0.4,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(n_vals)
    ax.set_xticklabels([str(int(v)) for v in n_vals])
    ax.set_xlabel("Sampling Budget $N$ (log$_2$ scale)")
    ax.set_ylabel("OlympiadBench Pass@1 Accuracy")
    ax.set_title("(b) OlympiadBench (Qwen2.5-7B, uPRM-70B & 32B, $N=1\\dots 128$)")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2,
        frameon=True, facecolor="white", framealpha=0.95, edgecolor="#D5D8DC",
    )

    fig.suptitle(
        "Stage 2 Multi-Verifier Scaling: Homogeneous Lock-In vs. Cross-Family Fusion & OlympiadBench 7B",
        fontsize=12.2, fontweight="bold", y=0.95,
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig9_joint_mn_and_sota(data: dict[str, Any], out_path: Path) -> None:
    """绘制图 9：AceMath (M=400) 8 类 SOTA 校准器对比（左）+ 2D (M, N) 热力图与 N_balance(M)（右）。"""
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 6.4))
    fig.subplots_adjust(wspace=0.25, bottom=0.28, top=0.86, left=0.065, right=0.96)

    ace = data["s4_acemath"]["joint_mn_sota_rm_b"]
    sota = ace["sota_comparison"]
    n_vals = np.asarray(ace["n_values"], dtype=float)

    # Panel (a): 8 SOTA Calibrators on AceMath (M=400, K=20)
    ax = axes[0]
    _style_axis(ax)
    sota_styles = [
        ("standard_bon", "Standard-BoN (Raw Score)", "#1F2937", "x", "--"),
        ("platt_scaling", "Platt Scaling (Logistic)", "#7F8C8D", "+", ":"),
        ("isotonic_pava", "Isotonic Regression (PAVA)", "#CC79A7", "v", "-."),
        ("histogram_mle", "Histogram MLE", "#D55E00", "D", ":"),
        ("laplace_smoothed", "Laplace Smoothed (s=2)", "#E69F00", "^", "--"),
        ("wilson_lcb", "Wilson-LCB", "#56B4E9", "s", "-"),
        ("kl_lcb", "KL-LCB (Chernoff-Sanov)", "#0072B2", "o", "-"),
        ("empirical_bernstein_lcb", "Empirical-Bernstein LCB", "#009E73", "P", "-."),
    ]
    for key, lbl, col, mk, ls in sota_styles:
        d = sota[key]
        acc = np.asarray(d["accuracy"], dtype=float)
        lo = np.asarray(d["ci_low"], dtype=float)
        hi = np.asarray(d["ci_high"], dtype=float)
        ax.plot(n_vals, acc, color=col, marker=mk, linestyle=ls, linewidth=1.9, markersize=4.5, label=lbl, zorder=3)
        ax.fill_between(n_vals, lo, hi, color=col, alpha=0.10, zorder=2)

    ax.annotate(
        "Empirical-Bernstein collapses\n(0.6032 vs 0.6962 for KL-LCB)",
        xy=(64, float(sota["empirical_bernstein_lcb"]["accuracy"][-1])),
        xytext=(10, 0.545),
        arrowprops=dict(arrowstyle="->", color="#009E73", lw=1.2),
        fontsize=8.3, fontweight="bold", color="#065F46",
        bbox=dict(boxstyle="round,pad=0.3", fc="#ECFDF5", ec="#009E73", lw=0.8),
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(n_vals)
    ax.set_xticklabels([str(int(v)) for v in n_vals])
    ax.set_xlabel("Sampling Budget $N$ (log$_2$ scale)")
    ax.set_ylabel("AceMath Holdout Pass@1 Accuracy")
    ax.set_title("(a) 8 SOTA Calibrators on AceMath-RewardBench ($M=400, K=20$)")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
        frameon=True, facecolor="white", framealpha=0.95, edgecolor="#D5D8DC",
    )

    # Panel (b): 2D (M, N) Advantage Heatmap (Opt 1/5 + KL-LCB minus Fixed-60 + MLE, pp)
    ax = axes[1]
    m_grid = np.asarray(ace["m_grid"], dtype=int)
    diff_pp = (
        np.asarray(ace["grid_opt_1_5_kl"], dtype=float)
        - np.asarray(ace["grid_fixed60_mle"], dtype=float)
    ) * 100.0
    # 读取 JSON 中的历史键名 n_match_theory_by_m，但在图表显示中统一使用 N_balance(M)
    n_balance = np.asarray(ace["n_match_theory_by_m"], dtype=float)

    vmax = max(1.0, float(np.max(np.abs(diff_pp))))
    im = ax.imshow(diff_pp, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(n_vals.size))
    ax.set_xticklabels([str(int(v)) for v in n_vals])
    ax.set_yticks(np.arange(m_grid.size))
    ax.set_yticklabels([f"{m:,}" for m in m_grid])

    # 叠加 Theorem 6C 理论交界线 N_balance(M)（沿单元格右边界阶梯线/淡化衬底绘制，确保热力图数字零遮挡）
    x_bal = np.interp(np.log2(np.clip(n_balance, n_vals[0], n_vals[-1])), np.log2(n_vals), np.arange(n_vals.size))
    import matplotlib.patheffects as pe

    for i in range(m_grid.size):
        for j in range(n_vals.size):
            val = diff_pp[i, j]
            txt_col = "white" if abs(val) > 0.55 * vmax else "#111827"
            halo_col = "#111827" if abs(val) > 0.55 * vmax else "#F8FAFC"
            ax.text(
                j, i, f"{val:+.2f}", ha="center", va="center", color=txt_col,
                fontsize=7.6, fontweight="bold", zorder=6,
                path_effects=[pe.withStroke(linewidth=2.8, foreground=halo_col, alpha=0.92)],
            )

    # 在数字下方绘制 Theorem 6C 理论交界线 N_balance(M)
    ax.plot(
        x_bal, np.arange(m_grid.size),
        color="#111827", linestyle="--", linewidth=2.4, alpha=0.78, zorder=3,
        label=r"Theorem 6C Boundary $N_{\mathrm{balance}}(M)$",
    )
    ax.plot(
        x_bal, np.arange(m_grid.size),
        color="#F59E0B", linestyle="--", linewidth=1.4, alpha=0.90, zorder=3.5,
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.03)
    cbar.set_label(r"Pass@1 Gain: Opt $K^*(M)^{1/5}$+KL-LCB vs. Fixed-60 MLE (pp)", fontsize=8.8)

    ax.set_xlabel("Inference Sampling Budget $N$")
    ax.set_ylabel("Offline Calibration Samples $M$")
    ax.set_title("(b) 2D $(M, N)$ Gain Surface (pp) & Theorem 6C $N_{\\mathrm{balance}}(M)$")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1,
        frameon=True, facecolor="white", framealpha=0.95, edgecolor="#D5D8DC",
    )

    fig.suptitle(
        "Stage 2 Calibration Scaling: 8 SOTA Baselines & Joint (M, N) Regime on AceMath (8,987 Problems)",
        fontsize=12.2, fontweight="bold", y=0.95,
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig10_simpson_and_cross_domain(data: dict[str, Any], out_path: Path) -> None:
    """绘制图 10：S3 三域真实辛普森倒挂分解（左）+ 定理 7B 真实双向交叉（右）。"""
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 6.4))
    fig.subplots_adjust(wspace=0.24, bottom=0.28, top=0.86, left=0.065, right=0.97)

    het3 = data["s3_strategic_ttc"]["cross_3domain_simpson_het"]
    het7 = data["s4_acemath"]["het_7strata_rm_b"]

    # Panel (a): 3-Domain Simpson Inversion (GSM8K + GPQA + AIME, ArmoRM-8B)
    ax = axes[0]
    _style_axis(ax)
    sd3 = het3["simpson_decomposition"]
    q_pool = np.asarray(sd3["q_pool_bins"], dtype=float)
    q_strata = np.asarray(sd3["q_strata_bins"], dtype=float)
    w_post = np.asarray(sd3["strata_post_weights"], dtype=float)
    s_names = het3["strata_names"]
    bins_x = np.arange(1, q_pool.size + 1)

    ax.plot(bins_x, q_pool, color="#111827", linewidth=2.6, marker="o", markersize=5.2,
            label="Pooled Empirical Posterior $q_{\\mathrm{pool}}(b)$", zorder=5)

    st_styles = {
        "GSM8K": ("#009E73", "^", "--", "Within-GSM8K (p=0.907, mean=0.164)"),
        "GPQA": ("#0072B2", "s", "-.", "Within-GPQA (p=0.197, mean=0.084)"),
        "AIME": ("#D55E00", "D", "-", "Within-AIME (p=0.022, mean=0.267)"),
    }
    for idx, name in enumerate(s_names):
        col, mk, ls, lbl = st_styles[name]
        mask = (~np.isnan(q_strata[idx])) & (w_post[idx] >= 0.005)
        ax.plot(bins_x[mask], q_strata[idx, mask], color=col, marker=mk, linestyle=ls,
                linewidth=1.9, markersize=4.5, label=lbl, zorder=4)

    ax.annotate(
        "Reverse Simpson Trap:\nAIME scores higher than GSM8K,\npulling $q_{\\mathrm{pool}}$ down to 0.086!",
        xy=(11, float(q_pool[10]) + 0.015),
        xytext=(8.35, 0.57),
        arrowprops=dict(arrowstyle="->", color="#B91C1C", lw=1.3),
        fontsize=8.1, fontweight="bold", color="#7F1D1D",
        bbox=dict(boxstyle="round,pad=0.35", fc="#FEF2F2", ec="#DC2626", lw=0.9),
        zorder=6,
    )

    ax.set_xticks(bins_x)
    ax.set_xlabel("Global ArmoRM Score Quantile Decile/Bin (Low $\\rightarrow$ High)")
    ax.set_ylabel("Empirical Correctness Probability")
    ax.set_ylim(-0.03, 1.04)
    ax.set_title("(a) Real Reverse Simpson Paradox: GSM8K + GPQA + AIME (Qwen2.5-7B)")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
        frameon=True, facecolor="white", framealpha=0.95, edgecolor="#D5D8DC",
    )

    # Panel (b): Theorem 7B Adaptive Crossover (Delta vs Pooled-KL-LCB in pp)
    ax = axes[1]
    _style_axis(ax)
    ax.axhline(0.0, color="#1F2937", linestyle="--", linewidth=1.4, label="Baseline: Pooled-KL-LCB (0.0 pp)")

    def _plot_delta(het_blk: dict[str, Any], prefix: str, algo_key: str, display_name: str, col: str, mk: str, ls: str) -> None:
        nv = np.asarray(het_blk["n_values"], dtype=float)
        base = np.asarray(het_blk["algorithms"]["Pooled-KL-LCB"]["accuracy"], dtype=float)
        targ = np.asarray(het_blk["algorithms"][algo_key]["accuracy"], dtype=float)
        delta_pp = (targ - base) * 100.0
        ax.plot(nv, delta_pp, color=col, marker=mk, linestyle=ls, linewidth=2.0, markersize=4.8,
                label=f"{prefix}: {display_name}", zorder=4)

    _plot_delta(het3, "S3 3-Domain", "Within-Prompt-Z-KL-LCB", "Within-Prompt-Z-KL-LCB", "#0072B2", "o", "-")
    _plot_delta(het3, "S3 3-Domain", "Hybrid-Adaptive-KL-LCB", "Hybrid-Fixed-KL-LCB", "#56B4E9", "s", "--")
    _plot_delta(het7, "S4 AceMath-7", "Within-Prompt-Z-KL-LCB", "Within-Prompt-Z-KL-LCB", "#D55E00", "D", "-.")
    _plot_delta(het7, "S4 AceMath-7", "Hybrid-Adaptive-KL-LCB", "Hybrid-Fixed-KL-LCB", "#009E73", "^", "-")

    ax.set_xscale("log", base=2)
    n_all = np.asarray(het3["n_values"], dtype=float)
    ax.set_xticks(n_all)
    ax.set_xticklabels([str(int(v)) for v in n_all])
    ax.set_xlabel("Sampling Budget $N$ (log$_2$ scale)")
    ax.set_ylabel("Pass@1 Gain vs. Pooled-KL-LCB (pp)")
    ax.set_title("(b) Theorem 7B Crossover: Within-Z vs. Hybrid-Fixed Across Real Regimes")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
        frameon=True, facecolor="white", framealpha=0.95, edgecolor="#D5D8DC",
    )

    fig.suptitle(
        "Stage 2 Prompt Heterogeneity: Cross-Domain Simpson Inversion & Adaptive Z-Score Crossover",
        fontsize=12.2, fontweight="bold", y=0.95,
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    in_path = RESULTS_DIR / "stage2_real_results.json"
    if not in_path.exists():
        raise FileNotFoundError(f"未找到 {in_path}，请先运行 python scripts/run_stage2_experiments.py")

    data = json.loads(in_path.read_text(encoding="utf-8"))

    fig8 = FIGURES_DIR / "stage2_heterogeneous_multi_rm.png"
    print(f"正在绘制 {fig8} ...")
    plot_fig8_heterogeneous_multi_rm(data, fig8)

    fig9 = FIGURES_DIR / "stage2_joint_mn_and_sota_calibrators.png"
    print(f"正在绘制 {fig9} ...")
    plot_fig9_joint_mn_and_sota(data, fig9)

    fig10 = FIGURES_DIR / "stage2_real_simpson_and_cross_domain.png"
    print(f"正在绘制 {fig10} ...")
    plot_fig10_simpson_and_cross_domain(data, fig10)

    print("Stage 2 三张学术图表（图 8–10）已生成完毕。")


if __name__ == "__main__":
    main()
