"""生成 TMLR 投稿级独立矢量图表（PDF + 300 DPI PNG）至 paper/tmlr/figures/。

严格隔离原则：绝不触碰或覆盖 NeurIPS 2026 归档图表（paper/figures/）。
全部实证数值严格读取自：
- `results/v9_consolidation_verification.json`
- `results/v10_release_verification.json`
- `results/v6_theory_verification.json`
- `results/stage2_real_results.json`
- `results/tmlr_diagnostics.json`

包含 5 组 TMLR 专属图表：
1. `fig1_framework_overview`: 对齐 TMLR 定理/命题编号 (Thm. 1, Prop. 1; Thms. 2--3, Props. 2--3; Prop. 3 Hoeffding bound)
2. `fig2_joint_mn_and_protocols`: Panel (b) 补齐 Wilson-LCB；Panel (c) 单元格数字增加背景等色描边消除虚线压字
3. `fig3_heterogeneity_and_shift`: Panel (a) 修正中性轴标并补全 -0.32 标签；Panel (c) 完整 7-Regime 矩阵对齐修复后真 i.i.d. 数据
4. `fig4_multi_rm_dependence`: Panel (b) 修正 N=4 与 5-Fold CV 标注；Panel (c) 显式标注 5-Fold CV
5. `fig5_tmlr_z_diagnostics`: TMLR 专属 3 联诊断图（Module A 一般 (pi_0,pi_1) 与左尾陷阱 + Module B 5 基准 Z-bin 轨迹 + Module C pi_0=0 消融曲线与 8-seed 稳健性）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
OUT_DIR = PROJECT_ROOT / "paper" / "tmlr" / "figures"

# 色盲安全 Okabe-Ito 学术配色
C_BLUE = "#0072B2"
C_VERMILION = "#D55E00"
C_GREEN = "#009E73"
C_AMBER = "#E69F00"
C_PURPLE = "#CC79A7"
C_SKY = "#56B4E9"
C_DARK = "#1F2937"
C_GRAY = "#6B7280"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "axes.titlesize": 10.0,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.2,
    "xtick.labelsize": 8.3,
    "ytick.labelsize": 8.3,
    "legend.fontsize": 7.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.85,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
})


def _style_ax(ax: plt.Axes) -> None:
    """设置统一坐标轴网格样式。"""
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.30, color="#9CA3AF")
    ax.set_axisbelow(True)


def _save_both(fig: plt.Figure, stem: str) -> None:
    """同时导出矢量 PDF（供 TMLR LaTeX 加载）与 300 DPI PNG（供视觉审校）。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / f"{stem}.pdf"
    png_path = OUT_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"已导出 TMLR 图表 {pdf_path.name} 与 {png_path.name}")


def _draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    fc: str,
    ec: str,
    tc: str = "#111827",
    fontsize: float = 8.0,
    bold: bool = False,
) -> None:
    """在 [0,1]x[0,1] 示意坐标系中绘制圆角流程框。"""
    box = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.15,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2.0,
        y + h / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        color=tc,
        zorder=4,
    )


def _draw_arrow(ax: plt.Axes, xy1: tuple[float, float], xy2: tuple[float, float], col: str = "#374151") -> None:
    """在流程图中绘制带清晰箭头的连线。"""
    arr = mpatches.FancyArrowPatch(
        xy1,
        xy2,
        arrowstyle="-|>,head_length=4.5,head_width=2.8",
        color=col,
        linewidth=1.15,
        zorder=2,
    )
    ax.add_patch(arr)


def plot_fig1_framework_overview() -> None:
    """绘制 Figure 1（TMLR 编号严格对齐）：双层模型边界、联合 (M,N) 相图与 Hybrid-Fixed 控制流。"""
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.70), gridspec_kw={"width_ratios": [1.08, 1.0, 1.08]})
    fig.subplots_adjust(wspace=0.22, left=0.02, right=0.99, top=0.87, bottom=0.14)

    # Panel (a): Two-level framework & scope boundary
    ax = axes[0]
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_title("(a) Hierarchical Model & Scope Boundary", pad=8)

    _draw_box(
        ax, 0.18, 0.80, 0.64, 0.14,
        "Prompt $X \\sim \\mathcal{P}_X$, prior $p(X)$\n$N$ i.i.d. candidates $(Y_i, S_i) \\mid X$",
        "#F3F4F6", C_DARK, fontsize=7.9, bold=True,
    )

    _draw_arrow(ax, (0.35, 0.785), (0.24, 0.64), C_BLUE)
    _draw_box(
        ax, 0.02, 0.45, 0.43, 0.175,
        "Fixed-Prompt Regime\n$P_{Y,S\\mid X} \\equiv P$\n(Thm. 1 & Prop. 1)",
        "#EFF6FF", C_BLUE, tc="#1E3A8A", fontsize=7.6, bold=True,
    )
    _draw_arrow(ax, (0.24, 0.435), (0.24, 0.305), C_BLUE)
    _draw_box(
        ax, 0.02, 0.06, 0.43, 0.23,
        "Sample-Split KL-LCB\n$K^*(M) \\asymp (M/\\ln M)^{\\frac{1}{2\\beta+1}}$\nAll $\\beta>0$ Pyke Spacings",
        "#DBEAFE", C_BLUE, fontsize=7.4,
    )

    _draw_arrow(ax, (0.65, 0.785), (0.76, 0.64), C_VERMILION)
    _draw_box(
        ax, 0.52, 0.45, 0.46, 0.175,
        "Cross-Prompt Heterogeneity\nShifts $(b(X), a(X)),\\ p(X)$\n(Thms. 2--3, Props. 2--3)",
        "#FFF7ED", C_VERMILION, tc="#9A3412", fontsize=7.5, bold=True,
    )
    _draw_arrow(ax, (0.76, 0.435), (0.76, 0.305), C_VERMILION)
    _draw_box(
        ax, 0.52, 0.06, 0.46, 0.23,
        "Pooled: Simpson Masking\nWithin-Z: $O_p(\\sqrt{\\ln N/N})$\n& $(\\pi_0, \\pi_1)$ Tail Inversion",
        "#FFEDD5", C_VERMILION, fontsize=7.4,
    )

    ax.plot([0.485, 0.485], [0.05, 0.65], linestyle=":", color=C_GRAY, linewidth=1.1)
    ax.text(0.485, 0.675, "Remark 1\nScope Boundary", ha="center", va="bottom", fontsize=6.9, color=C_GRAY, style="italic")

    # Panel (b): Joint (M, N) Minimax Phase Diagram (exact beta=2 theoretical rate)
    ax = axes[1]
    _style_ax(ax)
    m_exp = np.linspace(2.0, 6.0, 220)
    n_exp = np.linspace(0.3, 2.7, 200)
    M_grid, N_grid = np.meshgrid(10.0**m_exp, 10.0**n_exp, indexing="xy")

    beta = 2.0
    r_off = (np.log(M_grid) / M_grid) ** (beta / (2.0 * beta + 1.0))
    r_on = N_grid ** (-beta)
    log_regret = np.log10(r_off + r_on)

    cf = ax.contourf(m_exp, n_exp, log_regret, levels=16, cmap="YlGnBu_r", alpha=0.90)
    cbar = fig.colorbar(cf, ax=ax, pad=0.02, shrink=0.94)
    cbar.set_label(r"$\log_{10}\mathcal{R}^*(M,N)$ ($\beta=2$)", fontsize=8.0)
    cbar.ax.tick_params(labelsize=7.5)

    m_line = 10.0**m_exp
    n_bal = (m_line / np.log(m_line)) ** (1.0 / (2.0 * beta + 1.0))
    ax.plot(
        m_exp, np.log10(n_bal), color=C_VERMILION, linestyle="--", linewidth=2.2, zorder=5,
        label=r"$N_{\mathrm{balance}}(M)\asymp(M/\ln M)^{\frac{1}{2\beta+1}}$",
    )

    ax.text(
        2.20, 2.24, "Calibration-Dominated\n$N \\gg N_{\\mathrm{balance}}(M)$\n(More $N$ saturates)",
        color="white", fontsize=7.5, fontweight="bold",
        path_effects=[pe.withStroke(linewidth=2.0, foreground="#111827", alpha=0.85)],
    )
    ax.text(
        4.05, 0.80, "Compute-Dominated\n$N \\ll N_{\\mathrm{balance}}(M)$\n(Scale $N$ pays off)",
        color="#111827", fontsize=7.5, fontweight="bold",
        path_effects=[pe.withStroke(linewidth=2.0, foreground="white", alpha=0.85)],
    )

    ax.set_xlim(2.0, 6.0)
    ax.set_ylim(0.3, 2.65)
    ax.set_xticks([2, 3, 4, 5, 6])
    ax.set_xticklabels([r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$", r"$10^6$"])
    ax.set_yticks([0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_yticklabels(["3", "10", "32", "100", "316"])
    ax.set_xlabel("Offline Calibration Labels $M$")
    ax.set_ylabel("Online Compute Budget $N$")
    ax.set_title("(b) Joint $(M, N)$ Minimax Phase Regimes", pad=8)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#D1D5DB", fontsize=7.2)

    # Panel (c): Three-Branch Holdout Risk-Control (Hybrid-Fixed)
    ax = axes[2]
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_title("(c) Risk-Controlled Selector (Hybrid-Fixed)", pad=8)

    _draw_box(
        ax, 0.04, 0.78, 0.92, 0.16,
        "Disjoint Prompt Split: Train (50%) / Val (20%) / Test (30%)",
        "#F3F4F6", C_DARK, fontsize=7.8, bold=True,
    )

    _draw_arrow(ax, (0.20, 0.765), (0.17, 0.625), C_GRAY)
    _draw_arrow(ax, (0.50, 0.765), (0.50, 0.625), C_BLUE)
    _draw_arrow(ax, (0.80, 0.765), (0.83, 0.625), C_GREEN)

    _draw_box(ax, 0.02, 0.45, 0.29, 0.16, "Raw-BoN\n(Safe on clean\nheterogeneous)", "#F9FAFB", C_GRAY, fontsize=7.2)
    _draw_box(ax, 0.345, 0.45, 0.31, 0.16, "Pooled-KL-LCB\n(Separate baseline:\n+Support Clip)", "#EFF6FF", C_BLUE, fontsize=7.1)
    _draw_box(ax, 0.685, 0.45, 0.295, 0.16, "Within-Z-KL-LCB\n(Shift-robust via\ntail downweight)", "#ECFDF5", C_GREEN, fontsize=7.2)

    _draw_arrow(ax, (0.17, 0.435), (0.44, 0.31), C_GRAY)
    _draw_arrow(ax, (0.50, 0.435), (0.50, 0.31), C_BLUE)
    _draw_arrow(ax, (0.83, 0.435), (0.56, 0.31), C_GREEN)

    _draw_box(
        ax, 0.06, 0.135, 0.88, 0.16,
        "Holdout Gate on Val: $\\hat{m} = \\arg\\max_{m\\in\\mathcal{M}} \\hat{A}_{\\mathrm{val}}(m)$\nFreeze predictors ($\\varepsilon_{\\mathrm{refit}} = 0$, Prop. 3 Hoeffding bound)",
        "#FEF3C7", C_AMBER, tc="#92400E", fontsize=7.4, bold=True,
    )
    _draw_arrow(ax, (0.50, 0.12), (0.50, 0.065), C_DARK)
    ax.text(0.50, 0.015, "Deploy fixed $\\hat{m}$ on Held-Out Test (or Target Val under shift)", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=C_DARK)

    fig.suptitle("Unified Framework: Joint Minimax Limits, Prompt Heterogeneity, and Holdout Risk Control", fontsize=10.8, fontweight="bold", y=0.98)
    _save_both(fig, "fig1_framework_overview")


def plot_fig2_joint_mn_and_protocols(v9: dict[str, Any], v10: dict[str, Any], s2: dict[str, Any]) -> None:
    """绘制 Figure 2：三种分桶协议对照 + 含 Wilson-LCB 的同 K 公平消融 + 无压字 2D (M, N) 前沿。"""
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.95), gridspec_kw={"width_ratios": [1.02, 1.08, 1.08]})
    fig.subplots_adjust(wspace=0.28, left=0.05, right=0.98, top=0.84, bottom=0.25)

    # Panel (a): Quantile Protocols on AceMath RM-B (v9)
    ax = axes[0]
    _style_ax(ax)
    prot = v9["quantile_protocols_real_comparison"]["acemath_rm_b_protocols"]
    rows = prot["rows"]
    m_vals = np.array([r["m_total"] for r in rows], dtype=float)
    raw_acc = float(prot["standard_bon_acc"])

    series = [
        ("sample_split_kl_lcb", "Sample-Split KL-LCB (Zero Extra Data)", C_BLUE, "o", "-"),
        ("in_sample_kl_lcb", "In-Sample KL-LCB (Reuse M)", C_SKY, "s", "--"),
        ("semi_supervised_kl_lcb", "Semi-Supervised KL-LCB (+Unlabeled)", C_GREEN, "^", "-."),
        ("sample_split_fixed60_mle", "Sample-Split Fix-60 MLE", C_VERMILION, "D", ":"),
    ]
    for prefix, lbl, col, mk, ls in series:
        y = np.array([r[f"{prefix}_acc"] for r in rows], dtype=float)
        se = np.array([r[f"{prefix}_se"] for r in rows], dtype=float)
        ax.plot(m_vals, y, color=col, marker=mk, linestyle=ls, linewidth=1.7, markersize=4.0, label=lbl, zorder=3)
        ax.fill_between(m_vals, y - se, y + se, color=col, alpha=0.15, zorder=2)

    ax.axhline(raw_acc, color=C_DARK, linestyle="--", linewidth=1.1, label=f"Raw-BoN ($N=64$: {raw_acc:.4f})", zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("Offline Label Budget $M$ (log scale)")
    ax.set_ylabel("Holdout Pass@1 Accuracy")
    ax.set_ylim(0.630, 0.706)
    ax.set_title("(a) Quantile Protocols (AceMath RM-B)")
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.94, edgecolor="#D1D5DB", fontsize=6.7)

    # Panel (b): Same-K Fair Ablation under 6% Tail Contamination (v10) - 补全 Wilson-LCB
    ax = axes[1]
    _style_ax(ax)
    ab_opt = {r["m_total"]: r for r in v10["same_k_fair_calibrator_ablation"]["contaminated_acemath_rm_b_same_k_opt"]["rows"]}
    ab_k30 = {r["m_total"]: r for r in v10["same_k_fair_calibrator_ablation"]["contaminated_acemath_rm_b_same_k30"]["rows"]}
    raw_contam = float(v10["same_k_fair_calibrator_ablation"]["contaminated_acemath_rm_b_same_k30"]["standard_bon_n64"])

    m_pick = [100, 200, 800]
    x = np.arange(len(m_pick))
    w = 0.148
    bars_spec = [
        ("K=3, KL-LCB", [ab_opt[m]["kl_lcb_acc"] for m in m_pick], [ab_opt[m]["kl_lcb_se"] for m in m_pick], "#9CA3AF"),
        ("K=30, MLE", [ab_k30[m]["mle_acc"] for m in m_pick], [ab_k30[m]["mle_se"] for m in m_pick], C_VERMILION),
        ("K=30, Laplace", [ab_k30[m]["laplace_acc"] for m in m_pick], [ab_k30[m]["laplace_se"] for m in m_pick], C_AMBER),
        ("K=30, Wilson-LCB", [ab_k30[m]["wilson_lcb_acc"] for m in m_pick], [ab_k30[m]["wilson_lcb_se"] for m in m_pick], C_GREEN),
        ("K=30, KL-LCB", [ab_k30[m]["kl_lcb_acc"] for m in m_pick], [ab_k30[m]["kl_lcb_se"] for m in m_pick], C_BLUE),
    ]
    offsets = [-2.0 * w, -1.0 * w, 0.0, 1.0 * w, 2.0 * w]
    for off, (lbl, vals, ses, col) in zip(offsets, bars_spec):
        ax.bar(x + off, vals, width=w, yerr=ses, capsize=1.8, color=col, label=lbl, zorder=3, error_kw={"lw": 0.85, "ecolor": "#111827"})

    ax.axhline(raw_contam, color=C_DARK, linestyle="--", linewidth=1.2, label=f"Contam Raw ({raw_contam:.4f})", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$M={m}$" for m in m_pick])
    ax.set_ylim(0.24, 0.745)
    ax.set_ylabel("Pass@1 Accuracy (Sample-Split)")
    ax.set_title("(b) Same-$K$ Fair Ablation (6% Tail Contam)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True, facecolor="white", framealpha=0.95, edgecolor="#D1D5DB", fontsize=6.6)

    # Panel (c): 2D (M, N) Advantage Heatmap with N_balance(M) (背景同色描边避免虚线压字)
    ax = axes[2]
    ace = s2["s4_acemath"]["joint_mn_sota_rm_b"]
    n_vals = np.asarray(ace["n_values"], dtype=float)
    m_grid = np.asarray(ace["m_grid"], dtype=int)
    diff_pp = (np.asarray(ace["grid_opt_1_5_kl"], dtype=float) - np.asarray(ace["grid_fixed60_mle"], dtype=float)) * 100.0
    n_balance = np.asarray(ace["n_match_theory_by_m"], dtype=float)

    vmax = max(1.0, float(np.max(np.abs(diff_pp))))
    im = ax.imshow(diff_pp, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(n_vals.size))
    ax.set_xticklabels([str(int(v)) for v in n_vals], fontsize=7.8)
    ax.set_yticks(np.arange(m_grid.size))
    ax.set_yticklabels([f"{m:,}" if m >= 1000 else str(m) for m in m_grid], fontsize=7.6)

    x_bal = np.interp(np.log2(np.clip(n_balance, n_vals[0], n_vals[-1])), np.log2(n_vals), np.arange(n_vals.size))
    ax.plot(x_bal, np.arange(m_grid.size), color="#111827", linestyle="--", linewidth=1.9, zorder=3, label=r"Thm. 1 Balance Line $N_{\mathrm{balance}}(M)$")
    ax.plot(x_bal, np.arange(m_grid.size), color="#F59E0B", linestyle="--", linewidth=1.0, zorder=3.5)

    for i in range(m_grid.size):
        for j in range(n_vals.size):
            val = diff_pp[i, j]
            txt_col = "white" if abs(val) > 0.55 * vmax else "#111827"
            cell_bg = im.cmap(im.norm(val))
            ax.text(
                j, i, f"{val:+.1f}", ha="center", va="center", color=txt_col, fontsize=6.7, fontweight="bold", zorder=6,
                path_effects=[pe.withStroke(linewidth=3.4, foreground=cell_bg, alpha=0.98)],
            )

    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.95)
    cbar.set_label(r"Gain: Opt-$K^*$+KL-LCB vs. Fix-60 MLE (pp)", fontsize=7.6)
    cbar.ax.tick_params(labelsize=7.5)
    ax.set_xlabel("Online Compute Budget $N$")
    ax.set_ylabel("Offline Labels $M$")
    ax.set_title(r"(c) 2D $(M, N)$ Gain & $N_{\mathrm{balance}}(M)$ Boundary")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=True, facecolor="white", framealpha=0.95, edgecolor="#D1D5DB", fontsize=7.0)

    fig.suptitle("Offline Calibration Protocols, Fine-Bin LCB Regularization, and Joint $(M, N)$ Scaling", fontsize=10.8, fontweight="bold", y=0.96)
    _save_both(fig, "fig2_joint_mn_and_protocols")


def plot_fig3_heterogeneity_and_shift(v9: dict[str, Any], v10: dict[str, Any], tmlr: dict[str, Any]) -> None:
    """绘制 Figure 3：中性化门控对比轴标 + 补齐 -0.32 标签 + 完整 7-Regime 真 i.i.d. 分布漂移矩阵。"""
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.00), gridspec_kw={"width_ratios": [1.05, 0.90, 1.32]})
    fig.subplots_adjust(wspace=0.28, left=0.08, right=0.985, top=0.84, bottom=0.26)

    # Panel (a): Disjoint Single Split Comparisons
    ax = axes[0]
    _style_ax(ax)
    p1 = v9["disjoint_single_split_primary"]
    cm = v10["natural_nonmonotone_collapse_audit"]["college_math_rm_b_single_split"]

    rows_a = [
        ("S3 3-Domain\n(N=128)", p1["s3_3domain"]["algorithms"]["Hybrid-Fixed-KL-LCB"]["accuracy"][-1] - p1["s3_3domain"]["algorithms"]["Pooled-KL-LCB"]["accuracy"][-1], p1["s3_3domain"]["paired_test_vs_pooled"]["Hybrid-Fixed-KL-LCB"]["se_diff"][-1], p1["s3_3domain"]["paired_test_vs_pooled"]["Hybrid-Fixed-KL-LCB"]["p_value_ttest"][-1],
         p1["s3_3domain"]["algorithms"]["Hybrid-Fixed-KL-LCB"]["accuracy"][-1] - p1["s3_3domain"]["algorithms"]["Within-Prompt-Z-KL-LCB"]["accuracy"][-1]),
        ("RB2 Skywork\n(N=4)", p1["rb2_skywork"]["algorithms"]["Hybrid-Fixed-KL-LCB"]["accuracy"][-1] - p1["rb2_skywork"]["algorithms"]["Pooled-KL-LCB"]["accuracy"][-1], p1["rb2_skywork"]["paired_test_vs_pooled"]["Hybrid-Fixed-KL-LCB"]["se_diff"][-1], p1["rb2_skywork"]["paired_test_vs_pooled"]["Hybrid-Fixed-KL-LCB"]["p_value_ttest"][-1],
         p1["rb2_skywork"]["algorithms"]["Hybrid-Fixed-KL-LCB"]["accuracy"][-1] - p1["rb2_skywork"]["algorithms"]["Within-Prompt-Z-KL-LCB"]["accuracy"][-1]),
        ("AceMath RM-B\n(N=64)", p1["acemath_rm_b"]["algorithms"]["Hybrid-Fixed-KL-LCB"]["accuracy"][-1] - p1["acemath_rm_b"]["algorithms"]["Pooled-KL-LCB"]["accuracy"][-1], p1["acemath_rm_b"]["paired_test_vs_pooled"]["Hybrid-Fixed-KL-LCB"]["se_diff"][-1], p1["acemath_rm_b"]["paired_test_vs_pooled"]["Hybrid-Fixed-KL-LCB"]["p_value_ttest"][-1],
         p1["acemath_rm_b"]["algorithms"]["Hybrid-Fixed-KL-LCB"]["accuracy"][-1] - p1["acemath_rm_b"]["algorithms"]["Within-Prompt-Z-KL-LCB"]["accuracy"][-1]),
        ("AceMath RM-A\n(N=64)", p1["acemath_rm_a"]["algorithms"]["Hybrid-Fixed-KL-LCB"]["accuracy"][-1] - p1["acemath_rm_a"]["algorithms"]["Pooled-KL-LCB"]["accuracy"][-1], p1["acemath_rm_a"]["paired_test_vs_pooled"]["Hybrid-Fixed-KL-LCB"]["se_diff"][-1], p1["acemath_rm_a"]["paired_test_vs_pooled"]["Hybrid-Fixed-KL-LCB"]["p_value_ttest"][-1],
         p1["acemath_rm_a"]["algorithms"]["Hybrid-Fixed-KL-LCB"]["accuracy"][-1] - p1["acemath_rm_a"]["algorithms"]["Within-Prompt-Z-KL-LCB"]["accuracy"][-1]),
        ("OlympiadBench\n(N=128)", p1["olympiadbench_uprm70b"]["algorithms"]["Hybrid-Fixed-KL-LCB"]["accuracy"][-1] - p1["olympiadbench_uprm70b"]["algorithms"]["Pooled-KL-LCB"]["accuracy"][-1], p1["olympiadbench_uprm70b"]["paired_test_vs_pooled"]["Hybrid-Fixed-KL-LCB"]["se_diff"][-1], p1["olympiadbench_uprm70b"]["paired_test_vs_pooled"]["Hybrid-Fixed-KL-LCB"]["p_value_ttest"][-1],
         p1["olympiadbench_uprm70b"]["algorithms"]["Hybrid-Fixed-KL-LCB"]["accuracy"][-1] - p1["olympiadbench_uprm70b"]["algorithms"]["Within-Prompt-Z-KL-LCB"]["accuracy"][-1]),
        ("college_math\n(gate=Pooled)", cm["algorithms"]["Hybrid-Fixed-KL-LCB"]["accuracy"][-1] - cm["algorithms"]["Standard-BoN"]["accuracy"][-1], cm["paired_test_vs_standard_bon"]["Hybrid-Fixed-KL-LCB"]["se_diff"][-1], cm["paired_test_vs_standard_bon"]["Hybrid-Fixed-KL-LCB"]["p_value_ttest"][-1],
         cm["algorithms"]["Hybrid-Fixed-KL-LCB"]["accuracy"][-1] - cm["algorithms"]["Within-Prompt-Z-KL-LCB"]["accuracy"][-1]),
    ]
    rows_a = rows_a[::-1]
    y_pos = np.arange(len(rows_a))
    h = 0.34
    d_pool = np.array([r[1] * 100.0 for r in rows_a])
    se_pool = np.array([r[2] * 100.0 for r in rows_a])
    d_z = np.array([r[4] * 100.0 for r in rows_a])
    pvals = [r[3] for r in rows_a]

    ax.barh(y_pos + h / 2.0, d_pool, height=h, xerr=se_pool, capsize=2.0, color=C_BLUE, label="vs. Pooled (bottom: vs. Raw)", zorder=3, error_kw={"lw": 0.9})
    ax.barh(y_pos - h / 2.0, d_z, height=h, color=C_GREEN, label="vs. Within-Prompt-Z", zorder=3)
    ax.axvline(0.0, color=C_DARK, linewidth=1.0, zorder=2)

    for idx, (dp, pv, dz) in enumerate(zip(d_pool, pvals, d_z)):
        star = "**" if pv < 0.01 else ("*" if pv < 0.05 else "")
        if abs(dp) > 0.25 or star:
            xtxt = dp + (se_pool[idx] + 0.18 if dp >= 0 else -se_pool[idx] - 0.18)
            ax.text(xtxt, idx + h / 2.0, f"{dp:+.2f}{star}", va="center", ha="left" if dp >= 0 else "right", fontsize=6.7, color=C_BLUE, fontweight="bold")
        if abs(dz) > 0.20:
            xtxt2 = dz + (0.18 if dz >= 0 else -0.18)
            ax.text(xtxt2, idx - h / 2.0, f"{dz:+.2f}", va="center", ha="left" if dz >= 0 else "right", fontsize=6.7, color=C_GREEN, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0] for r in rows_a], fontsize=7.4)
    ax.set_xlim(-3.5, 11.8)
    ax.set_xlabel("Gate Selection (= Raw-BoN) vs. Baselines (pp)", fontsize=8.3)
    ax.set_title("(a) Disjoint 50/20/30 Single-Split Audit")
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.94, edgecolor="#D1D5DB", fontsize=6.8)

    # Panel (b): Exploratory Case (college_math RM-B)
    ax = axes[1]
    _style_ax(ax)
    nat_inst = [x for x in v10["natural_nonmonotone_collapse_audit"]["acemath_natural_overoptimization_instances"] if x["dataset"] == "AceMath/college_math/RM_B"][0]
    n_grid = np.array(cm["n_values"], dtype=float)
    exact_curve = np.array(nat_inst["exact_curve"], dtype=float)
    std_split = np.array(cm["algorithms"]["Standard-BoN"]["accuracy"], dtype=float)
    hyb_split = np.array(cm["algorithms"]["Hybrid-Fixed-KL-LCB"]["accuracy"], dtype=float)
    winz_split = np.array(cm["algorithms"]["Within-Prompt-Z-KL-LCB"]["accuracy"], dtype=float)

    ax.plot(n_grid, exact_curve, color=C_DARK, marker="o", linestyle="-", linewidth=1.8, markersize=4.0, label="Full Raw-BoN ($P=2{,}818$)", zorder=4)
    ax.plot(n_grid, std_split, color=C_VERMILION, marker="s", linestyle="--", linewidth=1.5, markersize=3.6, label="Split Raw-BoN ($P_{\\mathrm{test}}=845$)", zorder=3)
    ax.plot(n_grid, hyb_split, color=C_BLUE, marker="^", linestyle="-", linewidth=1.6, markersize=3.8, label="Split Hybrid-Fixed", zorder=3)
    ax.plot(n_grid, winz_split, color=C_GREEN, marker="D", linestyle=":", linewidth=1.5, markersize=3.4, label="Split Within-Z", zorder=3)

    ax.annotate(
        "Exploratory peak $N=16$: 0.4640\n$\\to N=64$: 0.4579 ($-0.60$ pp)\n$p_{\\mathrm{raw}}=0.0208$, Bonf.(14)$=0.291$",
        xy=(64, exact_curve[-1]),
        xytext=(3.2, 0.398),
        arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.0),
        fontsize=6.4,
        fontweight="bold",
        color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.23", fc="#F3F4F6", ec=C_GRAY, lw=0.7),
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(n_grid)
    ax.set_xticklabels([str(int(v)) for v in n_grid])
    ax.set_xlabel("Sampling Budget $N$ (log$_2$ scale)")
    ax.set_ylabel("Pass@1 Accuracy")
    ax.set_ylim(0.36, 0.475)
    ax.set_title("(b) Exploratory Case (college_math RM-B)")
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.94, edgecolor="#D1D5DB", fontsize=6.5)

    # Panel (c): Complete 7-Regime Distribution-Shift Matrix (与修复后的 tmlr_diagnostics.json Table 5 完全一致)
    ax = axes[2]
    _style_ax(ax)
    smat = tmlr["module_c_counterfactual_pi0_ablation"]["distribution_shift_matrix"]
    reg_entries = [
        (smat["1_clean"], "1\nClean"),
        (smat["2a_ood_matched"], "2a\nOOD\nMatch"),
        (smat["3a_ood_unseen_shift"], "3a\nOOD\nUnseen"),
        (smat["4a_ood_target_val"], "4a\nOOD\nTgtVal"),
        (smat["2b_in_support_matched"], "2b\nInSup\nMatch"),
        (smat["3b_in_support_unseen_shift"], "3b\nInSup\nUnseen"),
        (smat["4b_in_support_target_val"], "4b\nInSup\nTgtVal"),
    ]
    xr = np.arange(len(reg_entries))
    bw = 0.145
    meths = [
        ("raw_bon_n64", "Raw-BoN", "#9CA3AF"),
        ("pooled_kl_lcb_n64", "Pooled-LCB", C_SKY),
        ("within_prompt_z_n64", "Within-Z", C_GREEN),
        ("hybrid_fixed_n64", "Hybrid-Fixed", C_BLUE),
        ("support_clipped_n64", "Support-Clipped", C_VERMILION),
    ]
    offs = [-2.0 * bw, -1.0 * bw, 0.0, 1.0 * bw, 2.0 * bw]
    for off, (mkey, mlbl, col) in zip(offs, meths):
        vals = [float(entry[mkey]) for entry, _ in reg_entries]
        ax.bar(xr + off, vals, width=bw, color=col, label=mlbl, zorder=3)

    ax.axvline(3.5, color=C_GRAY, linestyle=":", linewidth=1.0, zorder=2)
    ax.set_xticks(xr)
    ax.set_xticklabels([rlbl for _, rlbl in reg_entries], fontsize=6.8)
    ax.set_ylim(0.18, 0.77)
    ax.set_ylabel("Holdout Pass@1 Accuracy ($N=64$)")
    ax.set_title("(c) Complete 7-Regime Shift Matrix (AceMath RM-B)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=3, frameon=True, facecolor="white", framealpha=0.95, edgecolor="#D1D5DB", fontsize=6.6)

    fig.suptitle("Cross-Prompt Heterogeneity, Exploratory Case, and Full Distribution-Shift Matrix", fontsize=10.8, fontweight="bold", y=0.96)
    _save_both(fig, "fig3_heterogeneity_and_shift")


def plot_fig4_multi_rm_dependence(s2: dict[str, Any], v6: dict[str, Any]) -> None:
    """绘制 Figure 4：跨家族相关性结构、RB2 N=4 5-Fold CV 谱路径对比与 OlympiadBench 5-Fold CV 曲线。"""
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.70), gridspec_kw={"width_ratios": [1.05, 0.94, 1.12]})
    fig.subplots_adjust(wspace=0.28, left=0.05, right=0.98, top=0.84, bottom=0.19)

    # Panel (a): Cross-family vs Homogeneous Conditional Dependence
    ax = axes[0]
    _style_ax(ax)
    s1a = s2["s1a_rb2"]
    s1b = s2["s1b_rb1"]
    s2o = s2["s2_olympiadbench"]
    groups = [
        ("RB2\nHomo-4", s1a["homo4_all"]),
        ("RB2\nHetero-6", s1a["hetero6_all"]),
        ("RB2\nTies", s1a["hetero6_ties"]),
        ("RB1\nCode-6", s1b["code_6with_deberta"]),
        ("Oly\n70B-4V", s2o["same_prm_70b_4views"]),
        ("Oly\n70B+32B", s2o["pair_prm_70b_vs_32b"]),
    ]
    xg = np.arange(len(groups))
    w = 0.25
    rho1 = [float(g[1]["symmetric_diag"]["mean_rho1"]) for g in groups]
    rho0 = [float(g[1]["symmetric_diag"]["mean_rho0"]) for g in groups]
    eta0 = [float(g[1]["copula_eta0"]) for g in groups]

    ax.bar(xg - w, rho1, width=w, color=C_BLUE, label=r"Correct Corr. $\bar{\rho}_1$", zorder=3)
    ax.bar(xg, rho0, width=w, color=C_VERMILION, label=r"Wrong Corr. $\bar{\rho}_0$", zorder=3)
    ax.bar(xg + w, eta0, width=w, color=C_GREEN, label=r"Copula Tail $\eta_0$", zorder=3)
    ax.set_xticks(xg)
    ax.set_xticklabels([g[0] for g in groups], fontsize=7.4)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Correlation / Tail Coefficient")
    ax.set_title("(a) Cross-Family vs. Homogeneous Tails")
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#D1D5DB", fontsize=7.0)

    # Panel (b): RB2 Hetero-6 N=4 5-Fold CV Spectral Path Comparison
    ax = axes[1]
    _style_ax(ax)
    accs = v6["real_data"]["rb2_hetero6_n4_5fold"]["accuracies"]
    spec_items = [
        ("Best1\n(Skywork)", float(accs["Best1(Skywork)"]), C_DARK),
        ("Fisher\n($\\tau=-1$)", float(accs["Fisher(tau=-1)"]), C_BLUE),
        ("Bayes\nQDA", float(accs["QDA"]), C_GREEN),
        ("Safe-Fisher\n($\\kappa \\leq 1$)", float(accs["Safe-Fisher(kappa<=1)"]), C_AMBER),
        ("GenEig\n(No Mean)", float(accs["GenEig"]), C_VERMILION),
    ]
    xs = np.arange(len(spec_items))
    vals = [s[1] for s in spec_items]
    cols = [s[2] for s in spec_items]
    bars = ax.bar(xs, vals, width=0.58, color=cols, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2.0, v + 0.015, f"{v:.4f}", ha="center", va="bottom", fontsize=7.1, fontweight="bold", color=C_DARK)

    ax.set_xticks(xs)
    ax.set_xticklabels([s[0] for s in spec_items], fontsize=7.2)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("RB2 Hetero-6 Accuracy ($P=1{,}723, N=4$)")
    ax.set_title("(b) RB2 Spectral Trade-off ($N=4$, 5-Fold CV)")

    # Panel (c): OlympiadBench Qwen2.5-7B with Dual PRMs (5-Fold CV)
    ax = axes[2]
    _style_ax(ax)
    d5 = s2o["discrete_5fold_70b"]
    cross4 = s2o["cross_prm_4views_70b_32b"]
    pair = s2o["pair_prm_70b_vs_32b"]
    n_vals = np.asarray(d5["n_values"], dtype=float)

    curves = [
        (d5["algorithms"]["Calibrated-Weighted-BoN (Wilson-LCB)"], "Calib.-Weighted (70B)", C_BLUE, "o", "-"),
        (cross4["cv_results"]["Logit-Stack"], "Logit-Stack (70B+32B)", C_GREEN, "s", "-"),
        (d5["algorithms"]["Majority-Voting"], "Majority-Voting", C_SKY, "^", ":"),
        (cross4["cv_results"]["Min-Ensemble"], "Min-Ensemble (Cross-4)", C_PURPLE, "P", "--"),
        (pair["cv_results"]["uPRM-70B-Last"], "Single uPRM-70B", C_VERMILION, "D", "-."),
        (pair["cv_results"]["uPRM-32B-Last"], "Single uPRM-32B", C_GRAY, "v", "-."),
    ]
    for src, lbl, col, mk, ls in curves:
        acc = np.asarray(src["accuracy"], dtype=float)
        se = np.asarray(src["se"], dtype=float)
        ax.plot(n_vals, acc, color=col, marker=mk, linestyle=ls, linewidth=1.6, markersize=3.8, label=lbl, zorder=3)
        ax.fill_between(n_vals, acc - se, acc + se, color=col, alpha=0.10, zorder=2)

    ax.set_xscale("log", base=2)
    ax.set_xticks(n_vals)
    ax.set_xticklabels([str(int(v)) for v in n_vals])
    ax.set_xlabel("Sampling Budget $N$ (log$_2$ scale)")
    ax.set_ylabel("OlympiadBench Pass@1 Accuracy")
    ax.set_ylim(0.32, 0.505)
    ax.set_title("(c) OlympiadBench 5-Fold CV (uPRM-70B/32B)")
    ax.legend(loc="lower right", ncol=2, frameon=True, facecolor="white", framealpha=0.92, edgecolor="#D1D5DB", fontsize=6.6)

    fig.suptitle("Multi-Verifier Dependence Structure, Spectral Trade-offs, and PRM Scaling", fontsize=10.8, fontweight="bold", y=0.96)
    _save_both(fig, "fig4_multi_rm_dependence")


def plot_fig5_tmlr_z_diagnostics(tmlr: dict[str, Any]) -> None:
    """绘制 Figure 5（TMLR 专属 3 联诊断图）：一般 (pi_0,pi_1) 与左尾陷阱 + 5 基准 Z-bin 走势 + pi_0=0 消融与 8-seed 稳健性。"""
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.10), gridspec_kw={"width_ratios": [1.05, 1.05, 1.12]})
    fig.subplots_adjust(wspace=0.27, left=0.055, right=0.985, top=0.84, bottom=0.22)

    # Panel (a): 一般 (pi_0, pi_1) Z*-空间后验曲线与 Case 4 左尾反超陷阱对比（内嵌 N-缩放对比）
    ax = axes[0]
    _style_ax(ax)
    mod_a = tmlr["module_a_general_pi1_theory"]
    c1 = mod_a["case1_compact_peak_loss"]
    c2 = mod_a["case2_high_pi1_practical_boundary"]
    c3 = mod_a["case3_filtered_pi0_zero_no_loss"]
    c4 = mod_a["case4_left_tail_overtake_trap"]

    ax.plot(c1["curve"]["z_grid"], c1["curve"]["q_pool_z"], color=C_BLUE, linewidth=1.8,
            label=r"Case 1 ($\pi_0=0.45, \pi_1=0.10$)", zorder=3)
    ax.plot(c2["adaptive_global_diag"]["z_grid_practical"], c2["adaptive_global_diag"]["q_pool_z_practical"],
            color=C_AMBER, linewidth=1.8, linestyle="--",
            label=r"Case 2 ($\pi_0=0.05, \pi_1=0.50$; peak $z^*=18.8$)", zorder=3)
    ax.plot(c3["curve"]["z_grid"], c3["curve"]["q_pool_z"], color=C_GREEN, linewidth=1.7, linestyle="-.",
            label=r"Case 3 ($\pi_0=0, \pi_1=0.50$; $L_{\mathrm{tail}}=1$)", zorder=3)
    ax.plot(c4["curve"]["z_grid"], c4["curve"]["q_pool_z"], color=C_VERMILION, linewidth=1.8,
            label=r"Case 4 ($\pi_0=0, \pi_1=0.10, \Delta=15$; two-tail $\to 1$)", zorder=3)

    ax.axhline(c1["curve"]["z_tail_posterior_limit"], color=C_BLUE, linestyle=":", linewidth=1.0, alpha=0.8)
    ax.scatter([c1["curve"]["z_peak"]], [c1["curve"]["q_z_peak"]], color=C_BLUE, s=26, zorder=5)
    ax.annotate(
        "$z^*=2.17$\n$(L_{\\mathrm{tail}}=0.182)$",
        xy=(c1["curve"]["z_peak"], c1["curve"]["q_z_peak"]),
        xytext=(3.1, 0.57),
        arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=0.8),
        fontsize=6.2, fontweight="bold", color=C_BLUE, va="center",
    )

    ax.set_xlim(-4.0, 8.0)
    ax.set_ylim(0.0, 1.06)
    ax.set_xlabel("Standardized Score $z$")
    ax.set_ylabel(r"Pooled Posterior $q_{\mathrm{pool}, Z^*}(z)$")
    ax.set_title("(a) General $(\\pi_0, \\pi_1)$ Regimes & Left-Tail Trap")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, frameon=True, facecolor="white", framealpha=0.95, edgecolor="#D1D5DB", fontsize=6.5)

    # Panel (a) 内嵌小图：Case 4 Unconstrained vs Right-Half (Z >= 0) N-缩放对比
    ax_in = ax.inset_axes([0.56, 0.08, 0.41, 0.30])
    ax_in.grid(True, linestyle=":", linewidth=0.4, alpha=0.35)
    nv4 = np.asarray(c4["unconstrained_two_sided"]["n_values"], dtype=float)
    raw4 = np.asarray(c4["unconstrained_two_sided"]["raw_bon_acc"], dtype=float)
    unc4 = np.asarray(c4["unconstrained_two_sided"]["sample_z_oracle_acc"], dtype=float)
    rgt4 = np.asarray(c4["right_half_constrained_z_ge_0"]["sample_z_oracle_acc"], dtype=float)
    ax_in.plot(nv4, raw4, color=C_DARK, linewidth=1.2, linestyle="-", label="Raw")
    ax_in.plot(nv4, rgt4, color=C_GREEN, linewidth=1.2, linestyle="--", marker="o", markersize=2.5, label="$Z\\geq 0$")
    ax_in.plot(nv4, unc4, color=C_VERMILION, linewidth=1.2, marker="s", markersize=2.5, label="Uncon.")
    ax_in.set_xscale("log", base=4)
    ax_in.set_xticks([16, 256, 4096])
    ax_in.set_xticklabels(["16", "256", "4k"], fontsize=6.0)
    ax_in.tick_params(axis="y", labelsize=6.0)
    ax_in.set_ylim(0.48, 1.04)
    ax_in.set_title("Case 4 Accuracy", fontsize=6.6, pad=2)
    ax_in.legend(loc="lower right", fontsize=5.6, frameon=True, framealpha=0.9, borderpad=0.2, handlelength=1.2)

    # Panel (b): 5 Benchmark K=20 Empirical Z-Bin Trajectories (Module B)
    ax = axes[1]
    _style_ax(ax)
    bench = tmlr["module_b_real_benchmarks_boundary_audit"]["benchmarks"]
    b_specs = [
        ("rb2_skywork", "RB2 Skywork ($\\hat{\\pi}_0=0$)", C_DARK, "o", "-"),
        ("acemath_rm_a", "AceMath RM-A ($\\hat{L}=0.145$)", C_SKY, "s", "--"),
        ("acemath_rm_b", "AceMath RM-B ($\\hat{L}=0.145$)", C_BLUE, "^", "-"),
        ("s3_3domain", "Strategic-TTC ($\\hat{L}=0.490$)", C_AMBER, "D", "-."),
        ("olympiadbench_uprm70b", "OlympiadBench ($\\hat{L}=0.293$)", C_VERMILION, "v", "-"),
    ]
    bins_x = np.arange(1, 21)
    for b_key, lbl, col, mk, ls in b_specs:
        q_bins = np.asarray(bench[b_key]["binned_k20"]["q_z_bins"], dtype=float)
        ax.plot(bins_x, q_bins, color=col, marker=mk, linestyle=ls, linewidth=1.6, markersize=3.2, label=lbl, zorder=3)

    ax.axhline(float(bench["s3_3domain"]["tail_limit_pred_pi1_over_pi0_plus_pi1"]), color=C_AMBER, linestyle=":", linewidth=1.2, alpha=0.85)
    ax.text(1.5, 0.505, r"S3 $\hat{L}_{\mathrm{tail}}=0.490$", fontsize=6.6, color="#92400E", fontweight="bold")
    ax.annotate(
        "Oly. peak bin 8 (0.623)\n$\\to$ bin 20 (0.258)",
        xy=(20, float(bench["olympiadbench_uprm70b"]["binned_k20"]["q_z_right_tail_hat"])),
        xytext=(8.6, 0.08),
        arrowprops=dict(arrowstyle="->", color=C_VERMILION, lw=0.9),
        fontsize=6.5, fontweight="bold", color=C_VERMILION,
    )

    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Empirical Sample-$Z$ Equal-Frequency Bin ($K=20$)")
    ax.set_ylabel(r"Empirical Bin Accuracy $\hat{q}_Z(k)$")
    ax.set_title("(b) Empirical $Z$-Bin Curves Across 5 Settings")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=True, facecolor="white", framealpha=0.95, edgecolor="#D1D5DB", fontsize=6.3)

    # Panel (c): Module C Counterfactual pi_0=0 LCB Curves & 8-Seed Stability
    ax = axes[2]
    _style_ax(ax)
    mod_c = tmlr["module_c_counterfactual_pi0_ablation"]
    regs = mod_c["regimes"]
    c_specs = [
        ("full_train", "Full Train ($\\hat{\\pi}_0=0.144$)", C_DARK, "-"),
        ("filtered_pi0_zero", "Filtered $\\hat{\\pi}_0=0$", C_BLUE, "-"),
        ("filtered_interior_only", "Interior Only ($\\hat{\\pi}_0=\\hat{\\pi}_1=0$)", C_GREEN, "--"),
    ]
    for r_key, lbl, col, ls in c_specs:
        lcb_vals = np.asarray(regs[r_key]["alg1_within_z_diag"]["bin_scores_lcb"], dtype=float)
        ax.plot(bins_x, lcb_vals, color=col, linestyle=ls, linewidth=1.75, marker="o", markersize=2.9, label=lbl, zorder=3)

    b16_20 = regs["filtered_pi0_zero"]["alg1_within_z_diag"]["bin16_to_bin20"]
    ax.annotate(
        f"$+12\\,\\mathrm{{pp}}$ shift; downturn persists\n($\\bar{{p}}_{{16}}={b16_20['bin16_mean_p']:.3f}\\to\\bar{{p}}_{{20}}={b16_20['bin20_mean_p']:.3f}$)",
        xy=(19.8, b16_20["bin20_lcb"]),
        xytext=(5.5, 0.86),
        arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=0.9),
        fontsize=6.5, fontweight="bold", color=C_BLUE,
        bbox=dict(boxstyle="round,pad=0.22", fc="#EFF6FF", ec=C_BLUE, lw=0.6),
    )

    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0.08, 0.98)
    ax.set_xlabel("Discrete Alg. 1 $Z$-Bin Index ($K=20$)")
    ax.set_ylabel("Training Bin KL-LCB Score")
    ax.set_title("(c) Counterfactual $\\hat{\\pi}_0=0$ LCB & 8-Seed Check")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, frameon=True, facecolor="white", framealpha=0.95, edgecolor="#D1D5DB", fontsize=6.5)

    # Panel (c) 内嵌小图：8-seed (Filtered - Size-Matched) 差异条形图
    ax_c_in = ax.inset_axes([0.50, 0.12, 0.46, 0.34])
    ax_c_in.grid(True, axis="y", linestyle=":", linewidth=0.4, alpha=0.4)
    ms = mod_c["multi_seed_8_summary"]["by_env"]
    env_keys = ["1_clean_test", "3a_ood_unseen_test", "3b_in_support_unseen_test"]
    env_lbls = ["Clean", "3a OOD", "3b InSup"]
    xe = np.arange(3)
    cont_m = [ms[e]["filtered_minus_size_matched"]["continuous_within_z"]["mean"] * 100.0 for e in env_keys]
    cont_sd = [ms[e]["filtered_minus_size_matched"]["continuous_within_z"]["sd"] * 100.0 for e in env_keys]
    alg1_m = [ms[e]["filtered_minus_size_matched"]["alg1_within_z"]["mean"] * 100.0 for e in env_keys]
    alg1_sd = [ms[e]["filtered_minus_size_matched"]["alg1_within_z"]["sd"] * 100.0 for e in env_keys]
    ax_c_in.bar(xe - 0.17, cont_m, width=0.32, yerr=cont_sd, capsize=1.5, color=C_SKY, label="Cont.", error_kw={"lw": 0.7})
    ax_c_in.bar(xe + 0.17, alg1_m, width=0.32, yerr=alg1_sd, capsize=1.5, color=C_BLUE, label="Alg.1", error_kw={"lw": 0.7})
    ax_c_in.axhline(0.0, color=C_DARK, linewidth=0.7)
    ax_c_in.set_xticks(xe)
    ax_c_in.set_xticklabels(env_lbls, fontsize=5.8)
    ax_c_in.tick_params(axis="y", labelsize=5.8)
    ax_c_in.set_ylim(-2.1, 1.4)
    ax_c_in.set_title("8-Seed $\\Delta(\\hat{\\pi}_0{=}0 - \\mathrm{Match})$ (pp)", fontsize=6.0, pad=2)
    ax_c_in.legend(loc="lower right", ncol=2, fontsize=5.4, frameon=True, framealpha=0.9, borderpad=0.2, handlelength=1.0)

    fig.suptitle("TMLR Z-Score Boundary Diagnostics: Theory, Five-Setting Audit, and Counterfactual Ablation", fontsize=10.8, fontweight="bold", y=0.96)
    _save_both(fig, "fig5_tmlr_z_diagnostics")


def main() -> None:
    v9 = json.loads((RESULTS_DIR / "v9_consolidation_verification.json").read_text(encoding="utf-8"))
    v10 = json.loads((RESULTS_DIR / "v10_release_verification.json").read_text(encoding="utf-8"))
    v6 = json.loads((RESULTS_DIR / "v6_theory_verification.json").read_text(encoding="utf-8"))
    s2 = json.loads((RESULTS_DIR / "stage2_real_results.json").read_text(encoding="utf-8"))
    tmlr = json.loads((RESULTS_DIR / "tmlr_diagnostics.json").read_text(encoding="utf-8"))

    plot_fig1_framework_overview()
    plot_fig2_joint_mn_and_protocols(v9, v10, s2)
    plot_fig3_heterogeneity_and_shift(v9, v10, tmlr)
    plot_fig4_multi_rm_dependence(s2, v6)
    plot_fig5_tmlr_z_diagnostics(tmlr)
    print("全部 5 组 TMLR 专属矢量图表（PDF + 300 DPI PNG）已在 paper/tmlr/figures/ 生成完毕。")


if __name__ == "__main__":
    main()
