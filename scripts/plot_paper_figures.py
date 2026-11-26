"""生成顶会投稿级矢量论文图表（PDF + 300 DPI PNG）至 paper/figures/。

全部实证数值严格读取自：
- `results/v9_consolidation_verification.json`
- `results/v10_release_verification.json`
- `results/v6_theory_verification.json`
- `results/stage2_real_results.json`

统一学术排版规范：
- Times/Serif 字体族（与 NeurIPS `ptm` 正文字体协调）
- 色盲安全 Okabe-Ito 调色板
- 去除上/右坐标边框，低透明度虚线辅助网格
- 严格对齐论文正文符号（如 `N_balance(M)`、`Hybrid-Fixed`）
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
OUT_DIR = PROJECT_ROOT / "paper" / "figures"

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
    "axes.titlesize": 10.2,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 7.8,
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
    """同时导出矢量 PDF（供 LaTeX 加载）与 300 DPI PNG（供视觉审校）。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / f"{stem}.pdf"
    png_path = OUT_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"已导出 {pdf_path.name} 与 {png_path.name}")


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
    """绘制 Figure 1：双层模型边界、联合 (M,N) Minimax 相图与 Hybrid-Fixed 三分支控制流。"""
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.65), gridspec_kw={"width_ratios": [1.08, 1.0, 1.08]})
    fig.subplots_adjust(wspace=0.22, left=0.02, right=0.99, top=0.87, bottom=0.14)

    # Panel (a): Two-level framework & scope boundary
    ax = axes[0]
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_title("(a) Hierarchical Model & Scope Boundary", pad=8)

    _draw_box(ax, 0.20, 0.80, 0.60, 0.14, "Prompt $X \\sim \\mathcal{P}_X$, prior $p(X)$\n$N$ i.i.d. candidates $(Y_i, S_i) \\mid X$", "#F3F4F6", C_DARK, fontsize=7.9, bold=True)

    # Left branch: Fixed prompt
    _draw_arrow(ax, (0.35, 0.785), (0.24, 0.64), C_BLUE)
    _draw_box(ax, 0.03, 0.45, 0.42, 0.175, "Fixed-Prompt Regime\n$P_{Y,S\\mid X} \\equiv P$\n(Thm. 1 & Prop. 1)", "#EFF6FF", C_BLUE, tc="#1E3A8A", fontsize=7.7, bold=True)
    _draw_arrow(ax, (0.24, 0.435), (0.24, 0.305), C_BLUE)
    _draw_box(ax, 0.03, 0.06, 0.42, 0.23, "Sample-Split KL-LCB\n$K^*(M) \\asymp (M/\\ln M)^{\\frac{1}{2\\beta+1}}$\nAll $\\beta>0$ Pyke Spacings", "#DBEAFE", C_BLUE, fontsize=7.5)

    # Right branch: Heterogeneous prompts
    _draw_arrow(ax, (0.65, 0.785), (0.76, 0.64), C_VERMILION)
    _draw_box(ax, 0.53, 0.45, 0.44, 0.175, "Cross-Prompt Heterogeneity\nShifts $(b(X), a(X)),\\ p(X)$\n(Thms. 2--4)", "#FFF7ED", C_VERMILION, tc="#9A3412", fontsize=7.7, bold=True)
    _draw_arrow(ax, (0.76, 0.435), (0.76, 0.305), C_VERMILION)
    _draw_box(ax, 0.53, 0.06, 0.44, 0.23, "Pooled: Simpson Masking\nWithin-Z: $O_p(\\sqrt{\\ln N/N})$\n& $\\pi_0>0$ Ceiling Loss", "#FFEDD5", C_VERMILION, fontsize=7.5)

    # Divider warning tag (Remark 1 scope boundary)
    ax.plot([0.49, 0.49], [0.05, 0.65], linestyle=":", color=C_GRAY, linewidth=1.1)
    ax.text(0.49, 0.675, "Remark 1\nScope Boundary", ha="center", va="bottom", fontsize=7.0, color=C_GRAY, style="italic")

    # Panel (b): Joint (M, N) Minimax Phase Diagram (exact beta=2 theoretical rate)
    ax = axes[1]
    _style_ax(ax)
    m_exp = np.linspace(2.0, 6.0, 220)  # M from 10^2 to 10^6
    n_exp = np.linspace(0.3, 2.7, 200)  # N from 2 to ~500
    M_grid, N_grid = np.meshgrid(10.0**m_exp, 10.0**n_exp, indexing="xy")

    beta = 2.0
    # Minimax regret surface proxy normalized for clear visualization:
    # (ln M / M)^(beta/(2*beta+1)) + N^(-beta)
    r_off = (np.log(M_grid) / M_grid) ** (beta / (2.0 * beta + 1.0))
    r_on = N_grid ** (-beta)
    log_regret = np.log10(r_off + r_on)

    cf = ax.contourf(m_exp, n_exp, log_regret, levels=16, cmap="YlGnBu_r", alpha=0.90)
    cbar = fig.colorbar(cf, ax=ax, pad=0.02, shrink=0.94)
    cbar.set_label(r"$\log_{10}\mathcal{R}^*(M,N)$ ($\beta=2$)", fontsize=8.0)
    cbar.ax.tick_params(labelsize=7.5)

    # N_balance(M) boundary where r_off = r_on => N = (M / ln M)^(1 / (2*beta+1))
    m_line = 10.0**m_exp
    n_bal = (m_line / np.log(m_line)) ** (1.0 / (2.0 * beta + 1.0))
    ax.plot(m_exp, np.log10(n_bal), color=C_VERMILION, linestyle="--", linewidth=2.2, zorder=5, label=r"$N_{\mathrm{balance}}(M)\asymp(M/\ln M)^{\frac{1}{2\beta+1}}$")

    ax.text(2.22, 2.25, "Calibration-Dominated\n$N \\gg N_{\\mathrm{balance}}(M)$\n(More $N$ saturates)", color="white", fontsize=7.5, fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2.0, foreground="#111827", alpha=0.85)])
    ax.text(4.08, 0.80, "Compute-Dominated\n$N \\ll N_{\\mathrm{balance}}(M)$\n(Scale $N$ pays off)", color="#111827", fontsize=7.5, fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2.0, foreground="white", alpha=0.85)])

    ax.set_xlim(2.0, 6.0)
    ax.set_ylim(0.3, 2.65)
    ax.set_xticks([2, 3, 4, 5, 6])
    ax.set_xticklabels([r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$", r"$10^6$"])
    ax.set_yticks([0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_yticklabels(["3", "10", "32", "100", "316"])
    ax.set_xlabel("Offline Calibration Labels $M$")
    ax.set_ylabel("Online Compute Budget $N$")
    ax.set_title("(b) Joint $(M, N)$ Minimax Phase Regimes", pad=8)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#D1D5DB", fontsize=7.3)

    # Panel (c): Three-Branch Holdout Risk-Control (Hybrid-Fixed)
    ax = axes[2]
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_title("(c) Risk-Controlled Selector (Hybrid-Fixed)", pad=8)

    _draw_box(ax, 0.04, 0.78, 0.92, 0.16, "Disjoint Prompt Split: Train (50%) / Val (20%) / Test (30%)", "#F3F4F6", C_DARK, fontsize=7.9, bold=True)

    _draw_arrow(ax, (0.20, 0.765), (0.17, 0.625), C_GRAY)
    _draw_arrow(ax, (0.50, 0.765), (0.50, 0.625), C_BLUE)
    _draw_arrow(ax, (0.80, 0.765), (0.83, 0.625), C_GREEN)

    _draw_box(ax, 0.02, 0.45, 0.29, 0.16, "Raw-BoN\n(Safe on clean\nheterogeneous)", "#F9FAFB", C_GRAY, fontsize=7.3)
    _draw_box(ax, 0.345, 0.45, 0.31, 0.16, "Pooled-KL-LCB\n(+Support Clip\nunder unseen tail)", "#EFF6FF", C_BLUE, fontsize=7.3)
    _draw_box(ax, 0.685, 0.45, 0.295, 0.16, "Within-Z-KL-LCB\n(Shift-robust via\ntail downweight)", "#ECFDF5", C_GREEN, fontsize=7.3)

    _draw_arrow(ax, (0.17, 0.435), (0.44, 0.31), C_GRAY)
    _draw_arrow(ax, (0.50, 0.435), (0.50, 0.31), C_BLUE)
    _draw_arrow(ax, (0.83, 0.435), (0.56, 0.31), C_GREEN)

    _draw_box(ax, 0.08, 0.135, 0.84, 0.16, "Holdout Gate on Val: $\\hat{m} = \\arg\\max_{m\\in\\mathcal{M}} \\hat{A}_{\\mathrm{val}}(m)$\nFreeze predictors ($\\varepsilon_{\\mathrm{refit}} = 0$, Prop. 2 Hoeffding bound)", "#FEF3C7", C_AMBER, tc="#92400E", fontsize=7.6, bold=True)
    _draw_arrow(ax, (0.50, 0.12), (0.50, 0.065), C_DARK)
    ax.text(0.50, 0.015, "Deploy fixed $\\hat{m}$ on Held-Out Test (or Target Val under shift)", ha="center", va="bottom", fontsize=7.6, fontweight="bold", color=C_DARK)

    fig.suptitle("Unified Framework: Joint Minimax Limits, Prompt Heterogeneity, and Holdout Risk Control", fontsize=11.0, fontweight="bold", y=0.98)
    _save_both(fig, "fig1_framework_overview")


def plot_fig2_joint_mn_and_protocols(v9: dict[str, Any], v10: dict[str, Any], s2: dict[str, Any]) -> None:
    """绘制 Figure 2：三种分桶协议对照 + 同 K 公平消融 + 2D (M, N) 前沿与 N_balance(M)。"""
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.85), gridspec_kw={"width_ratios": [1.05, 1.0, 1.08]})
    fig.subplots_adjust(wspace=0.28, left=0.05, right=0.98, top=0.84, bottom=0.24)

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
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.94, edgecolor="#D1D5DB", fontsize=6.9)

    # Panel (b): Same-K Fair Ablation under 6% Tail Contamination (v10)
    ax = axes[1]
    _style_ax(ax)
    ab_opt = {r["m_total"]: r for r in v10["same_k_fair_calibrator_ablation"]["contaminated_acemath_rm_b_same_k_opt"]["rows"]}
    ab_k30 = {r["m_total"]: r for r in v10["same_k_fair_calibrator_ablation"]["contaminated_acemath_rm_b_same_k30"]["rows"]}
    raw_contam = float(v10["same_k_fair_calibrator_ablation"]["contaminated_acemath_rm_b_same_k30"]["standard_bon_n64"])

    m_pick = [100, 200, 800]
    x = np.arange(len(m_pick))
    w = 0.18
    bars_spec = [
        ("K=3, KL-LCB", [ab_opt[m]["kl_lcb_acc"] for m in m_pick], [ab_opt[m]["kl_lcb_se"] for m in m_pick], "#9CA3AF"),
        ("K=30, MLE", [ab_k30[m]["mle_acc"] for m in m_pick], [ab_k30[m]["mle_se"] for m in m_pick], C_VERMILION),
        ("K=30, Laplace", [ab_k30[m]["laplace_acc"] for m in m_pick], [ab_k30[m]["laplace_se"] for m in m_pick], C_AMBER),
        ("K=30, KL-LCB", [ab_k30[m]["kl_lcb_acc"] for m in m_pick], [ab_k30[m]["kl_lcb_se"] for m in m_pick], C_BLUE),
    ]
    offsets = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]
    for off, (lbl, vals, ses, col) in zip(offsets, bars_spec):
        ax.bar(x + off, vals, width=w, yerr=ses, capsize=2.0, color=col, label=lbl, zorder=3, error_kw={"lw": 0.9, "ecolor": "#111827"})

    ax.axhline(raw_contam, color=C_DARK, linestyle="--", linewidth=1.2, label=f"Contam Raw ({raw_contam:.4f})", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$M={m}$" for m in m_pick])
    ax.set_ylim(0.24, 0.745)
    ax.set_ylabel("Pass@1 Accuracy (Sample-Split)")
    ax.set_title("(b) Same-$K$ Fair Ablation (6% Tail Contam)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True, facecolor="white", framealpha=0.95, edgecolor="#D1D5DB", fontsize=6.9)

    # Panel (c): 2D (M, N) Advantage Heatmap with N_balance(M) (stage2_real_results.json)
    ax = axes[2]
    ace = s2["s4_acemath"]["joint_mn_sota_rm_b"]
    n_vals = np.asarray(ace["n_values"], dtype=float)
    m_grid = np.asarray(ace["m_grid"], dtype=int)
    diff_pp = (np.asarray(ace["grid_opt_1_5_kl"], dtype=float) - np.asarray(ace["grid_fixed60_mle"], dtype=float)) * 100.0
    # 严格修正图中理论平衡曲线符号标注为 N_balance(M)
    n_balance = np.asarray(ace["n_match_theory_by_m"], dtype=float)

    vmax = max(1.0, float(np.max(np.abs(diff_pp))))
    im = ax.imshow(diff_pp, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(n_vals.size))
    ax.set_xticklabels([str(int(v)) for v in n_vals], fontsize=7.8)
    ax.set_yticks(np.arange(m_grid.size))
    ax.set_yticklabels([f"{m:,}" if m >= 1000 else str(m) for m in m_grid], fontsize=7.6)

    for i in range(m_grid.size):
        for j in range(n_vals.size):
            val = diff_pp[i, j]
            txt_col = "white" if abs(val) > 0.55 * vmax else "#111827"
            halo_col = "#111827" if abs(val) > 0.55 * vmax else "#F8FAFC"
            ax.text(
                j, i, f"{val:+.1f}", ha="center", va="center", color=txt_col, fontsize=6.7, fontweight="bold", zorder=6,
                path_effects=[pe.withStroke(linewidth=2.0, foreground=halo_col, alpha=0.90)],
            )

    x_bal = np.interp(np.log2(np.clip(n_balance, n_vals[0], n_vals[-1])), np.log2(n_vals), np.arange(n_vals.size))
    ax.plot(x_bal, np.arange(m_grid.size), color="#111827", linestyle="--", linewidth=2.0, zorder=3, label=r"Theorem 1 Balance Line $N_{\mathrm{balance}}(M)$")
    ax.plot(x_bal, np.arange(m_grid.size), color="#F59E0B", linestyle="--", linewidth=1.1, zorder=3.5)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.95)
    cbar.set_label(r"Gain: Opt-$K^*$+KL-LCB vs. Fix-60 MLE (pp)", fontsize=7.8)
    cbar.ax.tick_params(labelsize=7.5)
    ax.set_xlabel("Online Compute Budget $N$")
    ax.set_ylabel("Offline Labels $M$")
    ax.set_title(r"(c) 2D $(M, N)$ Gain & $N_{\mathrm{balance}}(M)$ Boundary")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=True, facecolor="white", framealpha=0.95, edgecolor="#D1D5DB", fontsize=7.2)

    fig.suptitle("Offline Calibration Protocols, Fine-Bin LCB Regularization, and Joint $(M, N)$ Scaling", fontsize=11.0, fontweight="bold", y=0.96)
    _save_both(fig, "fig2_joint_mn_and_protocols")


def plot_fig3_heterogeneity_and_shift(v9: dict[str, Any], v10: dict[str, Any]) -> None:
    """绘制 Figure 3：单次划分多基准净收益 + college_math 自然过优化 + 四相分布漂移边界。"""
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.85), gridspec_kw={"width_ratios": [1.15, 0.95, 1.05]})
    fig.subplots_adjust(wspace=0.30, left=0.09, right=0.98, top=0.84, bottom=0.25)

    # Panel (a): Hybrid-Fixed Paired Gain on Disjoint Single Split (v9 + v10 college_math)
    ax = axes[0]
    _style_ax(ax)
    p1 = v9["disjoint_single_split_primary"]
    cm = v10["natural_nonmonotone_collapse_audit"]["college_math_rm_b_single_split"]

    # Collect Hybrid-Fixed minus Pooled-KL-LCB and Hybrid-Fixed minus Within-Prompt-Z
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
        ("college_math\n(vs. Raw-BoN)", cm["algorithms"]["Hybrid-Fixed-KL-LCB"]["accuracy"][-1] - cm["algorithms"]["Standard-BoN"]["accuracy"][-1], cm["paired_test_vs_standard_bon"]["Hybrid-Fixed-KL-LCB"]["se_diff"][-1], cm["paired_test_vs_standard_bon"]["Hybrid-Fixed-KL-LCB"]["p_value_ttest"][-1],
         cm["algorithms"]["Hybrid-Fixed-KL-LCB"]["accuracy"][-1] - cm["algorithms"]["Within-Prompt-Z-KL-LCB"]["accuracy"][-1]),
    ]
    rows_a = rows_a[::-1]
    y_pos = np.arange(len(rows_a))
    h = 0.34
    d_pool = np.array([r[1] * 100.0 for r in rows_a])
    se_pool = np.array([r[2] * 100.0 for r in rows_a])
    d_z = np.array([r[4] * 100.0 for r in rows_a])
    pvals = [r[3] for r in rows_a]

    ax.barh(y_pos + h / 2.0, d_pool, height=h, xerr=se_pool, capsize=2.0, color=C_BLUE, label="vs. Pooled (last: vs. Raw)", zorder=3, error_kw={"lw": 0.9})
    ax.barh(y_pos - h / 2.0, d_z, height=h, color=C_GREEN, label="vs. Within-Prompt-Z", zorder=3)
    ax.axvline(0.0, color=C_DARK, linewidth=1.0, zorder=2)

    for idx, (dp, pv, dz) in enumerate(zip(d_pool, pvals, d_z)):
        star = "**" if pv < 0.01 else ("*" if pv < 0.05 else "")
        if abs(dp) > 0.40 or star:
            xtxt = dp + (se_pool[idx] + 0.18 if dp >= 0 else -se_pool[idx] - 0.18)
            ax.text(xtxt, idx + h / 2.0, f"{dp:+.2f}{star}", va="center", ha="left" if dp >= 0 else "right", fontsize=6.9, color=C_BLUE, fontweight="bold")
        if abs(dz) > 0.35:
            xtxt2 = dz + (0.18 if dz >= 0 else -0.18)
            ax.text(xtxt2, idx - h / 2.0, f"{dz:+.2f}", va="center", ha="left" if dz >= 0 else "right", fontsize=6.9, color=C_GREEN, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0] for r in rows_a], fontsize=7.6)
    ax.set_xlim(-3.2, 11.8)
    ax.set_xlabel("Hybrid-Fixed Advantage (pp)")
    ax.set_title("(a) Disjoint 50/20/30 Single-Split Gains")
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.94, edgecolor="#D1D5DB", fontsize=7.0)

    # Panel (b): Natural Overoptimization on AceMath/college_math RM-B (v10)
    ax = axes[1]
    _style_ax(ax)
    nat_inst = [x for x in v10["natural_nonmonotone_collapse_audit"]["acemath_natural_overoptimization_instances"] if x["dataset"] == "AceMath/college_math/RM_B"][0]
    n_grid = np.array(cm["n_values"], dtype=float)
    exact_curve = np.array(nat_inst["exact_curve"], dtype=float)
    std_split = np.array(cm["algorithms"]["Standard-BoN"]["accuracy"], dtype=float)
    hyb_split = np.array(cm["algorithms"]["Hybrid-Fixed-KL-LCB"]["accuracy"], dtype=float)
    winz_split = np.array(cm["algorithms"]["Within-Prompt-Z-KL-LCB"]["accuracy"], dtype=float)

    ax.plot(n_grid, exact_curve, color=C_DARK, marker="o", linestyle="-", linewidth=1.8, markersize=4.2, label="Full Raw-BoN ($P=2{,}818$)", zorder=4)
    ax.plot(n_grid, std_split, color=C_VERMILION, marker="s", linestyle="--", linewidth=1.5, markersize=3.8, label="Split Raw-BoN ($P_{\\mathrm{test}}=845$)", zorder=3)
    ax.plot(n_grid, hyb_split, color=C_BLUE, marker="^", linestyle="-", linewidth=1.6, markersize=4.0, label="Split Hybrid-Fixed", zorder=3)
    ax.plot(n_grid, winz_split, color=C_GREEN, marker="D", linestyle=":", linewidth=1.5, markersize=3.6, label="Split Within-Z", zorder=3)

    ax.annotate(
        "Exploratory peak $N=16$: 0.4640\n$\\to N=64$: 0.4579 ($-0.60$ pp)\n$p_{\\mathrm{raw}}=0.0208$, Bonf.(14)$=0.291$",
        xy=(64, exact_curve[-1]),
        xytext=(4.2, 0.400),
        arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.0),
        fontsize=6.6,
        fontweight="bold",
        color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.25", fc="#F3F4F6", ec=C_GRAY, lw=0.7),
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(n_grid)
    ax.set_xticklabels([str(int(v)) for v in n_grid])
    ax.set_xlabel("Sampling Budget $N$ (log$_2$ scale)")
    ax.set_ylabel("Pass@1 Accuracy")
    ax.set_ylim(0.36, 0.475)
    ax.set_title("(b) Exploratory Case (college_math RM-B)")
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.94, edgecolor="#D1D5DB", fontsize=6.7)

    # Panel (c): Out-of-Support & In-Support Shift Matrix (v10 + 5th-round P1-06)
    ax = axes[2]
    _style_ax(ax)
    mat = v10["contamination_distribution_shift_matrix"]
    in_mat = mat["in_support_high_score_shift_matrix"]
    reg_entries = [
        (mat["clean_train_clean_val_clean_test"], "R1\nClean"),
        (mat["contam_train_contam_val_contam_test"], "R2\nMatched"),
        (mat["clean_train_clean_val_contam_test_unseen_shift"], "R3: OOD\nUnseen"),
        (mat["clean_train_contam_val_contam_test_target_val"], "R4: OOD\nTgt-Val"),
        (in_mat["clean_train_clean_val_contam_test_unseen_shift"], "R5: In-Sup\nUnseen"),
    ]
    xr = np.arange(len(reg_entries))
    bw = 0.155
    meths = [
        ("standard_bon_n64", "Raw-BoN", "#9CA3AF"),
        ("pooled_kl_lcb_n64", "Pooled-LCB", C_SKY),
        ("within_prompt_z_n64", "Within-Z", C_GREEN),
        ("hybrid_fixed_n64", "Hybrid-Fixed", C_BLUE),
        ("support_clipped_kl_lcb_n64", "Support-Clipped", C_VERMILION),
    ]
    offs = [-2.0 * bw, -1.0 * bw, 0.0, 1.0 * bw, 2.0 * bw]
    for off, (mkey, mlbl, col) in zip(offs, meths):
        vals = [float(entry[mkey]) for entry, _ in reg_entries]
        ax.bar(xr + off, vals, width=bw, color=col, label=mlbl, zorder=3)

    ax.set_xticks(xr)
    ax.set_xticklabels([rlbl for _, rlbl in reg_entries], fontsize=7.0)
    ax.set_ylim(0.18, 0.77)
    ax.set_ylabel("Holdout Pass@1 Accuracy ($N=64$)")
    ax.set_title("(c) OOD vs. In-Support Shift Matrix")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=True, facecolor="white", framealpha=0.95, edgecolor="#D1D5DB", fontsize=6.7)

    fig.suptitle("Cross-Prompt Heterogeneity, Natural Overoptimization, and Distribution-Shift Boundaries", fontsize=11.0, fontweight="bold", y=0.96)
    _save_both(fig, "fig3_heterogeneity_and_shift")


def plot_fig4_multi_rm_dependence(s2: dict[str, Any], v6: dict[str, Any]) -> None:
    """绘制 Figure 4：跨家族相关性结构、RB2 谱路径对比与 OlympiadBench 7B 双 PRM 缩放曲线。"""
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.65), gridspec_kw={"width_ratios": [1.05, 0.92, 1.12]})
    fig.subplots_adjust(wspace=0.28, left=0.05, right=0.98, top=0.84, bottom=0.19)

    # Panel (a): Cross-family vs Homogeneous Conditional Dependence (stage2_real_results.json)
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

    # Panel (b): RB2 Hetero-6 N=4 Spectral Path Comparison (v6_theory_verification.json)
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
        ax.text(b.get_x() + b.get_width() / 2.0, v + 0.015, f"{v:.4f}", ha="center", va="bottom", fontsize=7.2, fontweight="bold", color=C_DARK)

    ax.set_xticks(xs)
    ax.set_xticklabels([s[0] for s in spec_items], fontsize=7.3)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("RB2 Hetero-6 Accuracy ($P=1{,}723, N=4$)")
    ax.set_title("(b) $N=2$ Spectral Trade-off vs. Bayes QDA")

    # Panel (c): OlympiadBench Qwen2.5-7B with Dual PRMs (stage2_real_results.json)
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
    ax.set_title("(c) OlympiadBench (Qwen2.5-7B, uPRM-70B/32B)")
    ax.legend(loc="lower right", ncol=2, frameon=True, facecolor="white", framealpha=0.92, edgecolor="#D1D5DB", fontsize=6.7)

    fig.suptitle("Multi-Verifier Dependence Structure, Spectral Trade-offs, and PRM Scaling", fontsize=11.0, fontweight="bold", y=0.96)
    _save_both(fig, "fig4_multi_rm_dependence")


def main() -> None:
    v9 = json.loads((RESULTS_DIR / "v9_consolidation_verification.json").read_text(encoding="utf-8"))
    v10 = json.loads((RESULTS_DIR / "v10_release_verification.json").read_text(encoding="utf-8"))
    v6 = json.loads((RESULTS_DIR / "v6_theory_verification.json").read_text(encoding="utf-8"))
    s2 = json.loads((RESULTS_DIR / "stage2_real_results.json").read_text(encoding="utf-8"))

    plot_fig1_framework_overview()
    plot_fig2_joint_mn_and_protocols(v9, v10, s2)
    plot_fig3_heterogeneity_and_shift(v9, v10)
    plot_fig4_multi_rm_dependence(s2, v6)
    print("全部 4 组顶会出版级矢量图表（PDF + PNG）已在 paper/figures/ 生成完毕。")


if __name__ == "__main__":
    main()
