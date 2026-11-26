# 多工具联合论文质量审计与修正备忘录 (`REVIEW_AUDIT`)

> **审计基准版本**：`v1.0.0`（优化前 `173d565`） $\longrightarrow$ **顶会投稿就绪优化版**  
> **协同工具链**：`citecheck` / `arxiv` / `paper-search` MCP + `ml-paper-writing`、`research-paper-writing`、`sciwrite`、`academic-plotting`、`paper-review-zh-cn`、`academic-writing-skills-zh-cn` Skills。

---

## 1. 关键缺陷定位与修订对照总表

| 严重度 | 发现来源工具 | 原版本 (`v1.0.0`) 问题诊断 | 修正措施与最终验证状态 |
| :---: | :--- | :--- | :--- |
| **CRITICAL** | `sciwrite` Pass 5 + JSON 对账 | `paper/main.tex` Table 2 最后一行 `college_math RM-B` 将 `Within-Z` 错录为 `0.4366`，且省略最后一列 `vs. Raw` 的 $p_t$ 值与完整统计显著性解释 | 核对 `v10_release_verification.json` 后精确更正为 **`0.4591`**，补全 **$p_t = 0.1810$** 并加表注明确区分全量 `P=2,818` 显著回落（$p_t=0.0208$）与 845 题单项子集测试无显著性；明确加粗/下划线标记规则 |
| **CRITICAL** | `ml-paper-writing` + `academic-plotting` | 论文仅嵌入 1 张 PNG 位图（`fig:simpson` 且全文未引用），缺少核心概览图 Figure 1，无矢量 `.pdf` 图表；`scripts/plot_stage2_figures.py` 图表渲染中残留旧符号 `N_match` 与旧名 `Hybrid-Adaptive` | 修复 `scripts/plot_stage2_figures.py` 展示标签并重绘；新建 `scripts/plot_paper_figures.py` 导出 4 组 Serif 字体 + 色盲安全矢量 `.pdf` 与 300 DPI `.png`（Figure 1–4 涵盖模型边界/Minimax 相图/Holdout 控制流、协议/同 $K$ 消融/2D 表面、单次划分净增益/自然坍塌/四相漂移、多 RM 相关性/spectral path/OlympiadBench PRM）并通过逐图视觉检查 |
| **CRITICAL** | `paper-review-zh-cn` + `research-paper-writing` | `paper/main.tex` 存在 **13 个未引用死标签**（含附录 `app:minimax`, `app:ensemble`, `app:index` 与主图 `fig:simpson`、方程 `eq:cond-bin`, `eq:n-balance`, `eq:joint-lower`, `eq:unif-class`） | 重构全部 `\Cref`/`\eqref` 交叉引用网络，实现 **0 个未定义引用、0 个未引用死标签**（经 `audit_paper_artifacts.py` 自动验证） |
| **MAJOR** | `sciwrite` Pass 6 + `ml-paper-writing` | 原 Abstract 高达 **362 词**，含密集复杂符号串，却缺失最关键的实证定量收益与 $p$ 值 | 遵循 5-Sentence 摘要范式压炼至约 **200 词**，保留核心 $(\frac{\ln M}{M})^{\frac{\beta}{2\beta+1}}+N^{-\beta}$、$O_p(\sqrt{(\ln N)/N})$ 定理率并前置 `+2.41 pp / +1.83 pp / +42.29 pp` 核心实证结果 |
| **MAJOR** | `citecheck` + `arxiv` + `paper-search` + `search_web` | 全部 27 篇文献 100% 真实存在（0 伪造、0 撤稿），但部分条目的会议发表状态或卷页/URL/arXiv eprint 未及时更新 | 精确升级 `snell2024scaling`（ICLR 2025 Oral）、`dorner2025roc`（ICLR 2026）、`rame2024warm`（补 `pages={42048--42073}`）、`huang2025best`（补 `volume={267}, pages={25075--25126}`）、`garivier2011kl`、`bubeck2011xarmed`、`minsker2013estimation`、`jin2021pessimism`、`muller1985kernel`；同步更新 `RELATED_WORK.md` 并实现 27/27 全文引用 |
| **MAJOR** | `ml-paper-writing` + `research-paper-writing` | 论文缺少独立 Related Work 章节、缺少核心算法伪代码块、缺少 NeurIPS 标准 `.sty` 模板/匿名切换、缺少 Reproducibility/Broader Impact 与 NeurIPS Checklist | 集成标准 `paper/neurips_2025.sty` 与 `paper/Makefile`（支持 `[preprint]` 与双盲提交一键切换）；新增 §2 Related Work、Algorithm 1（`Sample-Split KL-LCB`）、Algorithm 2（`Hybrid-Fixed`）、§7.4 多 RM 实证检验、§8 Limitations/Broader Impact/Reproducibility 与 Appendix F NeurIPS Checklist |
| **MINOR** | `paper-review` `display-notation-provenance` | `README.md` 仍残留旧版本标题 `V9 主次聚焦`、旧计数 `22 篇文献`，且 `q'_pool` 下界与论文略微差 $0.01\times 10^{-3}$ | 更新 `README.md` 至 `v1.0.0` 顶会优化版、`27 篇文献`、完整文件清单，并在两处同步明确精确下界 $+7.044\times 10^{-3} > +7.03\times 10^{-3}$ |

---

## 2. 核心主张—证据—代码与图表映射表（Claim–Evidence Map）

| 论文主命题 / 实证主张 | 数学证明与算法位置 | 实证图表 / 表格证据 | 核心校验 JSON 与源码实现 |
| :--- | :--- | :--- | :--- |
| 联合 $(M,N)$ Minimax 率 $\Theta\big((\frac{\ln M}{M})^{\frac{\beta}{2\beta+1}} + N^{-\beta}\big)$ 与平衡线 $N_{\mathrm{balance}}(M)$ | `paper/main.tex` §4 (`Theorem 1`), `Appendix B` | Figure 1(b), Figure 2(c) (`N_balance(M)`) | `src/tts_theory/calibration_theory.py`: `tight_joint_minimax_scaling` |
| 全 $\beta>0$ 样本分割 Pyke 乘性间距界 $\hat{\pi}_k \in [\frac{1}{2K}, \frac{143}{72K})$ | `paper/main.tex` §4 (`Proposition 1`, `Algorithm 1`) | Table 1, Figure 2(a)(b) | `calibration_theory.py`: `empirical_quantile_spacings_bounds`; `v9` & `v10` |
| 跨题池化辛普森解析遮蔽 ($q'_{\mathrm{pool}} > +7.044\times 10^{-3}$ on $[3.5,4.2]$) | `paper/main.tex` §5.1 (`Theorem 2`), `Appendix C` | Table 2 (`S3`, `RB2`), Figure 3(a) | `heterogeneity_theory.py`: `diagnose_simpson_confounding` |
| $N=2$ 样本 $Z$ 秩上界与 $\pi_0>0$ 含不可解题在线 $Z_{i,N}$ 极值收敛与上限损失 | `paper/main.tex` §5.2 (`Theorems 3–4`), `Appendix C` | Table 2 (`AceMath`, `OlympiadBench`), Figure 3(a) | `heterogeneity_theory.py`: `verify_sample_zscore_extremal_convergence` |
| 三分支留出门控风险控制 ($\mathrm{Hybrid\text{-}Fixed}$, $\varepsilon_{\mathrm{refit}}\equiv 0$) 与未见漂移对策 | `paper/main.tex` §5.3 (`Proposition 2`, `Algorithm 2`) | Table 2, Table 3, Figure 3(a)(b)(c) | `real_eval.py`: `evaluate_single_split_prompt_heterogeneity`; `v9` & `v10` |
| 异构多裁判 `Safe-Fisher`、Bayes QDA $N^{-2/K}$、零乘子 QP `Min` 与 Copula | `paper/main.tex` §6 (`Theorems 5–6`), `Appendix D` | Figure 4(a)(b)(c) (`RB2`, `OlympiadBench`) | `ensemble_theory.py`: `solve_positive_quadrant_qp`; `v6` & `stage2` JSON |

---

## 3. 自动化核验与投稿前诚实说明

1. **参考文献外部核验结果**：本轮已通过 `citecheck.check_bibliography` + `arxiv.get_abstract` + `paper-search.search_openalex` + 会议官网检索核实 `paper/references.bib` 全部 27 篇条目，并将升级后文本与 `RELATED_WORK.md` 逐字节对齐；更新后已再次调用 `citecheck.check_bibliography` 复核无异常。
2. **代码与脚本回归测试**：全部图表的重绘脚本与仓库 `tests/` 下 73 项离线单元测试保持 100% 兼容性，未改动 `src/tts_theory/` 公共接口及 JSON 读取键名。
3. **本地 TeX 编译限制说明**：当前 Windows 环境未安装 `pdflatex` / `xelatex` / `bibtex` / `tectonic`，故尚无法在本地直接生成 `paper/main.pdf`。取而代之，`scripts/audit_paper_artifacts.py` 已对括号、环境配对、样式兼容、全部图表 `.pdf`/`.png` 存在性、`0` 未定义引用、`0` 死标签、`27/27` BibTeX 引用覆盖及数值防回归进行了严格静态校验；请在 Overleaf 或 TeX Live 环境中进入 `paper/` 目录执行 `make` 后核查最终 PDF 页数与浮动体排版，如需双盲审稿只需将 `paper/main.tex` 第 5 行的 `\usepackage[preprint]{neurips_2025}` 替换为 `\usepackage{neurips_2025}`。
