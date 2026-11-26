"""论文产物自动化静态质量与编译页界审计脚本。

对 paper/ 与跨文档同步状态执行六项硬断言校验：
1. LaTeX 语法、环境栈配对、花括号平衡及 NeurIPS 2026 官方样式约束检查；
2. 标签与引用闭环检查（0 未定义引用、0 未引用死标签、27/27 BibTeX 覆盖率）；
3. 论文主图矢量 .pdf 与 300 DPI .png 文件完整性检查；
4. 实验数值与第五轮审稿数学表述防回归检查（Algorithm 1 平局规则、Safe-Fisher 正交分支、Z-Score 适用域、16 项 Checklist 等）；
5. 编译产物 paper/main.pdf 正文页数（<= 9 页）与 References 边界校验（若存在 PyPDF2）；
6. 公开文档本地绝对路径隐私泄漏检查。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = PROJECT_ROOT / "paper"
MAIN_TEX = PAPER_DIR / "main.tex"
MAIN_PDF = PAPER_DIR / "main.pdf"
REFS_BIB = PAPER_DIR / "references.bib"
STY_FILE = PAPER_DIR / "neurips_2026.sty"
FIG_DIR = PAPER_DIR / "figures"


def _strip_comments(tex: str) -> str:
    """移除 LaTeX 行内未转义注释（保留 \\%）。"""
    cleaned_lines = []
    for line in tex.splitlines():
        out = []
        escaped = False
        for ch in line:
            if ch == "%" and not escaped:
                break
            out.append(ch)
            escaped = (ch == "\\") and not escaped
            if ch != "\\":
                escaped = False
        cleaned_lines.append("".join(out))
    return "\n".join(cleaned_lines)


def _check_latex_syntax(tex: str, tex_nocomment: str) -> list[str]:
    """检查环境配对、花括号平衡以及宏包/NeurIPS 2026 官方样式约束。"""
    errors: list[str] = []

    if (PAPER_DIR / "neurips_2025.sty").exists():
        errors.append("仓库内仍残留旧版样式 paper/neurips_2025.sty，应使用 paper/neurips_2026.sty")

    if not STY_FILE.exists():
        errors.append("缺少官方样式文件 paper/neurips_2026.sty")
    else:
        sty_txt = STY_FILE.read_text(encoding="utf-8")
        if r"\ProvidesPackage{neurips_2026}" not in sty_txt:
            errors.append("paper/neurips_2026.sty 中 \\ProvidesPackage 与文件名不一致")

    if (
        r"\usepackage[preprint]{neurips_2026}" not in tex_nocomment
        and r"\usepackage{neurips_2026}" not in tex_nocomment
        and r"\usepackage[main]{neurips_2026}" not in tex_nocomment
    ):
        errors.append("main.tex 未加载官方 neurips_2026 样式包")

    if "neurips_2025" in tex_nocomment:
        errors.append("main.tex 正文仍残留旧样式引用 neurips_2025")

    if re.search(r"\\usepackage(\[[^\]]*\])?\{geometry\}", tex_nocomment):
        errors.append("main.tex 存在冗余的 geometry 宏包加载（与 neurips_2026.sty 冲突）")
    if re.search(r"\\usepackage(\[[^\]]*\])?\{lmodern\}", tex_nocomment):
        errors.append("main.tex 存在冗余的 lmodern 宏包加载（覆盖 neurips_2026.sty ptm 字体）")

    # 检查 \begin{...} / \end{...} 配对
    stack: list[str] = []
    for m in re.finditer(r"\\(begin|end)\{([^\}]+)\}", tex_nocomment):
        kind, env = m.group(1), m.group(2)
        if kind == "begin":
            stack.append(env)
        else:
            if not stack or stack[-1] != env:
                top = stack[-1] if stack else "EMPTY"
                errors.append(f"LaTeX 环境不匹配：遇到 \\end{{{env}}}，栈顶为 {top}")
                break
            stack.pop()
    if stack:
        errors.append(f"LaTeX 存在未闭合环境：{stack}")

    # 检查花括号平衡（忽略 \{ \} 及 verbatim 等风险块，本论文无 verbatim 块）
    depth = 0
    escaped = False
    for ch in tex_nocomment:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                errors.append("LaTeX 存在多余的右花括号 '}'")
                break
    if depth != 0:
        errors.append(f"LaTeX 花括号未配对，最终净深度 = {depth}")

    return errors


def _check_refs_and_cites(tex_nocomment: str, bib_txt: str) -> tuple[list[str], dict[str, object]]:
    """检查 label/ref 闭环及 BibTeX 引用完整性。"""
    errors: list[str] = []

    labels = set(re.findall(r"\\label\{([^\}]+)\}", tex_nocomment))
    refs: set[str] = set()
    for grp in re.findall(r"\\(?:Cref|cref|ref|eqref)\{([^\}]+)\}", tex_nocomment):
        for item in grp.split(","):
            key = item.strip()
            if key:
                refs.add(key)

    undefined_refs = sorted(refs - labels)
    unreferenced_labels = sorted(labels - refs)
    if undefined_refs:
        errors.append(f"存在未定义的交叉引用: {undefined_refs}")
    if unreferenced_labels:
        errors.append(f"存在未被引用的死标签: {unreferenced_labels}")

    bib_keys = set(re.findall(r"@\w+\{([^,\s]+),", bib_txt))
    cited_keys: set[str] = set()
    for grp in re.findall(r"\\cite[pt]?\{([^\}]+)\}", tex_nocomment):
        for item in grp.split(","):
            key = item.strip()
            if key:
                cited_keys.add(key)

    missing_cites = sorted(cited_keys - bib_keys)
    uncited_bib = sorted(bib_keys - cited_keys)
    if missing_cites:
        errors.append(f"存在 references.bib 缺失的引用 key: {missing_cites}")
    if uncited_bib:
        errors.append(f"references.bib 存在未被正引用的条目: {uncited_bib}")

    stats = {
        "n_labels": len(labels),
        "n_refs": len(refs),
        "n_bib_keys": len(bib_keys),
        "n_cited_keys": len(cited_keys),
        "undefined_refs": undefined_refs,
        "unreferenced_labels": unreferenced_labels,
        "missing_cites": missing_cites,
        "uncited_bib": uncited_bib,
    }
    return errors, stats


def _check_figures(tex_nocomment: str) -> tuple[list[str], list[str]]:
    """检查正文中引用的所有图像文件是否同时拥有有效 PDF 与 PNG 版本。"""
    errors: list[str] = []
    fig_stems = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^\}]+)\}", tex_nocomment)
    if len(fig_stems) < 4:
        errors.append(f"paper/main.tex 插图数量不足（仅 {len(fig_stems)} 张，预期 >= 4 张）")

    for stem in fig_stems:
        base = Path(stem).stem if "." in Path(stem).name else stem
        pdf_p = FIG_DIR / f"{base}.pdf"
        png_p = FIG_DIR / f"{base}.png"
        if not pdf_p.exists() or pdf_p.stat().st_size < 5000:
            errors.append(f"缺失或异常的矢量 PDF 图表: {pdf_p.relative_to(PROJECT_ROOT)}")
        if not png_p.exists() or png_p.stat().st_size < 5000:
            errors.append(f"缺失或异常的 PNG 预览图表: {png_p.relative_to(PROJECT_ROOT)}")

    return errors, fig_stems


def _check_numbers_abstract_and_hygiene(tex_nocomment: str, bib_txt: str) -> tuple[list[str], dict[str, object]]:
    """检查数值与数学表述防回归、Checklist 完整性、PDF 页界、摘要词数及路径洁净度。"""
    errors: list[str] = []

    if "0.4366" in tex_nocomment or "43.66" in tex_nocomment:
        errors.append("paper/main.tex 仍残留已废弃的错误数值 0.4366")
    if "0.4591" not in tex_nocomment or "0.1810" not in tex_nocomment:
        errors.append("paper/main.tex 缺少修正后的 college_math RM-B Within-Z (0.4591) 或 p_t (0.1810)")
    if "N_{\\mathrm{match}}" in tex_nocomment or "N_match" in tex_nocomment:
        errors.append("paper/main.tex 仍残留旧符号 N_match")

    # 第五轮审稿不变量校验（P0-01/P0-02/P1-01/P1-04/P1-05/P1-06/P0-08）
    if "breaking ties by raw score" in tex_nocomment.lower():
        errors.append("paper/main.tex 仍包含破坏 Minimax 证明的按 raw score 破平局分支")
    if "strictly uniform random tie-breaking" not in tex_nocomment:
        errors.append("paper/main.tex Algorithm 1 缺少 strictly uniform random tie-breaking 要求")
    if "Degenerate orthogonal branch" not in tex_nocomment:
        errors.append("paper/main.tex Theorem 5 缺少 Degenerate orthogonal branch 正交边界分支")
    if "whenever unsolvable prompts exist" in tex_nocomment.lower():
        errors.append("paper/main.tex 仍包含过宽的 'whenever unsolvable prompts exist' 表述")
    if r"\pi_0 > 0 = \pi_1" not in tex_nocomment:
        errors.append("paper/main.tex 缺少严格限定条件 \\pi_0 > 0 = \\pi_1")
    for required_tok in ("0.2908", "0.3629", "0.3785", "NeurIPS Paper Checklist"):
        if required_tok not in tex_nocomment:
            errors.append(f"paper/main.tex 缺少第五轮审稿必需核验条目或数值: {required_tok}")

    if r"\answerTODO" in tex_nocomment:
        errors.append("NeurIPS Paper Checklist 仍残留未填写的 \\answerTODO")
    n_checklist_answers = len(re.findall(r"Answer:\s*\\answer(?:Yes|No|NA)\{\}", tex_nocomment))
    if n_checklist_answers != 16:
        errors.append(f"NeurIPS Paper Checklist 回答数量必须为 16 项，当前检测到 {n_checklist_answers} 项")

    # 摘要词数与限定条件检查
    m_abs = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex_nocomment, re.S)
    abs_words = 0
    if not m_abs:
        errors.append("paper/main.tex 缺少 abstract 环境")
    else:
        raw_abs = m_abs.group(1)
        if r"\pi_0 > 0 = \pi_1" not in raw_abs:
            errors.append("Abstract 中缺少 Z-Score 严格适用限定 \\pi_0 > 0 = \\pi_1")
        norm_abs = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?\{([^\}]*)\}", r" \2 ", raw_abs)
        norm_abs = re.sub(r"\\[a-zA-Z]+", " ", norm_abs)
        norm_abs = re.sub(r"[${}\[\]\\^_~`]", " ", norm_abs)
        tokens = [t for t in re.split(r"\s+", norm_abs) if re.search(r"[A-Za-z0-9]", t)]
        abs_words = len(tokens)
        if not (160 <= abs_words <= 240):
            errors.append(f"Abstract 词数超出顶会目标区间 [160, 240]: {abs_words} 词")

    # 编译版 main.pdf 正文页数（<= 9 页）检验
    pdf_pages: int | None = None
    if MAIN_PDF.exists():
        try:
            import PyPDF2

            reader = PyPDF2.PdfReader(str(MAIN_PDF))
            pdf_pages = len(reader.pages)
            page_texts = [page.extract_text() or "" for page in reader.pages]
            for marker in ("Figure 4:", "8 Discussion"):
                found_idx = next((idx for idx, txt in enumerate(page_texts) if marker in txt), None)
                if found_idx is None:
                    errors.append(f"paper/main.pdf 未检出标记: {marker}")
                elif found_idx > 8:
                    errors.append(f"paper/main.pdf 正文内容溢出第 9 页：{marker} 出现在第 {found_idx + 1} 页")
            refs_idx = next((idx for idx, txt in enumerate(page_texts) if "References" in txt), None)
            if refs_idx is None:
                errors.append("paper/main.pdf 未检出 References")
            else:
                if any("Figure 4:" in txt for txt in page_texts[refs_idx:]):
                    errors.append("paper/main.pdf 中 Figure 4 浮动到了 References 页及之后")
        except ImportError:
            pass

    # README、RELATED_WORK 与 EXPERIMENT_REPORT 同步检查
    readme_txt = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    rel_txt = (PROJECT_ROOT / "RELATED_WORK.md").read_text(encoding="utf-8")
    report_txt = (PROJECT_ROOT / "results" / "EXPERIMENT_REPORT.md").read_text(encoding="utf-8")
    audit_txt = (PAPER_DIR / "REVIEW_AUDIT.md").read_text(encoding="utf-8") if (PAPER_DIR / "REVIEW_AUDIT.md").exists() else ""

    if "22 篇" in readme_txt:
        errors.append("README.md 仍残留过时的 '22 篇文献' 表述（应为 27 篇）")
    if "V9 主次聚焦" in readme_txt:
        errors.append("README.md 仍残留旧版本标题 'V9 主次聚焦'")

    m_bib_blk = re.search(r"```bibtex\s*(.*?)\s*```", rel_txt, re.S)
    if not m_bib_blk:
        errors.append("RELATED_WORK.md 缺少 BibTeX 代码块")
    elif m_bib_blk.group(1).strip() != bib_txt.strip():
        errors.append("RELATED_WORK.md 中的 BibTeX 与 paper/references.bib 未完全逐字节同步")

    for py_name in ("plot_paper_figures.py", "plot_stage2_figures.py"):
        py_txt = (PROJECT_ROOT / "scripts" / py_name).read_text(encoding="utf-8")
        cleaned_py = py_txt.replace("n_match_theory_by_m", "").replace("n_match", "")
        if "N_{\\mathrm{match}}" in cleaned_py or "N_match" in cleaned_py:
            errors.append(f"scripts/{py_name} 显示文本中仍残留旧符号 N_match")
        if "Hybrid-Adaptive" in cleaned_py:
            stripped_py = cleaned_py.replace('["Hybrid-Adaptive-KL-LCB"]', "").replace('"Hybrid-Adaptive-KL-LCB",', "")
            if "Hybrid-Adaptive" in stripped_py:
                errors.append(f"scripts/{py_name} 展示文本中仍残留旧命名 Hybrid-Adaptive")

    # 隐私与本地绝对路径泄漏检查
    leak_patterns = [r"file:///[A-Za-z]:/", r"[A-Za-z]:\\\\Users\\\\", r"[A-Za-z]:\\Users\\"]
    for doc_name, doc_txt in [
        ("README.md", readme_txt),
        ("RELATED_WORK.md", rel_txt),
        ("results/EXPERIMENT_REPORT.md", report_txt),
        ("paper/main.tex", tex_nocomment),
        ("paper/REVIEW_AUDIT.md", audit_txt),
    ]:
        for pat in leak_patterns:
            if re.search(pat, doc_txt):
                errors.append(f"{doc_name} 疑似包含本地绝对路径泄漏 ({pat})")

    return errors, {
        "abstract_approx_words": abs_words,
        "checklist_answers": n_checklist_answers,
        "main_pdf_pages": pdf_pages,
    }


def main() -> int:
    tex = MAIN_TEX.read_text(encoding="utf-8")
    bib = REFS_BIB.read_text(encoding="utf-8")
    tex_nc = _strip_comments(tex)

    e1 = _check_latex_syntax(tex, tex_nc)
    e2, ref_stats = _check_refs_and_cites(tex_nc, bib)
    e3, fig_stems = _check_figures(tex_nc)
    e4, meta_stats = _check_numbers_abstract_and_hygiene(tex_nc, bib)

    all_errors = e1 + e2 + e3 + e4

    print("=== 论文产物自动化静态审计报告 ===")
    print(f"- LaTeX 标签 / 引用数量: {ref_stats['n_labels']} labels / {ref_stats['n_refs']} unique refs")
    print(f"- BibTeX 条目 / 正文引用: {ref_stats['n_cited_keys']} / {ref_stats['n_bib_keys']}")
    print(f"- 矢量插图清单: {fig_stems}")
    print(f"- Abstract 估算英文词数: {meta_stats['abstract_approx_words']} 词")
    print(f"- NeurIPS 2026 Checklist 回答项: {meta_stats['checklist_answers']} / 16")
    if meta_stats["main_pdf_pages"] is not None:
        print(f"- 编译产物 paper/main.pdf 总页数: {meta_stats['main_pdf_pages']} 页（正文严格 <= 9 页）")

    if all_errors:
        print("\n[FAIL] 发现以下问题:")
        for err in all_errors:
            print(f"  - {err}")
        return 1

    print("\n[PASS] 全部静态语法、NeurIPS 2026 规范、第五轮审稿不变量、标签引用闭环、矢量图表及路径洁净度检查 100% 通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
