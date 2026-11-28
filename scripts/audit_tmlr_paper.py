r"""TMLR 论文双盲防泄密、独立引用闭环、TMLR 专属图表与静态质量自动化审计脚本。

对 paper/tmlr/ 执行六项硬断言校验：
1. 官方样式 tmlr.sty / tmlr.bst、独立文献库 references_tmlr.bib、源码与双编译 PDF 存在性；
2. LaTeX 环境配对、花括号平衡及禁止与 tmlr.sty 冲突的包（geometry, lmodern, numbers-natbib）；
3. 标签与引用闭环检查（0 未定义引用、0 未引用死标签、references_tmlr.bib 100% 闭环覆盖）；
4. 双图路径 \graphicspath{{figures/}{../figures/}} 与 >= 5 张 TMLR 专属矢量图（paper/tmlr/figures/*.pdf & *.png）完整性；
5. 双盲投稿 PDF (paper/tmlr/main_tmlr.pdf) 零身份泄密与反工程/去匿名痕迹硬扫描（禁止 Repo Theorem、PROOF.md、内部脚本/函数名及本地路径）；
6. 预印本 PDF (paper/tmlr/main_tmlr_preprint.pdf) 实名署名完整性及 PDF 无 ?? 未解析引用检查。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMLR_DIR = PROJECT_ROOT / "paper" / "tmlr"
MAIN_TEX = TMLR_DIR / "main_tmlr.tex"
PREPRINT_TEX = TMLR_DIR / "main_tmlr_preprint.tex"
ANON_PDF = TMLR_DIR / "main_tmlr.pdf"
PREPRINT_PDF = TMLR_DIR / "main_tmlr_preprint.pdf"
REFS_BIB = TMLR_DIR / "references_tmlr.bib"
TMLR_FIG_DIR = TMLR_DIR / "figures"
FALLBACK_FIG_DIR = PROJECT_ROOT / "paper" / "figures"

# 双盲稿件绝对禁止出现的敏感身份关键词（全部按不区分大小写匹配）
FORBIDDEN_ANON_TOKENS = (
    "xinkang",
    "3203235490",
    "0009-0007-5524-7050",
    "6riemann9",
    "24563692",
    "24563479",
    "independent researcher",
    "wuhan",
    "acknowledgments",
)

# 双盲 PDF 中严禁残留的内部工程/版本号/可检索仓库符号痕迹词（全部按不区分大小写匹配）
FORBIDDEN_ENGINEERING_LEAK_TOKENS = (
    "repo theorem",
    "proof.md",
    "thm. 6a",
    "thm. 7a",
    "prop. 6.1",
    "prop. 7.1",
    "verify_v9",
    "verify_v10",
    "verify_tmlr_diagnostics",
    "plot_paper_figures",
    "plot_tmlr_figures",
    "tight_joint_minimax_scaling",
    "empirical_quantile_spacings_bounds",
    "diagnose_simpson_confounding",
    "diagnose_within_prompt_residual",
    "verify_sample_zscore_extremal_convergence",
    "evaluate_single_split_prompt_heterogeneity",
    "solve_positive_quadrant_qp",
    "empiricalscorecalibrator",
    "v9_consolidation",
    "v10_release",
)


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


def _strip_preprint_only_blocks(tex_nc: str) -> str:
    """粗略剔除 \\ifdefined\\USEPREPRINT ... \\else / \\fi 中仅面向预印本的分支，以便检查双盲源码。"""
    # 逐行跟踪 \ifdefined\USEPREPRINT ... \else ... \fi（不支持多层嵌套时已足够覆盖本文件结构）
    out: list[str] = []
    mode = "both"  # "both" | "preprint" | "anon"
    for line in tex_nc.splitlines():
        s = line.strip()
        if s.startswith(r"\ifdefined\USEPREPRINT"):
            mode = "preprint"
            continue
        if mode == "preprint" and s.startswith(r"\else"):
            mode = "anon"
            continue
        if mode in ("preprint", "anon") and s.startswith(r"\fi"):
            mode = "both"
            continue
        if mode != "preprint":
            out.append(line)
    return "\n".join(out)


def _check_latex_and_style(tex_nc: str) -> list[str]:
    """核验 TMLR 样式完整性、包冲突约束与 LaTeX 语法配对。"""
    errors: list[str] = []
    for req_file in ("tmlr.sty", "tmlr.bst", "references_tmlr.bib", "main_tmlr.tex", "main_tmlr_preprint.tex"):
        if not (TMLR_DIR / req_file).exists():
            errors.append(f"缺少 TMLR 必需文件: paper/tmlr/{req_file}")

    if r"\usepackage{tmlr}" not in tex_nc:
        errors.append("main_tmlr.tex 必须保留默认双盲 \\usepackage{tmlr} 分支")
    if r"\bibliography{references_tmlr}" not in tex_nc:
        errors.append("main_tmlr.tex 必须使用独立的 \\bibliography{references_tmlr}")
    if r"\graphicspath{{figures/}{../figures/}}" not in tex_nc:
        errors.append("main_tmlr.tex 必须配置双路径 \\graphicspath{{figures/}{../figures/}}（TMLR 专属目录优先）")
    if re.search(r"\\PassOptionsToPackage\{[^}]*numbers[^}]*\}\{natbib\}", tex_nc):
        errors.append("main_tmlr.tex 不得为 natbib 传入 numbers 选项（与 tmlr.sty 冲突）")
    if re.search(r"\\usepackage(\[[^\]]*\])?\{geometry\}", tex_nc):
        errors.append("main_tmlr.tex 存在冗余的 geometry 宏包加载（与 tmlr.sty 冲突）")
    if re.search(r"\\usepackage(\[[^\]]*\])?\{lmodern\}", tex_nc):
        errors.append("main_tmlr.tex 存在冗余的 lmodern 宏包加载（tmlr.sty 已内含 lmodern）")
    if r"\bibliographystyle{tmlr}" not in tex_nc:
        errors.append("main_tmlr.tex 未使用官方 \\bibliographystyle{tmlr}")

    stack: list[str] = []
    for m in re.finditer(r"\\(begin|end)\{([^\}]+)\}", tex_nc):
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

    depth = 0
    escaped = False
    for ch in tex_nc:
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


def _check_refs_cites_and_figs(tex_nc: str, bib_txt: str) -> tuple[list[str], dict[str, object]]:
    """检查交叉引用、references_tmlr.bib 闭环与 >=5 张 TMLR 专属图表文件完整性。"""
    errors: list[str] = []
    labels = set(re.findall(r"\\label\{([^\}]+)\}", tex_nc))
    refs: set[str] = set()
    for grp in re.findall(r"\\(?:Cref|cref|ref|eqref)\{([^\}]+)\}", tex_nc):
        for item in grp.split(","):
            key = item.strip()
            if key:
                refs.add(key)

    undefined_refs = sorted(refs - labels)
    unreferenced_labels = sorted(labels - refs)
    if undefined_refs:
        errors.append(f"TMLR 论文存在未定义的交叉引用: {undefined_refs}")
    if unreferenced_labels:
        errors.append(f"TMLR 论文存在未被引用的死标签: {unreferenced_labels}")

    bib_keys = set(re.findall(r"@\w+\{([^,\s]+),", bib_txt))
    cited_keys: set[str] = set()
    for grp in re.findall(r"\\cite[pt]?\{([^\}]+)\}", tex_nc):
        for item in grp.split(","):
            key = item.strip()
            if key:
                cited_keys.add(key)

    missing_cites = sorted(cited_keys - bib_keys)
    uncited_bib = sorted(bib_keys - cited_keys)
    if missing_cites:
        errors.append(f"references_tmlr.bib 缺失 TMLR 引用的文献条目: {missing_cites}")
    if uncited_bib:
        errors.append(f"references_tmlr.bib 存在未被 TMLR 论文引用的条目: {uncited_bib}")

    fig_stems = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^\}]+)\}", tex_nc)
    if len(fig_stems) < 5:
        errors.append(f"main_tmlr.tex 插图数量不足（仅 {len(fig_stems)} 张，预期 >= 5 张）")

    tmlr_hits = 0
    for stem in fig_stems:
        base = Path(stem).stem if "." in Path(stem).name else stem
        tmlr_pdf = TMLR_FIG_DIR / f"{base}.pdf"
        tmlr_png = TMLR_FIG_DIR / f"{base}.png"
        fb_pdf = FALLBACK_FIG_DIR / f"{base}.pdf"
        if tmlr_pdf.exists() and tmlr_pdf.stat().st_size >= 5000:
            tmlr_hits += 1
            if not tmlr_png.exists() or tmlr_png.stat().st_size < 5000:
                errors.append(f"TMLR 图表缺少 300 DPI PNG 预览图: {tmlr_png.relative_to(PROJECT_ROOT)}")
        elif fb_pdf.exists() and fb_pdf.stat().st_size >= 5000:
            errors.append(f"插图 {base} 仍命中旧 NeurIPS 目录而非 TMLR 专属目录 paper/tmlr/figures/{base}.pdf")
        else:
            errors.append(f"缺失或异常的矢量 PDF 图表: paper/tmlr/figures/{base}.pdf")

    return errors, {
        "n_labels": len(labels),
        "n_refs": len(refs),
        "n_bib_keys": len(bib_keys),
        "n_cited_keys": len(cited_keys),
        "n_figs": len(fig_stems),
        "n_tmlr_figs": tmlr_hits,
    }


def _check_pdf_blindness_and_hygiene(tex_nc: str) -> tuple[list[str], dict[str, object]]:
    """对双盲 PDF 与预印本 PDF 执行全文防泄密硬扫描、反工程痕迹检测及路径洁净检查。"""
    errors: list[str] = []
    anon_pages: int | None = None
    pre_pages: int | None = None

    anon_tex_lower = _strip_preprint_only_blocks(tex_nc).lower()
    src_eng_leaks = [tok for tok in FORBIDDEN_ENGINEERING_LEAK_TOKENS if tok in anon_tex_lower]
    if src_eng_leaks:
        errors.append(f"main_tmlr.tex 双盲分支仍残留内部工程/仓库检索词: {src_eng_leaks}")

    try:
        import PyPDF2
    except ImportError:
        errors.append("缺少 PyPDF2，无法对双盲 PDF 执行文本防泄密扫描")
        return errors, {"anon_pages": None, "preprint_pages": None}

    if not ANON_PDF.exists():
        errors.append("缺失编译产物 paper/tmlr/main_tmlr.pdf")
    else:
        reader_a = PyPDF2.PdfReader(str(ANON_PDF))
        anon_pages = len(reader_a.pages)
        page_texts = [p.extract_text() or "" for p in reader_a.pages]
        first_page_norm = re.sub(r"\s+", " ", page_texts[0])
        if "Anonymous authors" not in first_page_norm:
            errors.append("双盲 PDF 第 1 页未检测到 'Anonymous authors'")
        if "Paper under double-blind review" not in first_page_norm:
            errors.append("双盲 PDF 第 1 页未检测到 'Paper under double-blind review'")
        if "Under review as submission to TMLR" not in first_page_norm:
            errors.append("双盲 PDF 第 1 页未检测到 TMLR 双盲页眉")

        full_text = "\n".join(page_texts)
        full_text_lower = full_text.lower()
        leaked = [tok for tok in FORBIDDEN_ANON_TOKENS if tok in full_text_lower]
        if leaked:
            errors.append(f"双盲 PDF 检测到严重身份信息泄密关键词: {leaked}")
        eng_leaked = [tok for tok in FORBIDDEN_ENGINEERING_LEAK_TOKENS if tok in full_text_lower]
        if eng_leaked:
            errors.append(f"双盲 PDF 检测到内部工程/仓库检索痕迹词: {eng_leaked}")
        if "??" in full_text:
            errors.append("双盲 PDF 检测到未解析交叉引用 '??'")

    if not PREPRINT_PDF.exists():
        errors.append("缺失编译产物 paper/tmlr/main_tmlr_preprint.pdf")
    else:
        reader_p = PyPDF2.PdfReader(str(PREPRINT_PDF))
        pre_pages = len(reader_p.pages)
        pre_texts = [p.extract_text() or "" for p in reader_p.pages]
        pre_p1 = re.sub(r"\s+", " ", pre_texts[0])
        if "Xinkang Zhang" not in pre_p1:
            errors.append("预印本 PDF 第 1 页未检测到实名作者 'Xinkang Zhang'")
        if "??" in "\n".join(pre_texts):
            errors.append("预印本 PDF 检测到未解析交叉引用 '??'")

    leak_patterns = [r"file:///[A-Za-z]:/", r"[A-Za-z]:\\\\Users\\\\", r"[A-Za-z]:\\Users\\"]
    for pat in leak_patterns:
        if re.search(pat, tex_nc):
            errors.append(f"main_tmlr.tex 疑似包含本地绝对路径泄漏 ({pat})")

    return errors, {"anon_pages": anon_pages, "preprint_pages": pre_pages}


def main() -> int:
    if not MAIN_TEX.exists():
        print(f"[FAIL] 缺失 {MAIN_TEX}")
        return 1
    if not REFS_BIB.exists():
        print(f"[FAIL] 缺失 {REFS_BIB}")
        return 1

    tex = MAIN_TEX.read_text(encoding="utf-8")
    bib = REFS_BIB.read_text(encoding="utf-8")
    tex_nc = _strip_comments(tex)

    e1 = _check_latex_and_style(tex_nc)
    e2, ref_stats = _check_refs_cites_and_figs(tex_nc, bib)
    e3, pdf_stats = _check_pdf_blindness_and_hygiene(tex_nc)

    all_errors = e1 + e2 + e3
    print("=== TMLR 论文产物防泄密、独立引用与图表质量自动化审计报告 ===")
    print(f"- LaTeX 标签 / 引用数量: {ref_stats['n_labels']} labels / {ref_stats['n_refs']} unique refs")
    print(f"- BibTeX 条目 / 正文引用: {ref_stats['n_cited_keys']} / {ref_stats['n_bib_keys']} (references_tmlr.bib)")
    print(f"- TMLR 专属矢量图数量: {ref_stats['n_tmlr_figs']} / {ref_stats['n_figs']}")
    print(f"- 双盲投稿 PDF 页数: {pdf_stats['anon_pages']} 页 | 预印本 PDF 页数: {pdf_stats['preprint_pages']} 页")

    if all_errors:
        print("\n[FAIL] 发现以下问题:")
        for err in all_errors:
            print(f"  - {err}")
        return 1

    print("\n[PASS] TMLR 双盲零泄密、反工程痕迹扫描、独立 BibTeX 闭环、5 张 TMLR 专属图与双 PDF 规范 100% 通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
