# LaTeX 编译脚本：paper/main.tex -> paper/main.pdf
# 用法：在 TeX Live / Overleaf / Linux/macOS 环境下于 paper/ 目录执行 `make`

FIGURES_DIR := figures
PDF_FIGS := $(wildcard $(FIGURES_DIR)/*.pdf)

all: main.pdf

main.pdf: main.tex references.bib neurips_2026.sty $(PDF_FIGS)
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	-bibtex main
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex

.PHONY: clean figures

figures:
	cd .. && python scripts/plot_paper_figures.py

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.synctex.gz main.pdf
