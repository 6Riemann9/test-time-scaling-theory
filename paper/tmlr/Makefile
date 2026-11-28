TECTONIC ?= tectonic
MAIN = main_tmlr
PREPRINT = main_tmlr_preprint

.PHONY: all anonymous preprint clean

all: anonymous preprint

anonymous: $(MAIN).pdf

preprint: $(PREPRINT).pdf

$(MAIN).pdf: $(MAIN).tex references_tmlr.bib tmlr.sty tmlr.bst
	$(TECTONIC) -X compile $(MAIN).tex

$(PREPRINT).pdf: $(PREPRINT).tex $(MAIN).tex references_tmlr.bib tmlr.sty tmlr.bst
	$(TECTONIC) -X compile $(PREPRINT).tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.pdf
