# Build the method note. Figures must exist first:
#     python scripts/make_method_figures.py
# which writes plots/method/*.png (tracked in git, so a reader gets the figures).

TEX  := method.tex
PDF  := method.pdf
FIGS := $(wildcard ../../plots/method/*.png)

.PHONY: all clean figures watch

all: $(PDF)

$(PDF): $(TEX) $(FIGS)
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(TEX)

figures:
	cd ../.. && python scripts/make_method_figures.py

watch:
	latexmk -pdf -pvc -interaction=nonstopmode $(TEX)

clean:
	latexmk -C
	rm -f *.aux *.log *.out *.fls *.fdb_latexmk *.bbl *.blg
