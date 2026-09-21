# Build the method note. Figures must exist first:
#     python scripts/make_method_figures.py
# which writes plots/method/*.png (tracked in git, so a reader gets the figures).

TEX  := method.tex
PDF  := method.pdf
# Figure 9 lives under plots/examples/ and is made by a different script, so a
# wildcard over plots/method/ alone silently stops the PDF rebuilding when it
# changes.
FIGS := $(wildcard ../../plots/method/*.png) $(wildcard ../../plots/examples/*.png) $(wildcard ../../plots/validation/*.png)

.PHONY: all clean figures watch

all: $(PDF)

$(PDF): $(TEX) $(FIGS)
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(TEX)

figures:
	cd ../.. && python scripts/make_method_figures.py
	cd ../.. && python scripts/score_examples.py

watch:
	latexmk -pdf -pvc -interaction=nonstopmode $(TEX)

clean:
	latexmk -C
	rm -f *.aux *.log *.out *.fls *.fdb_latexmk *.bbl *.blg
