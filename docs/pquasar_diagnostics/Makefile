# Build the p_quasar diagnostics note. Figures first:
#     cd ../.. && nice python scripts/method_unified/pquasar_diagnostics.py --rescore
TEX  := pquasar_diagnostics.tex
PDF  := pquasar_diagnostics.pdf
FIGS := $(wildcard ../../plots/pquasar_diagnostics/*.png)

.PHONY: all clean
all: $(PDF)
$(PDF): $(TEX) $(FIGS)
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(TEX)
clean:
	latexmk -C
