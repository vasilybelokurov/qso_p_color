# Build the unified method note. Figures must exist first:
#     make figures
# which runs scripts/method_unified/fig*.py and writes plots/method_unified/*.png.

TEX  := method_unified.tex
PDF  := method_unified.pdf
FIGS := $(wildcard ../../plots/method_unified/*.png)

.PHONY: all clean figures

all: $(PDF)

$(PDF): $(TEX) $(FIGS)
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(TEX)

figures:
	cd ../.. && for f in scripts/method_unified/fig*.py; do python $$f || exit 1; done

clean:
	latexmk -C
