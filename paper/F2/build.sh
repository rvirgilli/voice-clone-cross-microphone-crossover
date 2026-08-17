#!/bin/bash
# Build with the vendored ICASSP style: spconf.sty and IEEEbib.bst live in ../
set -e
cd "$(dirname "$0")"
export TEXINPUTS=..: BSTINPUTS=..:
pdflatex -interaction=nonstopmode main.tex >/dev/null
bibtex main >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null
echo "pages: $(pdfinfo main.pdf | awk '/Pages/{print $2}')"
