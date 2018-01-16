.ONESHELL:
SHELL = /bin/bash

pap_base := ../../../papers/personal/relational_models/property_v0/networks
md_fix := img/fix/*.md
md_files := figs/*.md
image_files := $(wildcard figs/*.png)
latex_files := $(wildcard figs/*.tex)
TARGET := long_version/expe_dev.tex

default: paper

paper: figs
	pushd $(pap_base)
	pdflatex $(TARGET)
	popd

figs: latex move

latex: $(md_files) $(pap_base)/$(md_fix)
	$(foreach f, $^, pandoc $f -o $(f:.md=.tex) --filter ./texfilter.py ;)

move: $(latex_files) $(image_files)
	@cp -v $^ $(pap_base)/img/corpus/

clean:
	@rm -vf  figs/*
