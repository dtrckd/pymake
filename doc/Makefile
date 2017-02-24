.ONESHELL:
SHELL = /bin/bash
.PHONY : graph

# @todo : docformat
html:
	epydoc ../pymake --graph all

Engine='circo dot fdp neato nop nop1 nop2 osage patchwork sfdp twopi'
graph:  
	@mkdir -p graph/
	@pushd graph/
	pyreverse ../../pymake -A -S -p pymake
	# convert .dot .png for all .dot...
	#dot -Tpng -K@> > graph/...
	@popd

clean:
	-@rm -rf graph/*
	-@rm -rf html/*
