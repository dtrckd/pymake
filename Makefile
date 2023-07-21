.ONESHELL:
SHELL = /bin/bash

PACKAGE := pmk
# Assumes Python3
pip3_version := $(shell pip3 --version 2>/dev/null)

.PHONY: doc

default: install_short

doc:
	#pushd doc/
	#make
	#pushd
	pdoc -f --html --output-dir doc pymake

install:
	# Dependancy: python3-tk
ifdef pip3_version
	#python3 setup.py install --user --record .$(PACKAGE).egg-info
	pip3 install --user -r requirements.txt
	python3 setup.py install --user
else
	@echo "error: please install the \`pip3' package"
	@exit 0
endif

install_short:
	python3 setup.py install --user

push_pip:
	rm -rf dist/
	python setup.py sdist
	twine upload dist/*

push_doc: doc
	cd doc/pymake
	export NEOCITIES_KEY=$(shell cat ${HOME}/src/config/credentials/pymake.neocities)
	neocities upload *
	cd -

uninstall:
	pip3 uninstall $(PACKAGE)
	rm -vf $(HOME)/.local/bin/$(PACKAGE)
	rm -f $(HOME)/.bash_completion.d/pymake_completion

build:
	python3 setup.py build

test:
	pushd tests
	DISPLAY= python3 functest.py
	popd

msa:
	ditaa wiki/msa.md wiki/msa.png


clean: clean_cython
	find -name "__pycache__" | xargs rm -rf
	find -name "*.pyc" | xargs rm -f
	find -name "*.pyd" | xargs rm -f
	find -name "*.pyo" | xargs rm -f
	#find -name "*.orig" | xargs rm -fi
	-@rm -rf build/
	-@rm -rf dist/
	-@rm -rf *.egg-info/

clean_cython:

