.ONESHELL:
SHELL = /bin/bash

PACKAGE := pymake
# Assumes Python3
pip3_version := $(shell pip3 --version 2>/dev/null)

default: install_short

docs: 
	pushd doc/
	make
	pushd

install:
# Dependancy: python3-tk
ifdef pip3_version
#python3 setup.py install --user --record .$(PACKAGE).egg-info
		pip3 install --user -r requirements.txt
		python3 setup.py install --user
		@echo 'fetching stirling number lookup table'
		@mkdir temp_junk/
		@cd temp_junk/
		wget https://github.com/dtrckd/raw_data/archive/master.zip
		unzip master.zip
		@gunzip raw_data-master/stirling/* && cat raw_data-master/stirling/* > ../pymake/util/stirling.npy
		cp -v ../pymake/util/stirling.npy $(shell python3 -c "import os, pymake; print(os.path.dirname(pymake.__file__))")/util/stirling.npy 
		@cd ..
		@rm -rf temp_junk/
else
		@echo "error: please install the \`pip3' package"
		@exit 0
endif

install_short:
	python3 setup.py install --user


networks_datasets:
	pushd data/networks 
	python3 fetch_datasets.py
	pushd

clean_datasets:
	@echo ''

uninstall:
	pip3 uninstall $(PACKAGE)
	rm -vf $(HOME)/.local/bin/pymake 

build:
	python3 setup.py build

clean: clean_cython
	find -name "__pycache__" | xargs rm -rf
	find -name "*.pyc" | xargs rm -f
	find -name "*.pyd" | xargs rm -f
	find -name "*.pyo" | xargs rm -f
	#find -name "*.orig" | xargs rm -fi
	-@rm -rf build/
	-@rm -rf dist/
	-@rm -rf $(PACKAGE).egg-info/

clean_cython:

