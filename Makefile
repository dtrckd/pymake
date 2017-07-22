PACKAGE := pymake
# Assumes Python3
pip3_version := $(shell pip3 --version 2>/dev/null)

default: install

docs: 
	pushd doc/
	make
	pushd

install:
ifdef pip3_version
#python3 setup.py install --user --record .$(PACKAGE).egg-info
		pip3 install --user -r requirements.txt
		python3 setup.py install --user
else
		@echo "error: please install the \`pip3' package"
		@exit 0
endif


networks_datasets:
	pushd data/networks 
	python3 fetch_datasets.py
	pushd

clean_datasets:
	@echo ''

uninstall:
	# Do not remove empty dir...
	#cat .$(PACKAGE).egg-info | xargs rm -fv
	pip3 uninstall $(PACKAGE)
	# Get the PATH, from globvar ?
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

