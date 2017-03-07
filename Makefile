PACKAGE := pymake

default: install
docs: 
	pushd doc/
	make
	pushd

networks_datasets:
	pushd data/networks 
	python3 fetch_datasets.py
	pushd

clean_datasets:
	@echo 'toto'

install:
	#python3 setup.py install --user --record .$(PACKAGE).egg-info
	python3 setup.py install --user

uninstall:
	# Do not remove empty dir...
	#cat .$(PACKAGE).egg-info | xargs rm -fv
	pip3 uninstall $(PACKAGE)

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

