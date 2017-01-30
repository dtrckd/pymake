
docs: 
	@echo 'todo'

networks_datasets:
	pushd data/networks 
	python3 fetch_datasets.py
	pushd

clean_datasets:
	@echo 'toto'

install:
	python3 setup.py install --user

unistall:
	pip3 uninstall bhp 

build:
	python3 setup.py build

clean: clean_cython
	find -name "__pycache__" | xargs rm -rf
	find -name "*.pyc" | xargs rm -f
	find -name "*.pyd" | xargs rm -f
	find -name "*.pyo" | xargs rm -f
	find -name "*.orig" | xargs rm -i
	-@rm -rf build/
	-@rm -rf dist/
	-@rm -rf bhp.egg-info/

