.ONESHELL:
SHELL = /bin/bash

default: 

fetch_data: net_data text_data lut

net_data:
	pushd data/networks 
	python3 fetch_datasets.py
	pushd

text_data:


lut:
	@echo 'fetching stirling number lookup table'
	@mkdir temp_junk/
	@cd temp_junk/
	wget https://github.com/dtrckd/raw_data/archive/master.zip
	unzip master.zip
	@gunzip raw_data-master/stirling/* && cat raw_data-master/stirling/* > ../pymake/util/stirling.npy
	cp -v ../pymake/util/stirling.npy $(shell python3 -c "import os, pymake; print(os.path.dirname(pymake.__file__))")/util/stirling.npy 
	@cd ..
	@rm -rf temp_junk/



clean_data:
