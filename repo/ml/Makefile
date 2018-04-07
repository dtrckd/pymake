.ONESHELL:
SHELL = /bin/bash

default:  install

install: fetch_data
	pip3 install --user -r requirements.txt


fetch_data: net_data text_data lut

net_data:
	pushd data/
	mkdir -p .pmk/
	python3 fetch_datasets.py
	pushd

text_data:


lut:
	@echo 'fetching stirling number lookup table'
	@mkdir temp_junk/
	@cd temp_junk/
	wget https://github.com/dtrckd/raw_data/archive/master.zip
	unzip master.zip
	@mkdir -p ../data/lut/
	@gunzip raw_data-master/stirling/* && cat raw_data-master/stirling/* > ../data/lut/stirling.npy
	#cp -v data/lut/stirling.npy $(shell python3 -c "import os, pymake; print(os.path.dirname(pymake.__file__))")/util/stirling.npy 
	@cd ..
	@rm -rf temp_junk/



clean_data:
