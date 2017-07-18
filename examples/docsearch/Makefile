
cermine_files := data/cermine/cermine-* data/cermine/classes/

setup: dependancy cermine
	pymake update

dependancy:
	pip3 install --user beautifulsoup4

cermine: $(cermine_files)

$(cermine_files):
	wget https://github.com/dtrckd/CERMINE/raw/master/build/cermine.zip
	mkdir -p data
	unzip cermine.zip -d data/cermine/
	rm cermine.zip



