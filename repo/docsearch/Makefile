
cermine_files := data/cermine/cermine-* data/cermine/classes/

setup: dependancy cermine
	pmk update

dependancy:
	pip3 install --user beautifulsoup4 nltk
	#sudo apt-get install poppler-utils
	echo "Who will need the program called \`pdftotext' in the package \`poppler-utils' (debian)."
	python3 -c 'import nltk; nltk.download("stopwords")'


cermine: $(cermine_files)

$(cermine_files):
	wget https://github.com/dtrckd/CERMINE/raw/master/build/cermine.zip
	mkdir -p data/lib/
	unzip cermine.zip -d data/lib/cermine/
	rm cermine.zip



