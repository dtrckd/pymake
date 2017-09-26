from pymake.util.vocabulary import Vocabulary
from pymake.util.utils import get_global_settings, hash_objects
from pymake.index.indexmanager import IndexManager


import subprocess
import os
import whoosh as ws

def extract_pdf(pdf_file, page_limit=None):
    try:
        import textract_ # longer than pdftotext
        text = textract.process(pdf_file).decode('utf8')
    except ImportError:
        try:
            if page_limit is not None:
                text = subprocess.check_output(['pdftotext', '-f', '0', '-l', str(page_limit),  pdf_file, '-']).decode('utf8')
            else:
                text = subprocess.check_output(['pdftotext', pdf_file, '-']).decode('utf8')
        except subprocess.CalledProcessError:
            print("error : pdf read error for : %s" % (pdf_file))
            text = None

    return text

def match_pattern(text, patterns):
    ''' Return True if patterns match in text.
        (not optimized, What fuzzy solution)

        Parameters
        ----------
        pattern: str or list of str.
        text: a string

    '''
    if not patterns:
        return False

    patterns = [patterns] if type(patterns) is str else patterns
    for pattern in patterns:
        if pattern in text:
            return True
    return False


class tfidf(IndexManager):
    ''' Index documents.
        * Whoosh based.
        * format supported :
            * pdf
    '''

    _DATA_PATH = os.path.join(get_global_settings('project_data'), 'tfidf')

    _SCHEMA   = {'document' : ws.fields.Schema(hash   = ws.fields.ID(stored = True, unique=True),
                                               shortpath = ws.fields.ID(stored = True, unique=True),
                                               fullpath = ws.fields.ID(stored = True, unique=True),
                                               title  = ws.fields.KEYWORD(stored = True),
                                               authors = ws.fields.KEYWORD(stored = True), # names of the authors '||' separated
                                               references = ws.fields.KEYWORD(stored = True), # names of the references '||' separated
                                               date  = ws.fields.KEYWORD(stored = True), # date of publication (@todo: find it by cross reference !)
                                               content = ws.fields.TEXT),
                 #source  = '', # name of the journal/conf ertc
                 #type = '', # journal/conf etc
                }

    def __init__(self, expe):
        self.expe = expe
        super().__init__(default_index='document')

    def doc_yielder(self, path):
        ''' find all pdf and yield do2bow doc '''

        path = os.path.expanduser(path)

        if not os.path.exists(path):
            self.log.error('path error: %s' % path)

        for root, dirnames, filenames in os.walk(path):
            for filename in filenames:
                if not filename.endswith(('.pdf','.PDF')):
                    continue

                fullpath = os.path.join(root, filename)
                if match_pattern(fullpath, self.expe.get('exclude_path')):
                    continue

                yield fullpath


    def doc2xml(self, hit):
        import shutil

        # 0. Init cermine usage. (one path/pdf at a time).
        path = os.path.basename(hit['fullpath'])
        pwd = os.getenv('PWD')
        os.chdir(os.path.join(pwd, 'data/cermine/'))
        if not os.path.exists('pdf_temp'):
            os.makedirs('pdf_temp')
        shutil.copy(hit['fullpath'], 'pdf_temp/')

        # 1. run Cermine
        jar = 'cermine-impl-1.14-SNAPSHOT-jar-with-dependencies.jar'
        classes = 'pl.edu.icm.cermine.ContentExtractor'
        try:
            self.log.info('extracting content of: %s' % (hit['shortpath']))
            output = subprocess.check_output(['java', '-cp', jar, classes, '-path', 'pdf_temp/'])
        except Exception as e:
            self.log.error(e)
            self.log.error('Please install Cermine for pdf data extraction.')
            return {}

        # 2. get the xml information
        xml_strings = open('pdf_temp/'+ path.rpartition('.')[0] + '.cermxml').read()

        os.remove('pdf_temp/' + path) # remove the copied pdf
        os.chdir(pwd)
        return xml_strings

    # Two assumptions :
    #    * string is a pdf,
    #    * is a structured is as as scientific paper (journal ?).
    def extract_structured_kw(self, hit):
        structured = {}

        xml_strings = self.doc2xml(hit)

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            self.log.error('Please install BeautifulSoup4 to parse xml doc.')
            return {}
        import re
        soup = BeautifulSoup(xml_strings, 'lxml')

        #titles = soup.findAll(re.compile(".*title.*"))

        # Main title
        # max probable title from cermine
        front = soup.front
        front_titles = front.findAll(re.compile(".*title.*"))
        #print(front_titles)
        main_title = ' '.join([o.string or '' for o in front_titles]).strip()
        structured['title'] = main_title

        authors = soup.findAll(attrs={'contrib-type':'author'})
        authors = [o.findAll('string-name') for o in authors]
        authors = sum(authors, [])
        authors = ' || '.join([o.string for o in authors])
        structured['authors'] = authors

        # Institution, Journal, Year etc...
        pass

        # References
        references = [ ' '.join(str(r).split()) for r in soup.findAll('mixed-citation')]
        structured['references'] = ' || '.join(references)

        return structured


    def fit(self):
        voca = Vocabulary(exclude_stopwords=True)
        writer = self.get_writer(reset=self.expe.reset, online=True)
        setattr(self, 'writer', writer)

        for _it, path in enumerate(self.doc_yielder(self.expe.path)):

            fullpath = path
            shortpath = fullpath[len(os.path.expanduser(self.expe.path)):]

            is_known = False
            is_duplicated = False

            if self.getfirst(shortpath, 'shortpath'):
                # don't update document
                # could compute a diff here...
                is_known = True # assume already indexed
            else:
                text = extract_pdf(fullpath)
                text = voca.remove_stopwords(text)
                #bow = voca.doc2bow(text)
                if text in (None, ''):
                    # do nothing
                    continue

                doc = dict(shortpath=shortpath, fullpath=fullpath)
                doc['content'] = text
                doc['hash'] = hash_objects(text)

                first_m = self.getfirst(doc['hash'], 'hash')
                if first_m:
                    #if not 'content' in first_m:
                    #    writer.delete_by_term('hash', doc['hash'])
                    #    continue
                    # don't update document
                    self.log.warning("Duplicate file detected: %s renaming to %s" % (first_m['shortpath'], shortpath))
                    first_m['shortpath'] = shortpath
                    writer.update_document(**first_m)
                    is_duplicated = True
                else:
                    if self.expe.extract_structure:
                        # structured content
                        structured = self.extract_structured_kw(doc)
                        doc.update(structured)

            if not (is_known or is_duplicated):
                print("indexing `%s'" % (path))
                writer.add_document(**doc)

        return

    def close(self):
        if hasattr(self, 'writer'):
            self.writer.close()


