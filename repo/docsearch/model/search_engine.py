from pymake.util.vocabulary import Vocabulary
from pymake.util.utils import hash_objects
from pymake.index.indexmanager import IndexManager
from pymake import get_pymake_settings


import os
import re
import subprocess
import whoosh as ws


def glob_path(path):
    if path.startswith('/home/'):
        # allow index sharing between machine
        pp = path.split('/')
        pp[2] = os.getenv('USER')
        path = '/'.join(pp)
    return path

def extract_pdf(pdf_file, page_limit=None):
    pdf_file = glob_path(pdf_file)
    try:
        import textract_ # longer than pdftotext
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

    _DATA_PATH = os.path.join(get_pymake_settings('project_data'), 'tfidf')

    _SCHEMA   = {'document' : ws.fields.Schema(hash      = ws.fields.ID(stored = True, unique=True),
                                               shortpath = ws.fields.ID(stored = True, unique=True),
                                               fullpath  = ws.fields.ID(stored = True, unique=True),
                                               title     = ws.fields.KEYWORD(stored = True),
                                               authors   = ws.fields.KEYWORD(stored = True), # names of the authors '||' separated
                                               references = ws.fields.KEYWORD(stored = True), # names of the references '||' separated
                                               date  = ws.fields.KEYWORD(stored = True), # date of publication (@todo: find it by cross reference !)
                                               content = ws.fields.TEXT),
                 #source  = '', # name of the journal/conf ertc
                 #type = '', # journal/conf etc
                }

    def __init__(self, expe):
        self.expe = expe
        super().__init__(default_index='document')

    def doc_yielder(self, path):
        ''' find all pdf and yield do2bow doc '''

        path = os.path.expanduser(path)

        if os.path.isfile(path):
            self.expe.path = path.rpartition('/')[0] +'/'
            for p in  [path]:
                yield p
        elif not os.path.exists(path):
            self.log.error('path error: %s' % path)
            exit()

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
        filename = os.path.basename(hit['fullpath'])
        fullpath = hit['fullpath']
        shortpath = hit['shortpath']
        pwd = os.getenv('PWD')
        os.chdir(os.path.join(pwd, 'data/lib/cermine/'))
        cermine_tar_dir = 'pdf_temp/'+filename.rpartition('.')[0] + '/'
        if not os.path.exists(cermine_tar_dir):
            os.makedirs(cermine_tar_dir)
        shutil.copy(hit['fullpath'], cermine_tar_dir)


        # 1. run Cermine
        jar = 'cermine-impl-1.14-SNAPSHOT-jar-with-dependencies.jar'
        classes = 'pl.edu.icm.cermine.ContentExtractor'
        try:
            self.log.info('extracting content of: %s' % (shortpath))
            output = subprocess.check_output(['java', '-cp', jar, classes, '-path', cermine_tar_dir])
        except Exception as e:
            self.log.error('Cermine Error %s : ' % e)
            self.log.error('Please try install/upgrade Cermine for pdf data extraction.')
            os.remove(cermine_tar_dir + filename) # remove the copied pdf
            os.chdir(pwd)
            return {}

        # 2. get the xml information
        cermine_file = cermine_tar_dir+ filename.rpartition('.')[0] + '.cermxml'
        if not os.path.isfile(cermine_file):
            self.log.error('Cermine failed...')
            return {}
        xml_strings = open(cermine_file).read()

        os.remove(cermine_tar_dir + filename) # remove the copied pdf
        os.chdir(pwd)
        return xml_strings

    # Two assumptions :
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

        try:
            soup = BeautifulSoup(xml_strings, 'lxml')
        except Exception as e:
            self.log.error('BeautifulSoup fail to parse a file:  %s : ' % e)
            return {}

        #titles = soup.findAll(re.compile(".*title.*"))

        # Main title
        # max probable title from cermine
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

        # Institution, Journal, Year etc...
        pass

        # References
        references = [ ' '.join(str(r).split()) for r in soup.findAll('mixed-citation')]
        structured['references'] = ' || '.join(references)

        return structured


    def fit(self):
        voca = Vocabulary(exclude_stopwords=True)
        writer = self.get_writer(reset=self.expe.reset, online=True)
        setattr(self, 'writer', writer)

        for _it, path in enumerate(self.doc_yielder(self.expe.path)):

            fullpath = path
            shortpath = '/' +  fullpath[len(os.path.expanduser(self.expe.path)):].rstrip('/').lstrip('/')

            is_known = False
            is_duplicated = False

            if self.getfirst(shortpath, 'shortpath'):
                # don't update document
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
                    # don't update document
                    self.log.warning("Duplicate file detected: %s renaming to %s" % (first_m['shortpath'], shortpath))
                    first_m['shortpath'] = shortpath
                    writer.update_document(**first_m)
                    is_duplicated = True
                else:
                    if self.expe.extract_structure:
                        # structured content
                        structured = self.extract_structured_kw(doc)
                        doc.update(structured)

            if not (is_known or is_duplicated):
                print("indexing `%s'" % (path))
                try:
                    writer.add_document(**doc)
                except Exception as e:
                    print('indexing doc %s failed!' % fullpath)

        return

    def close(self):
        if hasattr(self, 'writer'):
            try:
                self.writer.close()
            except Exception as e:
                print('Whoosh error: %s' %e)



