import os, re
from string import punctuation
from pymake.util.utils import basestring


import numpy as np
import scipy as sp


class Vocabulary(object):

    recover_list = {"wa":"was", "ha":"has"}

    regex = {
        'word' : re.compile(r'[a-zA-Z0-9_\-]+'),
        'not_word' : re.compile(r'[^a-zA-Z0-9_\-]+'),
        'ponctuation' : re.compile(r'[\.\,\;\:\!\?\(\)\{\}\[\]\'\"\`]+'),
    }

    def __init__(self, exclude_stopwords=False, lemmatize=True):

        try:
            import nltk
            _NLTK_DISABLED = False
        except:
            _NLTK_DISABLED = True

        self.vocas = []        # id to word
        self.token2id = dict() # word to id
        self.docfreq = []      # id to document frequency
        self.exclude_stopwords = exclude_stopwords

        stopwords_list = []
        if exclude_stopwords:
            # Too much strict
            #with open (os.path.join(os.path.dirname(__file__), 'stopwords.txt'), "r") as _f:
            #    stopwords_list = _f.read().replace('\n', '').split()
            if not _NLTK_DISABLED:
                stopwords_list += nltk.corpus.stopwords.words('english')
            stopwords_list = set(stopwords_list)
        self.stopwords_list = stopwords_list

        if lemmatize:
            if not _NLTK_DISABLED:
                self.wlemm = nltk.WordNetLemmatizer()
            else:
                print ('Warning: no lemmatizer !')


    def is_stopword(self, w):
        return w in self.stopwords_list

    def lemmatize(self, w0):
        if not hasattr(self, 'wlemm'):
            #self.log.debug()
            print('No lematization')
            return w0
        w = self.wlemm.lemmatize(w0.lower())
        if w in self.recover_list: return self.recover_list[w]
        return w

    def token2id(self):
        return self.token2id

    def id2token(self):
        if hasattr(self, '_id2token') and len(self.token2id) == len(self._id2token):
            return self._id2token
        else:
            self._id2token = dict((v,k) for k, v in self.token2id.iteritems())
            return self._id2token

    def term_to_id(self, term0):
        term = self.lemmatize(term0)
        if not self.regex['word'].match(term): return None
        if self.exclude_stopwords and self.is_stopword(term): return None
        try:
            term_id = self.token2id[term]
        except:
            term_id = len(self.vocas)
            self.token2id[term] = term_id
            self.vocas.append(term)
            self.docfreq.append(0)
        return term_id

    def remove_stopwords(self, doc):
        if isinstance(doc, basestring):
            doc = doc.split()
        elif doc is None:
            return None
        else:
            doc = doc
        return ' '.join([w for w in doc if not self.is_stopword(w)]).strip()

    # Bag of words !
    def doc2bow(self, doc):
        l = dict()
        words = dict()
        doc = self.regex['ponctuation'].sub(' ', doc).split() if isinstance(doc, basestring) else doc
        for term in doc:
            id = self.term_to_id(term)
            if id != None:
                l[id] = l.get(id, 0) + 1
                if not id in words:
                    words[id] = 1
                    self.docfreq[id] += 1 # It counts in how many documents a word appears. If it appears in only a few, remove it from the vocabulary using cut_low_freq()
        if "close" in dir(doc): doc.close()
        return sorted(l.items())

    def cut_low_freq(self, corpus, threshold=1):
        new_vocas = []
        new_docfreq = []
        self.token2id = dict()
        conv_map = dict()
        for id, term in enumerate(self.vocas):
            freq = self.docfreq[id]
            if freq > threshold:
                new_id = len(new_vocas)
                self.token2id[term] = new_id
                new_vocas.append(term)
                new_docfreq.append(freq)
                conv_map[id] = new_id
        self.vocas = new_vocas
        self.docfreq = new_docfreq

        def conv(doc):
            new_doc = []
            for id in doc:
                if id in conv_map: new_doc.append(conv_map[id])
            return new_doc
        return [conv(doc) for doc in corpus]

    def __getitem__(self, v):
        return self.vocas[v]

    def size(self):
        return len(self.vocas)

    def is_stopword_id(self, id):
        return self.vocas[id] in self.stopwords_list

    # if string in input
    #   * return a corpus as list of list of string.
    # Else
    #   * return a numpy array of cooc-matrix
    @classmethod
    def parse_corpus(cls, fname, bdir=""):
        dico = None
        if type(fname) is str:
            if not os.path.exists(fname): raise EnvironmentError('%s ?' % fname)
            if os.path.isdir(fname):
                import fnmatch
                bdir = fname
                corpus_files = []
                dico_files = []
                for root, dirnames, filenames in os.walk(bdir):
                    for filename in fnmatch.filter(filenames, '*.txt'):
                        if filename.startswith(('dico.','vocab.')):
                            dico_files.append(os.path.join(root, filename))
                        else:
                            corpus_files.append(os.path.join(root, filename))

                if len(corpus_files) == 1:
                    # Parse sparse matrix:  DOC_ID WORD_ID COUNT
                    with open(corpus_files[0],'r') as f:
                        # Read the header information
                        n_instances = int(f.readline())
                        n_features = int(f.readline())
                        n_nnz = int(f.readline())

                        # Create cooc-matrix
                        #data = sp.sparse.csr_matrix((n_instances, n_features))
                        data = sp.sparse.lil_matrix((n_instances, n_features), dtype=int)
                        for line in f:
                            doc_id, word_id, count = list(map(int, line.split()))
                            data[doc_id-1, word_id-1] = count
                        data = data.tocsr()
                    # If dictionnary, return it
                    with open(dico_files[0],'r') as f:
                        dico = {}
                        for i, line in enumerate(f):
                            word = line.split()
                            assert(len(word) == 1)
                            #dict((v,k) for k, v in self.token2id.iteritems())
                            dico[i] = word[0]
                else:
                    # Parse documents as each of them is a .txt file
                    data = [cls.parse_document_f(f) for f in corpus_files]

            #elif os.path.isfile(fname): ?
        elif type(fname) is list:
            # List of string as document
            docs = fname
            data = [cls.parse_document_l(d) for d in docs]
        else:
            raise NotImplementedError('file input: %s' % fname)

        return data, dico

    @classmethod
    def parse_document_f(cls, d):
        return [word.lower() for line in open(d).readlines() for word in cls.regex['not_word'].sub(' ', line).split()]

    @classmethod
    def parse_document_l(cls, d):
        return [word.lower() for word in cls.regex['not_word'].sub(' ', d).split()]



