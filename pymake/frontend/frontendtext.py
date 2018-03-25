import sys, os
from collections import defaultdict
from string import Template

from .frontend import DataBase
from pymake.util.vocabulary import Vocabulary



class frontendText(DataBase):
    """ Frontend for text Corpus """

    def __init__(self, expe=None):
        super(frontendText, self).__init__(expe)
        self._data_type = 'text'

    def load_data(self, randomize=False):
        """ Load data according to different scheme:
            * Corpus from file dataset
            * Corpus from random generator
            """

        corpus_name = self.corpus_name
        self.get_corpus(corpus_name)

        # @DEBUG
        if str(self.N).isdigit() and int(self.N) > self.data.shape[0]:
            raise ValueError('Sampling of corpus %s too big (-n options)' % self.N)

        if randomize:
            self.shuffle_docs()
        return self.data

    def make_testset(self, ratio):
        self.log.warning('check WHY and WHEN overflow in stirling matrix !?')
        self.log.warning('debug why error and i get walue superior to 6000 in the striling matrix ????')
        D = self.data.shape[0]
        d = int(D * ratio)
        data = self.data[:d]
        data_t = self.data[d:]
        return data, data_t

    # @Debug: remove docu for wich count > 6000 because of stirling matrix !
    def sample(self, N=None, **args):
        N = N or self.N
        n = self.data.shape[0]
        if not N or N == 'all':
            self.N = 'all'
            # To remove !
            if  self.corpus_name == '20ngroups':
                data = self.data[:10000]
                empty_words =  np.where(data.sum(0).A[0] == 0)[0]
                new_cols = np.delete(np.arange(data.shape[1]), empty_words)
                self.data = data[:, new_cols]

        else:
            N = int(N)
            data = self.data[:N]
            # Here we come to streaming problem !
            # @DEBUG manage id2word
            if type(data) is np.array:
                empty_words =  np.where(data.sum(0) == 0)[0]
                self.data = np.delete(data, empty_words, axis=1)
            elif data.format == 'csr':
                empty_words =  np.where(data.sum(0).A[0] == 0)[0]
                new_cols = np.delete(np.arange(data.shape[1]), empty_words)
                self.data = data[:, new_cols]

        # @debug to remove
        _l =  (self.data >= 6000).sum(1).A.T[0]
        print(_l)
        tt = self.data[_l > 0]
        for t in tt:
            print(t[t>=6000])
        self.data = self.data[ _l == 0 ]
        return self.data

    ### Get and prepropress text
    #   See Vocabulary class...
    #   * Tokenisation from scratch
    #   * Stop Word from scratch
    #   * Lemmatization from Wornet
    #   * Load or Save in a Gensim context
    #       - Load has priority over Save
    # @Debug: There is a convertion to gensim corpus to use the serialization library and then back to scipy corpus.
    #   Can be avoided by using our own library of serialization, using Gensim if needed ?!
    def textloader(self, target, bdir=None, corpus_name="", n=None):
        if type(target) is str and os.path.isfile(target):
            bdir = os.path.dirname(target)
        elif bdir is None:
            bdir = self.basedir
        fn = 'corpus'
        if n:
            fn += str(n)
        elif type(target) is not str:
            n = len(target)
            fn += str(n)

        if corpus_name:
            fname = bdir + '/'+fn+'_' + corpus_name + '.mm'
        else:
            fname = bdir + '/'+fn+'.mm'

        if self._load_data and os.path.isfile(fname):
            data = gensim.corpora.MmCorpus(fname)
            data = gensim.matutils.corpus2csc(data, dtype=int).T
            id2word = dict(gensim.corpora.dictionary.Dictionary.load_from_text(fname + '.dico'))
        else:
            prin('re-Building Corpus...')
            raw_data, id2word = Vocabulary.parse_corpus(target)

            # Corpus will be in bag of words format !
            if type(raw_data) is list:
                voca = Vocabulary(exclude_stopwords=True)
                data = [voca.doc2bow(doc) for doc in raw_data]
                data = gensim.matutils.corpus2csc(data, dtype=int).T # Would be faster with #doc #term #nnz
            else:
                data = raw_data

            if self._save_data:
                make_path(bdir)
                _data = gensim.matutils.Sparse2Corpus(data, documents_columns=False)
                voca_gensim = gensim.corpora.dictionary.Dictionary.from_corpus(_data, id2word)
                voca_gensim.save_as_text(fname+'.dico')
                gensim.corpora.MmCorpus.serialize(fname=fname, corpus=_data)
                #@Debug how to get the corpus from list of list ?
                #_data = gensim.corpora.MmCorpus(fname)

        return data, id2word

    def get_corpus(self, corpus_name):
        self.make_io_path()
        bdir = self.basedir
        data_t = None
        data_t = None
        if corpus_name == 'random':
            # mmmh speak !
            data = self.random()
        if corpus_name == 'lucene':
            raise NotImplementedError
            #searcher = warm_se(config)
            #q = config.get('q'); q['limit'] = config['limit_train']
            #id2word = searcher.get_id2word()
            #corpus = searcher.self.parse_corpus(q, vsm=config['vsm'], chunk=1000, batch=True)
        elif corpus_name == '20ngroups_sklearn':
            from sklearn.datasets import fetch_20newsgroups
            ngroup_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=None)
            ngroup_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=None)
            train_data = ngroup_train.data
            test_data = ngroup_test.data
            #corpus, id2word = self.textloader(train_data, bdir=bdir, corpus_name='train', n=config.get('N'))
            corpus, id2word = self.textloader(train_data, bdir=bdir, corpus_name='train')
            corpus_t, id2word_t = self.textloader(test_data, bdir=bdir, corpus_name='test')

            K = self.K
            #################
            ### Group Control
            test_classes = ngroup_test.target
            train_classes = ngroup_train.target
            if K == 6 and len(ngroup_test.target_names) != 6:
                # Wrap to subgroups
                target_names = ['comp', 'misc', 'rec', 'sci', 'talk', 'soc']
                map_ = dict([(0,5), (1,0), (2,0), (3,0), (4,0), (5,0), (6,1), (7,2), (8,2), (9,2), (10,2), (11,3), (12,3), (13,3), (14,3), (15,5), (16,4), (17,4), (18,4), (19,5)])
                test_classes = set_v_to(test_classes, map_)
                train_classes = set_v_to(train_classes, map_)
            else:
                target_names = ngroup_test.target_names
            C = len(target_names)

        elif corpus_name == 'wikipedia':
            # ? file type
            # Create
            command = './gensim/gensim/scripts/make_wikicorpus_ml.py '
            command += '/work/data/wikipedia/enwiki-latest-pages-articles.xml.bz2 ../pymake/data/wikipedia/wiki_en'
            os.system(command)
            # Load
            error = 'Load Wikipedia corpus'
            raise NotImplementedError(error)
        elif corpus_name == 'odp':
            # SVMlight file type
            from sklearn.datasets import load_svmlight_files, load_svmlight_file
            fn_train = os.path.join(bdir, 'train.txt')
            fn_test = os.path.join(bdir, 'test.txt')
            # More feature in test than train !!!
            data, train_classes = load_svmlight_file(fn_train)
            data_t, test_classes = load_svmlight_file(fn_test)
            id2word = None
        elif corpus_name in ('reuter50', 'nips12', 'nips', 'enron', 'kos', 'nytimes', 'pubmed') or corpus_name == '20ngroups' :
            # DOC_ID FEAT_ID COUNT file type
            data, id2word = self.textloader(bdir, corpus_name=corpus_name)
        else:
            raise ValueError('Which corpus to Load ?')

        self.data = data
        self.id2word = id2word
        if data_t is None:
            pass
            #raise NotImplementedError('Corpus test ?')
        else:
            self.data_t = data_t

        return True

    def get_data_prop(self):
        prop = defaultdict()
        prop.update( {'corpus': self.corpus_name,
                      'instances' : self.data.shape[1] })
        nnz = self.data.sum()
        _nnz = self.data.sum(axis=1)
        dct = {'features': self.data.shape[1],
               'nnz': nnz,
               'nnz_mean': _nnz.mean(),
               'nnz_var': _nnz.var(),
               'train_size': None,
               'test_size': None,
              }
        prop.update(dct)
        return prop

    def template(self, dct):
        text_templ = '''###### $corpus_name
        Building: $time minutes
        Documents: $instances
        Nnz: $nnz
        Nnz mean: $nnz_mean
        Nnz var: $nnz_var
        Vocabulary: $features
        train: $train_size
        test: $test_size
        \n'''
        return Template(templ).substitute(dct)

    def print_vocab(self, data, id2word):
        if id2word:
            return gensim.corpora.dictionary.Dictionary.from_corpus(data, id2word) #; print voca

    def shuffle_docs(self):
        self.shuffle_instances()

    # Return a vector with document generated from a count matrix.
    # Assume sparse matrix
    @staticmethod
    def sparse2stream(data):
        #new_data = []
        #for d in data:
        #    new_data.append(d[d.nonzero()].A1)
        bow = []
        for doc in data:
            # Also, see collections.Counter.elements() ...
            bow.append( np.array(list(chain(* [ doc[0,i]*[i] for i in doc.nonzero()[1] ]))))
        bow = np.array(bow)
        #map(np.random.shuffle, bow)
        return bow


    # Debug
    def run_lda(self):
        pass
   #     # Cross Validation settings...
   #     #@DEBUG: do we need to remake the vocabulary ??? id2word would impact the topic word distribution ?
   #     if corpus_t is None:
   #         pass
   #         #take 80-20 %
   #         # remake vocab and shape !!!
   #         # manage downside
   #     try:
   #         total_corpus = len(corpus)
   #         total_corpus_t = len(corpus_t)
   #     except:
   #         total_corpus = corpus.shape[0]
   #         total_corpus_t = corpus.shape[0]
   #     if config.get('N'):
   #         N = config['N']
   #     else:
   #         N = total_corpus
   #     corpus = corpus[:N]
   #     n_percent = float(N) / total_corpus
   #     n_percent = int(n_percent * total_corpus_t) or 10
   #     heldout_corpus = corpus_t[:n_percent]

   #     ############
   #     ### Load LDA
   #     load = config['load_model']
   #     # Path for LDA model!
   #     bdir = '../PyNPB/data/'
   #     bdir = os.path.join(bdir,config.get('corpus'), config.get('bdir', ''))
   #     lda = lda_gensim(corpus, id2word=id2word, K=K, bdir=bdir, load=load, model=config['model'], alpha=config['hyper'], n=config['N'], heldout_corpus=heldout_corpus)
   #     lda.inference_time = datetime.now() - last_d
   #     last_d = ellapsed_time('LDA Inference -- '+config['model'], last_d)

   #     ##############
   #     ### Log Output
   #     lda.print_topics(K)

   #     ##############
   #     ### Prediction
   #     corpus_t = corpus
   #     if config['predict'] and true_classes is not None and C == K:
   #         true_classes = train_classes
   #         predict_class = []
   #         confusion_mat = np.zeros((K,C))
   #         startt = datetime.now()
   #         for i, d in enumerate(corpus_t):
   #             d_t = lda.get_document_topics(d, minimum_probability=0.01)
   #             t = max(d_t, key=lambda item:item[1])[0]
   #             predict_class.append(t)
   #             c = true_classes[i]
   #             confusion_mat[t, c] += 1
   #         last_d = ellapsed_time('LDA Prediction', startt)
   #         predict_class = np.array(predict_class)
   #         lda.confusion_matrix = confusion_mat

   #         map_kc = map_class2cluster_from_confusion(confusion_mat)
   #         #new_predict_class = set_v_to(predict_class, dict(map_kc))

   #         print "Confusion Matrix, KxC:"
   #         print confusion_mat
   #         print map_kc
   #         print [(k, target_names[c]) for k,c in map_kc]

   #         purity = confusion_mat.max(axis=1).sum() / len(corpus_t)
   #         print 'Purity (K=%s, C=%s, D=%s): %s' % (K, C, len(corpus_t), purity)

   #         #precision = np.sum(new_predict_class == true_classes) / float(len(predict_class)) # equal !!!
   #         precision = np.sum(confusion_mat[zip(*map_kc)]) / float(len(corpus_t))
   #         print 'Ratio Groups Control: %s' % (precision)

   #     if save:
   #         ## Too big
   #         lda.expElogbeta = None
   #         lda.sstats = None
   #         lda.save(lda.fname)

   #     if config.get('_verbose'):
   #         #print lda.top_topics(corpus)
   #         for d in corpus:
   #             print lda.get_document_topics(d, minimum_probability=0.01)

   #     print lda
   #     if type(corpus) is not list:
   #         print corpus
   #         print corpus_t
   #     self.print_vocab(corpus, id2word)


