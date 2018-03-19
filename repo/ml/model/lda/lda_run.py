#!/usr/bin/env python
# -*- coding: utf-8 -*-

from util.utils import argParse
from util.vocabulary import Vocabulary, parse_corpus
from frontend.frontend import frontEndBase

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import scipy as sp
#np.set_printoptions(threshold='nan')

import os
import logging
import os.path

sys.path.insert(1, './gensim')
import gensim as gsm
from gensim.models import ldamodel, ldafullbaye


Models = { 'ldamodel': ldamodel, 'ldafullbaye': ldafullbaye }

########################################################################"
### LDA Worker
def lda_gensim(corpus=None, id2word=None, K=10, alpha='auto', save=False, bdir='tmp/', model='ldamodel', load=False, n=None, heldout_corpus=None, updatetype='batch'):
    try: n = len(corpus) if corpus is not None else n
    except: n = corpus.shape[0]
    fname = bdir + "/%s_%s_%s_%s.gensim" % ( model, str(K), alpha, n)
    if load:
        return Models[model].LdaModel.load(fname)

    if hasattr(corpus, 'tocsc'):
        # is csr sparse matrix
        corpus = corpus.tocsc()
        corpus = gsm.matutils.Sparse2Corpus(corpus, documents_columns=False)
        if heldout_corpus is not None:
            heldout_corpus = heldout_corpus.tocsc()
            heldout_corpus = gsm.matutils.Sparse2Corpus(heldout_corpus, documents_columns=False)
    elif isanparray:
        # up tocsc ??!!! no !
        dense2corpus
    # Passes is the iterations for batch onlines and iteration the max it in the gamma treshold test loop
    # Batch setting !
    if updatetype == 'batch':
        lda = Models[model].LdaModel(corpus, id2word=id2word, num_topics=K, alpha=alpha,
                                     iterations=100, eval_every=None, update_every=None, passes=50, chunksize=200000, fname=fname, heldout_corpus=heldout_corpus)
    elif updatetype == 'online':
        lda = Models[model].LdaModel(corpus, id2word=id2word, num_topics=K, alpha=alpha,
                                     iterations=100, eval_every=None, update_every=1, passes=1, chunksize=2000, fname=fname, heldout_corpus=heldout_corpus)

    if save:
        lda.expElogbeta = None
        lda.sstats = None
        lda.save(fname)
    return lda

##################
###### MAIN ######
##################
if __name__ == '__main__':
    conf = defaultdict(lambda: False, dict(
        _verbose     = 0,
        host        = 'localhost',
        index       = 'search',
        corpus      = '20ngroups', # nips12, wiki, lucene
        vsm         = 'tf',
        extra_feat  = False,
        limit_train = 10000,
        limit_predict = None,
        chunk = 10000,
        model = 'ldafullbaye',
        K = 10,
        alpha = 'auto',
        load_corpus = True,
        save_corpus = True,
        load_lda = False,
        save_lda = True,
        predict = False,
    ))
    conf.update(argParse(_USAGE))

    K = conf['K']
    if conf.get('simulate'):
        print '''--- Simulation settings ---
        Model : %s
        Target : %s
        K: %s
        alpha: %s
        Load corpus: %s
        Save corpus: %s''' % (conf['model'], conf['corpus'], conf['K'], conf['alpha'], conf['load_corpus'], conf['save_corpus'])
        exit()

    ###############
    ### Load Corpus
    corpus_name = conf.get('corpus')
    frontend = frontEndBase(**conf)

    startt = datetime.now()
    load = conf['load_corpus']
    save = conf['save_corpus']
    for corpus_name in ['nips12', '20ngroups', 'nips', 'enron', 'kos', 'nytimes', 'pubmed']:
        frontend.get_text_corpus(corpus_name, save, load, **conf)
    last_d = ellapsed_time('Prepropressing', startt)

    exit()


    # Cross Validation settings...
    #@DEBUG: do we need to remake the vocabulary ??? id2word would impact the topic word distribution ?
    if corpus_t is None:
        pass
        #take 80-20 %
        # remake vocab and shape !!!
        # manage downside
    try:
        total_corpus = len(corpus)
        total_corpus_t = len(corpus_t)
    except:
        total_corpus = corpus.shape[0]
        total_corpus_t = corpus.shape[0]
    if conf.get('N'):
        N = conf['N']
    else:
        N = total_corpus
    corpus = corpus[:N]
    n_percent = float(N) / total_corpus
    n_percent = int(n_percent * total_corpus_t) or 10
    heldout_corpus = corpus_t[:n_percent]

    ############
    ### Load LDA
    load = conf['load_lda']
    save = conf['save_lda']
    # Path for LDA model!
    bdir = '../PyNPB/data/'
    bdir = os.path.join(bdir,conf.get('corpus'), conf.get('bdir', ''))
    lda = lda_gensim(corpus, id2word=id2word, K=K, bdir=bdir, load=load, model=conf['model'], alpha=conf['alpha'], n=conf['N'], heldout_corpus=heldout_corpus)
    lda.inference_time = datetime.now() - last_d
    last_d = ellapsed_time('LDA Inference -- '+conf['model'], last_d)

    ##############
    ### Log Output
    print
    logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
    lda.print_topics(K)
    print

    ##############
    ### Prediction
    corpus_t = corpus
    if conf['predict'] and true_classes is not None and C == K:
        true_classes = train_classes
        predict_class = []
        confusion_mat = np.zeros((K,C))
        startt = datetime.now()
        for i, d in enumerate(corpus_t):
            d_t = lda.get_document_topics(d, minimum_probability=0.01)
            t = max(d_t, key=lambda item:item[1])[0]
            predict_class.append(t)
            c = true_classes[i]
            confusion_mat[t, c] += 1
        last_d = ellapsed_time('LDA Prediction', startt)
        predict_class = np.array(predict_class)
        lda.confusion_matrix = confusion_mat

        map_kc = map_class2cluster_from_confusion(confusion_mat)
        #new_predict_class = set_v_to(predict_class, dict(map_kc))

        print "Confusion Matrix, KxC:"
        print confusion_mat
        print map_kc
        print [(k, target_names[c]) for k,c in map_kc]

        purity = confusion_mat.max(axis=1).sum() / len(corpus_t)
        print 'Purity (K=%s, C=%s, D=%s): %s' % (K, C, len(corpus_t), purity)

        #precision = np.sum(new_predict_class == true_classes) / float(len(predict_class)) # equal !!!
        precision = np.sum(confusion_mat[zip(*map_kc)]) / float(len(corpus_t))
        print 'Ratio Groups Control: %s' % (precision)

    if save:
        ## Too big
        lda.expElogbeta = None
        lda.sstats = None
        lda.save(lda.fname)

    if conf.get('_verbose'):
        #print lda.top_topics(corpus)
        for d in corpus:
            print lda.get_document_topics(d, minimum_probability=0.01)

    print
    print lda
    if type(corpus) is not list:
        print corpus
        print corpus_t
    if id2word:
        voca = gsm.corpora.dictionary.Dictionary.from_corpus(corpus, id2word) #; print voca

