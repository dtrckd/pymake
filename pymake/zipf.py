#!/usr/bin/python -u
# -*- coding: utf-8 -*-

from random import choice
from util.utils import *
from util.vocabulary import Vocabulary, parse_corpus
from frontend.manager import ModelManager, FrontendManager

import numpy as np
import scipy as sp
#np.set_printoptions(threshold='nan')

import logging
import os
import os.path

_USAGE = '''zipf [-vhswp] [-k [rvalue] [-n N] [-d basedir] [-lall] [-l type] [-m model] [-c corpus] [-i iterations]

Default load corpus and run a model !!

##### Argument Options
--hyper|alpha  : hyperparameter optimization ( asymmetric | symmetric | auto)
-lall          : load all; Corpus and LDA model
-l type        : load type ( corpus | lda)
-i iterations  : Iterations number
-c corpus      : Pickup a Corpus (20ngroups | nips12 | wiki | lucene)
-m model       : Pickup a Model (ldamodel | ldafullbaye)
-n | --limit N : Limit size of corpus
-d basedir     : base directory to save results.
-k K           : Number of topics.
##### Single argument
-p           : Do prediction on test data
-s           : Simulate output
-w|-nw           : Write/noWrite convergence measures (Perplexity etc)
-h | --help  : Command Help
-v           : Verbosity level

Examples:
./zipf.py -n 1000 -k 30 --alpha fix -m immsb -c generator3 -l model --refdir debug5
./zipf.py -n 1000 -k 10 --alpha auto --homo 0 -m ibp_cgs -c generator3 -l model --refdir debug5 -nld


'''

# @Debug:
#   * get_json and get_data_prop are not consistent/aligned.

##################
###### MAIN ######
##################
if __name__ == '__main__':
    config = defaultdict(lambda: False, dict(
        ##### Global settings
        verbose                       = logging.INFO,
        ##### Models Hyperparameters
        #model                         = 'lda_cgs',
        model                         = 'ilda',
        hyper                         = 'auto',
        K                             = 3,
        N                             = 3,
        chunk                         = 10000,
        iterations                    = 2,
        ###
        homo                          = False, # learn W in IBP
        ###### I/O settings
        refdir                        = 'debug',
        load_data                   = True,
        save_data                   = True,
        load_model                    = False,
        save_model                    = False,
        write                         = False, # -w/-nw
        #####
        predict                       = False,
    ))
    config.update(argParse(_USAGE))

    lgg = setup_logger('root','%(message)s', config.get('verbose') )

    # Silly ! think different
    if config.get('lall'):
        # Load All
        config.update(load_data=True, load_model=True)
    if config.get('load') == 'corpus':
        config['load_data'] = True
    elif config.get('load') == 'model':
        config['load_model'] = True

    ############################################################
    ##### Simulation Output
    if config.get('simul'):
        print '''--- Simulation settings ---
        Model : %s
        Corpus : %s
        K : %s
        N : %s
        hyper : %s
        Output : %s''' % (config['model'], config['corpus_name'],
                         config['K'], config['N'], config['hyper'],
                         config['output_path'])
        exit()

    # Initializa Model
    frontend = FrontendManager.get(config)
    data = frontend.load_data(randomize=False)
    data = frontend.sample()
    model = ModelManager(config=config)

    N = frontend.N
    if config.get('load_model'):
        ### Generate data from a fitted model
        model = model.load()
        y, theta, phi = model.generate(N)
        #y = np.triu(y) + np.triu(y, 1).T
    else:
        ### Generate data from a un-fitted model
        alpha = .5
        gmma = 1.
        delta = .1
        model = model.model
        model.update_hyper((alpha, gmma, delta))
        y, theta, phi = model.generate(N, config['K'])

    ### Baselines
    #R = rescal(data, config['K'])
    R = None

    K = theta.shape[1]
    ###############################################################
    ### Expe Wrap up debug
    print 'model: %s, corpus: %s \
    K = %s, N =  %s'% (frontend.corpus_name, frontend.corpus_name, model.K, frontend.N)

    from plot import *

    ### Analysis On Generated Model
    community_distribution_learn, local_attach_learn, clusters_learn = model.communities_analysis(theta, y)

    degree_learn = plot_degree(y, noplot=True)
    try:
        max_c, max_cd = max(local_attach_learn.items(), key = lambda x: len(x[1]))
        max_local_attach_learn = sorted(local_attach_learn.pop(max_c), reverse=True)
        min_c, min_cd = min(local_attach_learn.items(), key = lambda x: len(x[1]))
        min_local_attach_learn = sorted(local_attach_learn.pop(min_c), reverse=True)

        rd_com_1 = choice(local_attach_learn.keys())
        rd_local_attach_learn_1 = sorted(local_attach_learn.pop(rd_com_1), reverse=True)
        rd_com_2 = choice(local_attach_learn.keys())
        rd_local_attach_learn_2 = sorted(local_attach_learn.pop(rd_com_2), reverse=True)
    except Exception:
    #except IndexError('Number of community too small for evaluation...'):
        ll = ['min_local_attach_learn', 'max_local_attach_learn', 'rd_local_attach_learn_1','rd_local_attach_learn_2']
        for v in ll:
            if not v in globals():
                 globals()[v] = []


    ### Analysis on True Data
    d = frontend.get_json()
    print 'K= %d, Precision global %s, local: %s , Rappel %s' % (K, d.get('g_precision'), d['Precision'], d['Rappel'])
    degree_source = sorted(d['degree_all'].values(), reverse=True)

    try:
        community_distribution_source, local_attach_source, clusters_source = frontend.communities_analysis()
    except:
        print 'waring: could execute: frontend.communities_analysis(); probabily self.clusters not defined'
        local_attach_source = d['Local_Attachment']
        community_distribution_source = d['Community_Distribution']
        ### In the future
        #cluster_source = d['clusters']
        clusters_source = clusters_learn

    try:
        max_c, max_cd = max(local_attach_source.items(), key = lambda x: len(x[1]))
        max_local_attach_source = sorted(local_attach_source.pop(max_c), reverse=True)
        min_c, min_cd = min(local_attach_source.items(), key = lambda x: len(x[1]))
        min_local_attach_source = sorted(local_attach_source.pop(min_c), reverse=True)

        rd_com_1 = choice(local_attach_source.keys())
        rd_local_attach_source_1 = sorted(local_attach_source.pop(rd_com_1), reverse=True)
        rd_com_2 = choice(local_attach_source.keys())
        rd_local_attach_source_2 = sorted(local_attach_source.pop(rd_com_2), reverse=True)
    except Exception:
    #except IndexError('Number of community too small for evaluation...'):
        ll = ['min_local_attach_source', 'max_local_attach_source', 'rd_local_attach_source_1','rd_local_attach_source_2', rd_com_1]
        for v in ll:
            if not v in globals():
                 globals()[v] = []


    ### Re-order adjacency matrix
    nodelist = [k[0] for k in sorted(zip(range(len(clusters_source)), clusters_source), key=lambda k: k[1])]
    data_r = data[nodelist, :][:, nodelist]

    #nodelist = [k[0] for k in sorted(zip(range(len(clusters_learn)), clusters_learn), key=lambda k: k[1])]
    y_r = y[nodelist, :][:, nodelist]

    ### Plot

    #adjshow(data, title='Data')
    #adjshow(y, title='Bayesian')

    #adjshow_l([data, y], title=['Data', 'Bayesian'])
    #adjshow_l([data_r, y_r], title=['Data reorderded', 'Bayesian reordered'])
    print data.shape, y.shape
    adjshow_4([data, y, data_r, y_r], title=['Data', 'Bayesian', 'Data reorderded', 'Bayesian reordered'])

    if R is not None:
        R_r = R[nodelist, :][:, nodelist]
        adjshow(R, title='Rescal')
        #adjshow(R_r, title='Rescal re-ordored')

    ### Degree distribution instead of Zipf plot
    #max_local_attach_source , _ = np.histogram(max_local_attach_source ,bins  = len(max_local_attach_source), density  = True)
    #max_local_attach_learn  , _ = np.histogram(max_local_attach_learn ,bins   = len(max_local_attach_learn), density   = True)
    #min_local_attach_source , _ = np.histogram(min_local_attach_source ,bins  = len(min_local_attach_source), density  = True)
    #min_local_attach_learn  , _ = np.histogram(min_local_attach_learn ,bins   = len(min_local_attach_learn), density   = True)
    #rd_local_attach_source_1, _ = np.histogram(rd_local_attach_source_1 ,bins = len(rd_local_attach_source_1), density = True)
    #rd_local_attach_learn_1 , _ = np.histogram(rd_local_attach_learn_1 ,bins  = len(rd_local_attach_learn_1), density  = True)
    #rd_local_attach_source_2, _ = np.histogram(rd_local_attach_source_2 ,bins = len(rd_local_attach_source_2), density = True)
    #rd_local_attach_learn_2 , _ = np.histogram(rd_local_attach_learn_2 ,bins  = len(rd_local_attach_learn_2), density  = True)
    #degree_source           , _ = np.histogram(degree_source ,bins            = len(degree_source), density            = True)
    #degree_learn            , _ = np.histogram(degree_learn ,bins             = len(degree_learn), density             = True)

    marker = '.'
    plt.figure()
    plt.subplot(1,2,1)
    x = np.arange(1, len(community_distribution_source)+1)
    plt.loglog(x, sorted(community_distribution_source, reverse=True), marker=marker, label='source')
    x = np.arange(1, len(community_distribution_learn)+1)
    plt.loglog(x, sorted(community_distribution_learn, reverse=True), marker=marker, label='learn')
    plt.title('Communities Distribution')
    plt.legend()
    plt.subplot(1,2,2)
    x = np.arange(1, len(degree_source)+1)
    plt.loglog(x, degree_source, marker=marker, label='source')
    x = np.arange(1, len(degree_learn)+1)
    plt.loglog(x, degree_learn, marker=marker, label='learn')
    plt.title('Global Degree Distribution')
    plt.legend()


    #### Locall Preferential attachment
    #plt.figure()
    #plt.suptitle('Local Degree Distribution')
    #plt.subplot(2,2,1)
    #x = np.arange(1, len(max_local_attach_source)+1)
    #plt.loglog(x, max_local_attach_source, marker=marker, label='source')
    #x = np.arange(1, len(max_local_attach_learn)+1)
    #plt.loglog(x, max_local_attach_learn, marker=marker, label='learn')
    #plt.title('biggest')
    #plt.legend()
    #plt.subplot(2,2,2)
    #x = np.arange(1, len(rd_local_attach_source_1)+1)
    #plt.loglog(x, rd_local_attach_source_1, marker=marker, label='source')
    #x = np.arange(1, len(rd_local_attach_learn_1)+1)
    #plt.loglog(x, rd_local_attach_learn_1, marker=marker, label='learn')
    #plt.title('random class 1')
    #plt.legend()
    #plt.subplot(2,2,3)
    #x = np.arange(1, len(rd_local_attach_source_2)+1)
    #plt.loglog(x, rd_local_attach_source_2, marker=marker, label='source')
    #x = np.arange(1, len(rd_local_attach_learn_2)+1)
    #plt.loglog(x, rd_local_attach_learn_2, marker=marker, label='learn')
    #plt.title('random class 2')
    #plt.legend()
    #plt.subplot(2,2,4)
    #x = np.arange(1, len(min_local_attach_source)+1)
    #plt.loglog(x, min_local_attach_source, marker=marker, label='source')
    #x = np.arange(1, len(min_local_attach_learn)+1)
    #plt.loglog(x, min_local_attach_learn, marker=marker, label='learn')
    #plt.title('lowest')
    #plt.legend()

    ##for c, d in local_attach.items():
    ##    plt.subplot(len(local_attach), 1, int(c)+1)
    ##    plt.loglog(sorted(d, reverse=True))
    ##    plt.title('Communities ' + c)


    #### Analyse of phi and theta  ?
    #plt.figure()
    #plt.imshow(np.repeat(theta, 10, axis=1))
    #plt.title('theta')
    #plt.colorbar()

    #plt.figure()
    #plt.imshow(phi)
    #plt.title('phi')
    #plt.colorbar()

    display(True)

