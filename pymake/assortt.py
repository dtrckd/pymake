#!/usr/bin/python -u
# -*- coding: utf-8 -*-

from util.utils import argParse
from frontend.manager import ModelManager, FrontendManager

from numpy import ma
import numpy as np
import scipy as sp
#np.set_printoptions(threshold='nan')

import logging
import os
import os.path

_USAGE = '''assort [-vhswp] [-k [rvalue] [-n N] [-d basedir] [-lall] [-l type] [-m model] [-c corpus] [-i iterations]

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
./assortt.py -n 1000 -k 10 --alpha auto --homo 0 -m ibp_cgs -c generator3 -l model --refdir debug5 -nld


'''

##################
###### MAIN ######
##################
if __name__ == '__main__':
    config = defaultdict(lambda: False, dict(
        ##### Global settings
        verbose                       = logging.INFO,
        host                          = 'localhost',
        index                         = 'search',
        ##### Input Features / Corpus
        corpus_name                   = 'kos',
        vsm                           = 'tf',
        limit_train                   = 10000,
        limit_predict                 = None,
        extra_feat                    = False,
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
        bdir                          = '../data',
        load_data                   = True,
        save_data                   = False,
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
    # Load model
    model = ModelManager(config=config)


    if config.get('load_model'):
        ### Generate data from a fitted model
        model = model.load()
    else:
        ### Generate data from a un-fitted model
        model = model.model

    d = frontend.assort(model)
    print d
    #frontend.update_json(d)


    ### Percentage of Homophily
#    # Source
#    sim_zeros_source = sim_source[data < 1]
#    simtest_source = np.ones((data==1).sum()) / (data < 1).sum()
#    for i, _1 in enumerate(zip(*np.where(data == 1))):
#        t = (sim_source[_1] > sim_zeros_source).sum()
#        simtest_source[i] *= t
#
#    # Learn
#    sim_zeros_learn = sim_learn[y < 1]
#    simtest_learn = np.ones((y==1).sum()) / (y < 1).sum()
#    for i, _1 in enumerate(zip(*np.where(y == 1))):
#        t = (sim_learn[_1] > sim_zeros_learn).sum()
#        simtest_learn[i] *= t
#
#    print 'Probability that link where 1sim is sup to 0sim:\n \
#            source:: mean: %f, var: %f,\n \
#            learn :: mean: %f, var: %f' % (simtest_source.mean(), simtest_source.var(), simtest_learn.mean(), simtest_learn.var())
#
    ### Plot the vector of probability
    #from plot import *
    #plt.figure()
    #plt.imshow(np.tile(simtest_source[:, np.newaxis], 100))
    #plt.title('Simtest Source')
    #plt.colorbar()

    #plt.figure()
    #plt.subplot(1,2,1)
    #plt.imshow(sim_source)
    #plt.title('Source Similarity')

    #plt.subplot(1,2,2)
    #plt.imshow(sim_learn)
    #plt.title('Learn Similarity')

    #plt.subplots_adjust(left=0.06, bottom=0.1, right=0.9, top=0.87)
    #cax = plt.axes([0.93, 0.15, 0.025, 0.7])
    #plt.colorbar(cax=cax)

    #plt.figure()
    #plt.imshow(phi)
    #plt.colorbar()



    #display(True)

