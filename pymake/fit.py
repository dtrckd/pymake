#!/usr/bin/python -u
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

from frontend.manager import ModelManager, FrontendManager
from utils.utils import *

import numpy as np
import scipy as sp
#np.set_printoptions(threshold='nan')

import logging

USAGE = '''build_model [-vhswp] [-k [rvalue] [-n N] [-d basedir] [-lall] [-l type] [-m model] [-c corpus] [-i iterations]

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
-r | --random [type] : Generate a random networ for training
--homo int     : homophily 0:default, 1: ridge, 2: smooth
##### Single argument
-p           : Do prediction on test data
-s           : Simulate output
-w|-nw           : Write/noWrite convergence measures (Perplexity etc)
-h | --help  : Command Help
-v           : Verbosity level

Examples:
# Load corpus and infer modef (eg LDA)
./lda_run.py -k 6 -m ldafullbaye -p:
# Load corpus and model:
./lda_run.py -k 6 -m ldafullbaye -lall -p
# Network corpus:
./fit.py -m immsb -c generator1 -n 100 -i 10
# Various networks setting:
./fit.py -m ibp_cgs --homo 0 -c clique6 -n 100 -k 3 -i 20


'''

if __name__ == '__main__':
    config = defaultdict(lambda: False, dict(
        ##### Global settings
        verbose             = logging.INFO,
        host                = 'localhost',
        index               = 'search',
        ###### I/O settings
        refdir              = 'debug',
        load_data           = True,
        save_data           = False,
        load_model          = False,
        save_model          = True,
        write               = False, # -w/-nw
        #####
        predict             = False,
        repeat      = False,
    ))
    ##### Experience Settings
    Expe = dict(
        corpus_name = 'clique2',
        model_name  = 'immsb',
        hyper       = 'auto',
        K           = 3,
        N           = 10,
        chunk       = 10000,
        iterations  = 2,
        homo        = 0, # learn W in IBP
    )

    config.update(Expe)
    config.update(argParse(USAGE))

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
        print('''--- Simulation settings ---
        Model : %s
        Corpus : %s
        K : %s
        N : %s
        hyper : %s
        Output : %s''' % (config['model'], config['corpus_name'],
                         config['K'], config['N'], config['hyper'],
                         config['output_path']))
        exit()

    ############################################################
    ##### Load Data
    frontend = FrontendManager.get(config)

    now = datetime.now()
    data = frontend.load_data(randomize=False)
    data = frontend.sample()
    last_d = ellapsed_time('Data Preprocessing Time', now)

    if 'Text' in str(type(frontend)):
        lgg.warning('check WHY and WHEN overflow in stirling matrix !?')
        print('debug why error and i get walue superior to 6000 in the striling matrix ????')
        data, data_t = frontend.cross_set(ratio=0.8)
    elif 'Network' in str(type(frontend)):
        if config['refdir'] in ('debug11', 'debug1111','debug111111'):
            # Random training set on 20% on Data / debug5 - debug11 -- Unbalanced
            percent_hole = 0.2
            data = frontend.get_masked(percent_hole)
            config['symmetric'] = frontend.is_symmetric()
            data_t = None
        elif config['refdir']  in ('debug10', 'debug1010', 'debug101010'):
            # Random training set on 20% on Data vertex (0.2 * data == 1) / debug6 - debug 10 -- Balanced
            percent_hole = 0.2
            data = frontend.get_masked_1(percent_hole)
            config['symmetric'] = frontend.is_symmetric()
            data_t = None
        else:
            # Random training set on 20% on Data / debug5 - debug11
            percent_hole = 0.2
            data = frontend.get_masked(percent_hole)
            config['symmetric'] = frontend.is_symmetric()
            data_t = None

        #print frontend.nodes_list
    else:
        raise ValueError('Unknow data type')

    ############################################################
    ##### Load Model
    #models = ('ilda_cgs', 'lda_cgs', 'immsb', 'mmsb', 'ilfm_gs', 'lda_vb', 'ldafull_vb')
    # Hyperparameter
    delta = .1
    # Those are sampled
    alpha = .5
    gmma = 1.
    hyperparams = {'alpha': alpha, 'delta': delta, 'gmma': gmma}

    config['hyperparams'] = hyperparams

    #### Debug
    #config['write'] = False
    #model = ModelManager(data, config)
    #model.init_loop_test()
    #exit()

    # Initializa Model
    model = ModelManager(data, config, data_t=data_t)
    last_d = ellapsed_time('Init Model Time', last_d)

    #### Run Inference / Learning Model
    model.fit()
    last_d = ellapsed_time('Inference Time: %s'%(model.output_path), last_d)

    #### Predict Future
    model.predict(frontend)

