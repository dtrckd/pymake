#!/usr/bin/python -u
# -*- coding: utf-8 -*-

from frontend.manager import ModelManager, FrontendManager
from util.utils import *

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

    ############################################################
    ##### Load Data
    frontend = FrontendManager.get(config)

    now = datetime.now()
    data = frontend.load_data(randomize=False)
    data = frontend.sample()
    last_d = ellapsed_time('Data Preprocessing Time', now)

    # Initializa Model
    model = ModelManager(data, config)
    last_d = ellapsed_time('Init Model Time', last_d)

    #### Run Inference / Learning Model
    model.fit()
    last_d = ellapsed_time('Inference Time: %s'%(model.output_path), last_d)

    #### Predict Future
    model.predict(frontend)

