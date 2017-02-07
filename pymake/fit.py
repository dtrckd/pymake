#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

from frontend.manager import ModelManager, FrontendManager
from util.argparser import argparser
from util.utils import Now, ellapsed_time

import numpy as np
import scipy as sp
#np.set_printoptions(threshold='nan')

if __name__ == '__main__':
    ##### Experience Settings
    exp = dict(
        corpus_name = 'clique2',
        #corpus = "lucene"
        model_name  = 'immsb',
        hyper       = 'auto',
        testset_ratio = 0.2,
        K           = 3,
        N           = 42,
        chunk       = 10000,
        iterations  = 6,
        homo        = 0, # learn W in IBP
    )

    exp.update(argparser.gramexp())

    argparser.simulate(exp)


    ############################################################
    ##### Load Data
    frontend = FrontendManager.get(exp)
    now = Now()
    frontend.load_data(randomize=False)
    frontend.sample(exp['N'])
    last_d = ellapsed_time('Data Preprocessing Time', now)

    ############################################################
    ##### Load Model
    #models = ('ilda_cgs', 'lda_cgs', 'immsb', 'mmsb', 'ilfm_gs', 'lda_vb', 'ldafull_vb')
    # Hyperparameter
    delta = .1
    # Those are sampled
    alpha = .5
    gmma = 1.
    hyperparams = {'alpha': alpha, 'delta': delta, 'gmma': gmma}

    exp['hyperparams'] = hyperparams

    #### Debug
    #model = ModelManager(exp, frontend)
    #model.initialization_test()
    #exit()

    # Initializa Model
    model = ModelManager(exp)
    last_d = ellapsed_time('Init Model Time', last_d)

    #### Run Inference / Learning Model
    model.fit(frontend)
    last_d = ellapsed_time('Inference Time: %s'%(model.output_path), last_d)

    #### Predict Future
    # @debug remove frontend...
    model.predict(frontend=frontend)

