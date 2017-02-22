#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

from pymake import Expe, ModelManager, FrontendManager, GramExp
from util.utils import Now, ellapsed_time
import logging
lgg = logging.getLogger('root')

import numpy as np
import scipy as sp
#np.set_printoptions(threshold='nan')

if __name__ == '__main__':

    ### Experience Settings
    g = GramExp(Expe(
        corpus = 'clique2',
        model  = 'immsb',
        hyper       = 'auto',
        refdir      = 'debug',
        testset_ratio = 0.2,
        K           = 3,
        N           = 42,
        chunk       = 10000,
        iterations  = 3,
        homo        = 0, # learn W in IBP
    ))

    lgg.info(g.exp_tensor.table())
    expe = g.lod[0]

    ### Load Data
    now = Now()
    frontend = FrontendManager.load(expe)
    last_d = ellapsed_time('Data Preprocessing Time', now)

    ### Load Model
    #models = ('ilda_cgs', 'lda_cgs', 'immsb', 'mmsb', 'ilfm_gs', 'lda_vb', 'ldafull_vb')
    # Hyperparameter
    alpha = expe.get('alpha', .1)
    gmma = expe.get('gmma', .5)
    delta = expe.get('delta', 1.)
    hyperparams = {'alpha': alpha, 'delta': delta, 'gmma': gmma}

    expe['hyperparams'] = hyperparams

    ### Debug
    #model = ModelManager(expe, frontend)
    #model.initialization_test()
    #exit()

    ### Initializa Model
    model = ModelManager(expe)
    last_d = ellapsed_time('Init Model Time', last_d)

    ### Run Inference / Learning Model
    model.fit(frontend)
    last_d = ellapsed_time('Inference Time: %s'%(model.output_path), last_d)

    ### Predict Future
    # @debug remove frontend...
    model.predict(frontend=frontend)

