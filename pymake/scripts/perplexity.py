#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-

from pymake.frontend.manager import ModelManager, FrontendManager
from pymake.frontend.frontendnetwork import frontendNetwork
from pymake.util.utils import *
from pymake.util.math import *
from pymake.plot import *
from pymake.frontend.frontend_io import *
from pymake.expe.spec import _spec_; _spec = _spec_()
from pymake.expe.format import *
from pymake.util.argparser import argparser

import itertools

""" Perplexity convergence plots
"""

####################################################
### Config
config = dict(
    save_plot = False,
    gen_size      = 1000,
    epoch         = 10 , #20
)
config.update(argparser.generate(''))

# Corpuses
Corpuses = _spec.CORPUS_SYN_ICDM_1
Corpuses += _spec.CORPUS_REAL_ICDM_1
### Models
Models = _spec.MODELS_GENERATE


for m in Models:
    m['debug'] = 'debug11'

if config.get('K'):
    for m in Models:
        m['K'] = config['K']

for opt in ('alpha','gmma', 'delta'):
    if config.get(opt):
        globals()[opt] = config[opt]

for corpus_name in Corpuses:

    config['corpus'] = corpus_name

    plt.figure()
    for Model in Models:

        config.update(Model)
        frontend = frontendNetwork(config)

        ###################################
        ### Generate data from a fitted model
        ###################################
        Model.update(corpus=corpus_name)
        model = ModelManager(config=config).load(Model)

        # __future__ remove
        try:
            # this try due to mthod modification entry in init not in picke object..
            Model['hyperparams'] = model.get_hyper()
        except:
            model._mean_w = 0
            Model['hyperparams'] = 0

        if model is None:
            continue
        theta, phi = model.get_params()

        ###################################
        ### Expe Show Setup
        ###################################
        N = theta.shape[0]
        K = theta.shape[1]
        model_name = Model['model']
        model_hyper = Model['hyperparams']
        print('corpus: %s, model: %s, K = %s, N =  %s, hyper: %s'.replace(',','\n') % (corpus_name, model_name, K, N, str(model_hyper)) )

        ###################################
        ### Visualize
        ###################################
        perplexity(**globals())

    plt.xlabel('Iterations')
    plt.ylabel('Entropie')
    plt.legend(loc="upper right", prop={'size':10})
    plt.title('Perplexity: %s' % _spec.name(corpus_name))

    display(False)

if not config.get('save_plot'):
    display(True)

