#!/usr/bin/python -u
# -*- coding: utf-8 -*-

from frontend.manager import ModelManager, FrontendManager
from frontend.frontendnetwork import frontendNetwork
from utils.utils import *
from utils.math import *
from plot import *
from frontend.frontend_io import *
from expe.spec import _spec_; _spec = _spec_()
from expe.format import *
from utils.argparser import argparser

from collections import Counter, defaultdict
import itertools

""" Perplexity convergence plots
"""

####################################################
### Config
config = defaultdict(lambda: False, dict(
    write_to_file = False,
    gen_size      = 1000,
    epoch         = 10 , #20
))
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
        Model['hyperparams'] = model.get_hyper()

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
    plt.title('Perplexity: %s' % corpus_[corpus_name][0])

    display(False)

if not config.get('write_to_file'):
    display(True)

