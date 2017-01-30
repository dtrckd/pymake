#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-

from frontend.manager import ModelManager, FrontendManager
from frontend.frontendnetwork import frontendNetwork
from util.utils import *
from util.math import *
from plot import *
from expe.spec import _spec_; _spec = _spec_()
from expe.format import *
from util.argparser import argparser

from collections import Counter, defaultdict
import itertools

""" AUC-ROC analysis on test data
"""

####################################################
### Config
config = defaultdict(lambda: False, dict(
    load_data = True
    write_to_file = False,
    gen_size      = 1000,
    epoch         = 10 , #20
))
config.update(argparser.generate(''))

alpha = 0.5
gmma = 0.5
delta = 0.1
delta = (0.1, 0.1)

# Corpuses
Corpuses = _spec.CORPUS_SYN_ICDM_1
Corpuses += _spec.CORPUS_REAL_ICDM_1
### Models
Models = _spec.MODELS_GENERATE

#Models = [dict ((
#    ('data_type'    , 'networks'),
#    ('debug'        , 'debug11') , # ign in gen
#    #('model'        , 'mmsb_cgs')   ,
#    ('model'        , 'immsb')   ,
#    ('K'            , 10)        ,
#    ('N'            , 'all')     , # ign in gen
#    ('hyper'        , 'auto')    , # ign in ge
#    ('homo'         , 0)         , # ign in ge
#    #('repeat'      , '*')       ,
#))]

for m in Models:
    m['debug'] = 'debug111111'
    m['repeat'] = 5

if config.get('K'):
    for m in Models:
        m['K'] = config['K']

for opt in ('alpha','gmma', 'delta'):
    if config.get(opt):
        globals()[opt] = config[opt]

for corpus_name in Corpuses:
    frontend = frontendNetwork(config)
    data = frontend.load_data(corpus_name)
    data = frontend.sample()

    plt.figure()
    for Model in Models:

        ###################################
        ### Generate data from a fitted model
        ###################################
        Model.update(corpus=corpus_name)
        model = ModelManager(config=config).load(Model)
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
        roc_test(**globals())

    plt.plot([0, 1], [0, 1], linestyle='--', color='k', label='Luck')
    plt.legend(loc="lower right", prop={'size':10})
    plt.title(_spec.name(corpus_name))

    display(False)

if not config.get('write_to_file'):
    display(True)

