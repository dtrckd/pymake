#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-

from frontend.manager import ModelManager, FrontendManager
from frontend.frontendnetwork import frontendNetwork
from util.utils import *
from util.math import *
from plot import *
from frontend.frontend_io import *
from expe.spec import _spec
from expe.format import *
from util.argparser import argparser

import itertools

""" Density - Small world asymptotic analysis
"""

####################################################
### Config
config = dict(
    save_plot = False,
    gen_size      = 1000,
    mode    = 'generative',
    epoch         = 10 , #20
)
config.update(argparser.generate(''))

# Corpuses
Corpuses = _spec.CORPUS_SYN_ICDM_1
Corpuses += _spec.CORPUS_REAL_ICDM_1
### Models
Models = _spec.MODELS_GENERATE

Corpuses = ('generator7',)
Models = [{'model':'immsb'}]
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
    m['debug'] = 'debug11'

if config.get('K'):
    for m in Models:
        m['K'] = config['K']

for opt in ('alpha','gmma', 'delta'):
    if config.get(opt):
        globals()[opt] = config[opt]

delta = (0.5, 0.5)

Hyper = [ (0.5, 0.5), (1., 1.), (0.5,3.), (3., .5),]
n = 500
_N = np.arange(2, n)
K = np.zeros(n-1)

for corpus_name in Corpuses:
    #frontend = frontendNetwork(config)
    #data = frontend.load_data(corpus_name)
    #data = frontend.sample()

    for Model in Models:

        curves = []
        plt.figure()
        for alpha, gmma in Hyper:
            for j, N in enumerate(_N):

                ### Generate data from a unfitted model
                #N = config['gen_size']
                ### Generate data from a un-fitted model
                if Model['model'] == 'ibp':
                    keys_hyper = ('alpha','delta')
                    hyper = (alpha, delta)
                else:
                    keys_hyper = ('alpha','gmma','delta')
                    hyper = (alpha, gmma, delta)
                Model['hyperparams'] = dict(zip(keys_hyper, hyper))
                Model['hyper'] = 'fix' # dummy
                model = ModelManager(config=config).load(Model, init=True)
                #model.update_hyper(hyper)
                if model is None:
                    continue

                ### Class dynamics
                k = model.limit_k(N)
                K[j] = k
                ttle = 'Class dynamics %s' % Model['model']

                ### Density
                #y, theta, phi = model.generate(N, mode=config['generative'])
                #K[j] = float(y.sum()) / (N**2)
                #ttle = 'density %s' % Model['model']

            #np.save('../results/networks/aistat16/limit_k', K)
            plt.plot(K, label='alpha=%s, gamma=%s'%(alpha, gmma))
        plt.legend(loc="upper left", prop={'size':10})
        plt.title(ttle)

        display(False)

if not config.get('save_plot'):
    display(True)

