#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

from frontend.manager import ModelManager, FrontendManager
from frontend.frontendnetwork import frontendNetwork
from utils.utils import *
from utils.math import *
from utils.algo import Annealing
from plot import *
from frontend.frontend_io import *
from expe.spec import _spec_; _spec = _spec_()
from expe import format
from utils.argparser import argparser

from collections import Counter, defaultdict

USAGE = '''\
# Usage:
    generate [-w] [-k K] [-n N] [--[hypername]] [-g|-p]] [analysis]

-g: generative model (evidence)
-p: predicted data (model fitted)

analysis in [clustering, zipf, (to complete)]

# Examples
    parallel ./generate.py -w -k {}  ::: $(echo 5 10 15 20)
    ./generate.py --alpha 1 --gmma 1 -n 1000 --seed
'''

#######################
### Config
#######################
config = defaultdict(lambda: False, dict(
    block_plot = False,
    write_to_file = False,
    do            = 'zipf',
    #generative    = 'evidence',
    generative    = 'predictive',
    gen_size      = 1000,
    epoch         = 20 , #20
    #### Path Spec
    debug         = 'debug11'
    #debug         = 'debug111111', repeat        = 5,
))
config.update(argparser.generate(USAGE))

#######################
### Corpuses
#######################
#Corpuses = _spec.CORPUS_SYN_ICDM_1
#Corpuses = _spec.CORPUS_REAL_ICDM_1
Corpuses = _spec.CORPUS_SYN_ICDM_1 + _spec.CORPUS_REAL_ICDM_1

#Corpuses = ('generator7',)
#Corpuses = ('fb_uc',)

alpha = 1; gmma = 1; delta = (1, 5)

#######################
### Models
#######################
#Models = _spec.MODELS_GENERATE
Models = [dict ((
    ('data_type'    , 'networks'),
    ('debug'        , 'debug11') , # ign in gen
    #('model'        , 'mmsb_cgs')   ,
    ('model'        , 'immsb')   ,
    ('K'            , 10)        ,
    ('N'            , 'all')     , # ign in gen
    ('hyper'        , 'auto')    , # ign in gen
    ('homo'         , 0)         , # ign in gen
    ('repeat'      , '')       ,
))]


# @debug: n value cause file crash
for k in Models[0].keys():
    if k in config:
        for m in Models:
            m[k] = config[k]

for opt in ('alpha','gmma', 'delta'):
    if config.get(opt):
        globals()[opt] = config[opt]

#  to get track of last experimentation in expe.format
nb_of_iteration = len(Corpuses) * len(Models) -1
_it = 0
for corpus_pos, corpus_name in enumerate(Corpuses):
    _end = _it == nb_of_iteration
    frontend = frontendNetwork(config)
    data = frontend.load_data(corpus_name)
    data = frontend.sample()

    lgg.info('---')
    lgg.info(_spec.name(corpus_name))
    lgg.info('---')

    for Model in Models:
        lgg.info(_spec.name(Model['model']))
        lgg.info('---')

        ###################################
        ### Setup Models
        ###################################
        if config['generative'] == 'predictive':
            ### Generate data from a fitted model
            Model.update(corpus=corpus_name)
            model = ModelManager(config=config).load(Model)
            #model = model.load(Model)
            try:
                # this try due to mthod modification entry in init not in picke object..
                Model['hyperparams'] = model.get_hyper()
            except:
                model._mean_w = 0
                Model['hyperparams'] = 0
            N = data.shape[0]
        elif config['generative'] == 'evidence':
            N = config['gen_size']
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
        else:
            raise NotImplementedError('What generation context ? evidence/generative..')

        if model is None:
            continue

        ###################################
        ### Generate data
        ###################################
        ### Defaut random graph (Evidence), is directed
        y, theta, phi = model.generate(N, Model['K'], _type=config['generative'])
        Y = [y]
        for i in range(config.get('epoch',1)-1):
            ### Mean and var on the networks generated
            pij = model.link_expectation(theta, phi)
            pij = np.clip(model.link_expectation(theta, phi), 0, 1)
            Y += [sp.stats.bernoulli.rvs(pij)]
            ### Mean and variance  on the model generated
            #y, theta, phi = model.generate(N, Model['K'], _type=config['generative'])
            #Y += [y]
        #y = data
        #Y = [y]

        ### @TODO: Baselines / put in args input.
        #R = rescal(data, config['K'])
        R = None

        N = theta.shape[0]
        K = theta.shape[1]
        if frontend.is_symmetric():
            for y in Y:
                frontend.symmetrize(y)
                frontend.symmetrize(R)

        ###################################
        ### Expe Show Setup
        ###################################
        model_name = Model['model']
        model_hyper = Model['hyperparams']
        lgg.info('=== M_e Mode === ')
        lgg.info('Expe: %s' % config['do'])
        lgg.info('Mode: %s' % config['generative'])
        lgg.info('corpus: %s, model: %s, K = %s, N =  %s, hyper: %s'.replace(',','\n') % (_spec.name(corpus_name), _spec.name(model_name), K, N, str(model_hyper)) )

        ###################################
        ### Visualize
        ###################################
        g = None # ?; remove !

        analysis = getattr(format, config['do'])
        analysis(**globals())

        #format.debug(**globals())

        _it += 1
        display(config['block_plot'])

if not config.get('write_to_file'):
    display(True)

