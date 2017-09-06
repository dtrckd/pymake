#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals


from pymake import ExpTensor, ModelManager, FrontendManager, GramExp

from expe.spec import _spec_; _spec = _spec_()
from expe import format

import logging
lgg = logging.getLogger('root')


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


Corpuses = _spec.CORPUS_NET_ALL
Corpuses = _spec.CORPUS_REAL_V2

#Models = _spec.MODELS_GENERATE
Spec = ExpTensor ((
    ('corpus', Corpuses),
    ('data_type'    , 'networks'),
    ('debug'        , 'debug11') , # ign in gen
    #('model'        , 'mmsb_cgs')   ,
    ('model'        , 'immsb')   ,
    ('K'            , 10)        ,
    ('N'            , 'all')     , # ign in gen
    ('hyper'        , 'auto')    , # ign in gen
    ('homo'         , 0)         , # ign in gen
    ('_repeat'      , '')       ,
    #
    ('alpha', 1),
    ('gmma', 1),
    ('delta', [(1, 5)]),
))

config = dict(
    block_plot = False,
    save_plot  = False,
    do            = 'zipf',
    #generative    = 'evidence',
    mode    = 'predictive',
    gen_size      = 1000,
    epoch         = 20 , #20
    #### Path Spec
    #debug         = 'debug11'
    debug         = 'debug111111',
    _repeat        = 0,
    spec = Spec
)


def generate(pt, expe, gramexp):
    _it = pt['expe']
    corpus_pos = pt['corpus']
    model_pos = pt['model']

    #  to get track of last experimentation in expe.format
    frontend = FrontendManager.load(expe)

    lgg.info('---')
    lgg.info('Expe : %s -- %s' % (_spec.name(expe.corpus), _spec.name(expe.model)))
    lgg.info('---')

    _end = _it == (len(gramexp)-1)

    if expe.mode == 'predictive':
        ### Generate data from a fitted model
        model = ModelManager.from_expe(expe)

        # __future__ remove
        try:
            # this try due to mthod modification entry in init not in picke object..
            expe.hyperparams = model.get_hyper()
        except:
            if model is None:
                lgg.error('No model for %s' % pt)
                return
            else:
                model._mean_w = 0
                expe.hyperparams = 0

        N = frontend.data.shape[0]
    elif expe['generative'] == 'evidence':
        N = expe.gen_size
        ### Generate data from a un-fitted model
        if expe.model == 'ibp':
            keys_hyper = ('alpha','delta')
            hyper = (alpha, delta)
        else:
            keys_hyper = ('alpha','gmma','delta')
            hyper = (alpha, gmma, delta)
        expe.hyperparams = dict(zip(keys_hyper, hyper))
        expe.hyper = 'fix' # dummy
        model = ModelManager.from_expe(expe, init=True)
        #model.update_hyper(hyper)
    else:
        raise NotImplementedError('What generation context ? evidence/generative..')

    if model is None:
        return

    ###################################
    ### Generate data
    ###################################
    ### Defaut random graph (Evidence), is directed
    y, theta, phi = model.generate(N, expe.K, _type=expe.mode)
    Y = [y]
    for i in range(expe.epoch - 1):
        ### Mean and var on the networks generated
        pij = model.likelihood(theta, phi)
        pij = np.clip(model.likelihood(theta, phi), 0, 1)
        Y += [sp.stats.bernoulli.rvs(pij)]
        ### Mean and variance  on the model generated
        #y, theta, phi = model.generate(N, Model['K'], _type=expe['generative'])
        #Y += [y]
    #y = data
    #Y = [y]

    ### @TODO: Baselines / put in args input.
    #R = rescal(data, expe['K'])
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
    model_hyper = Model['hyperparams']
    lgg.info('=== M_e Mode === ')
    lgg.info('Expe: %s' % expe.do)
    lgg.info('Mode: %s' % expe.model)
    lgg.info('corpus: %s, model: %s, K = %s, N =  %s, hyper: %s'.replace(',','\n') % (_spec.name(expe.corpus), _spec.name(expe.model), K, N, str(model_hyper)) )

    ###################################
    ### Visualize
    ###################################
    g = None # ?; remove !

    analysis = getattr(format, expe.do)
    analysis(**globals())

    #format.debug(**globals())

    _it += 1
    display(expe.block_plot)


    # arggg, nasty
    # How to call it ?
    if not expe.save_plot:
        display(True)


if __name__ == '__main__':
    GramExp(config, USAGE).pymake(generate)

